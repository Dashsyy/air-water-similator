/**
 * D3Q19 Lattice Boltzmann solver on WebGPU.
 *
 * - Pull streaming with interpolated (Bouzidi) bounce-back on the model, which places the wall at
 *   its true sub-cell position, and halfway bounce-back on tunnel walls.
 * - Regularised BGK collision with a Smagorinsky sub-grid model, for stability and low noise at
 *   high Reynolds numbers.
 * - Populations are stored shifted by their lattice weight (f - w) to keep float32 precision.
 * - Aerodynamic force on the solid comes from the momentum exchange method and is
 *   accumulated on the GPU in fixed point, then read back asynchronously.
 *
 * World and lattice coordinates coincide: cell (i, j, k) occupies [i, i+1] x [j, j+1] x [k, k+1].
 * The wind blows along +X, +Y is up.
 */

export const CELL_FLUID = 0;
/** Part of the model: no-slip, and its surface forces are measured. */
export const CELL_SOLID = 1;
/** Tunnel wall such as the floor: no-slip, but excluded from the force measurement. */
export const CELL_WALL = 2;

const FORCE_SCALE = 1 << 20;

const LBM_COMMON = /* wgsl */ `
struct Params {
  dims: vec4<u32>,   // nx, ny, nz, cell count
  flow: vec4<f32>,   // inflow speed, tau0, force scale, interpolated walls on/off
  tune: vec4<f32>,   // eddy-viscosity factor next to walls, unused
};

const C = array<vec3<i32>, 19>(
  vec3<i32>(0, 0, 0),
  vec3<i32>(1, 0, 0), vec3<i32>(-1, 0, 0),
  vec3<i32>(0, 1, 0), vec3<i32>(0, -1, 0),
  vec3<i32>(0, 0, 1), vec3<i32>(0, 0, -1),
  vec3<i32>(1, 1, 0), vec3<i32>(-1, -1, 0),
  vec3<i32>(1, -1, 0), vec3<i32>(-1, 1, 0),
  vec3<i32>(1, 0, 1), vec3<i32>(-1, 0, -1),
  vec3<i32>(1, 0, -1), vec3<i32>(-1, 0, 1),
  vec3<i32>(0, 1, 1), vec3<i32>(0, -1, -1),
  vec3<i32>(0, 1, -1), vec3<i32>(0, -1, 1),
);

const W = array<f32, 19>(
  1.0 / 3.0,
  1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0,
  1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
  1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
  1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
);

const OPP = array<u32, 19>(0u, 2u, 1u, 4u, 3u, 6u, 5u, 8u, 7u, 10u, 9u, 12u, 11u, 14u, 13u, 16u, 15u, 18u, 17u);

// Equilibrium population minus its lattice weight. rho1 is (rho - 1).
fn feqShifted(q: u32, rho1: f32, u: vec3<f32>) -> f32 {
  let cu = dot(vec3<f32>(C[q]), u);
  return W[q] * (rho1 + (1.0 + rho1) * (3.0 * cu + 4.5 * cu * cu - 1.5 * dot(u, u)));
}
`;

const STEP_SHADER = LBM_COMMON + /* wgsl */ `
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<storage, read> flags: array<u32>;
@group(0) @binding(4) var macroTex: texture_storage_3d<rgba16float, write>;
@group(0) @binding(5) var<storage, read_write> force: array<atomic<i32>, 4>;
@group(0) @binding(6) var<storage, read> wallDistance: array<f32>;

fn cellIndex(p: vec3<i32>) -> u32 {
  return u32(p.x) + P.dims.x * (u32(p.y) + P.dims.y * u32(p.z));
}

fn density1(idx: u32) -> f32 {
  let n = P.dims.w;
  var r = 0.0;
  for (var q = 0u; q < 19u; q++) { r += src[q * n + idx]; }
  return r;
}

fn velocityAt(idx: u32) -> vec3<f32> {
  let n = P.dims.w;
  var r = 0.0;
  var m = vec3<f32>(0.0);
  for (var q = 0u; q < 19u; q++) {
    let v = src[q * n + idx];
    r += v;
    m += v * vec3<f32>(C[q]);
  }
  return m / (1.0 + r);
}

@compute @workgroup_size(4, 4, 4)
fn init(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (any(gid >= P.dims.xyz)) { return; }
  let n = P.dims.w;
  let idx = cellIndex(vec3<i32>(gid));
  let solid = flags[idx] != 0u;
  let u = vec3<f32>(P.flow.x, 0.0, 0.0);
  for (var q = 0u; q < 19u; q++) {
    dst[q * n + idx] = select(feqShifted(q, 0.0, u), 0.0, solid);
  }
  textureStore(macroTex, vec3<i32>(gid), vec4<f32>(select(u, vec3<f32>(0.0), solid), 0.0));
}

@compute @workgroup_size(4, 4, 4)
fn step(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (any(gid >= P.dims.xyz)) { return; }
  let dims = P.dims.xyz;
  let n = P.dims.w;
  let p = vec3<i32>(gid);
  let idx = cellIndex(p);
  let uIn = vec3<f32>(P.flow.x, 0.0, 0.0);

  let kind = flags[idx];

  // Tunnel wall, for example a fixed floor under a grounded model.
  if (kind == 2u) {
    for (var q = 0u; q < 19u; q++) { dst[q * n + idx] = 0.0; }
    textureStore(macroTex, p, vec4<f32>(0.0));
    return;
  }

  // Free-stream boundary on the inlet and the four side faces.
  if (gid.x == 0u || gid.y == 0u || gid.z == 0u || gid.y == dims.y - 1u || gid.z == dims.z - 1u) {
    for (var q = 0u; q < 19u; q++) { dst[q * n + idx] = feqShifted(q, 0.0, uIn); }
    textureStore(macroTex, p, vec4<f32>(uIn, 0.0));
    return;
  }

  // Solid cell: hold it at rest and extrapolate surface pressure from fluid neighbours.
  if (kind == 1u) {
    for (var q = 0u; q < 19u; q++) { dst[q * n + idx] = 0.0; }
    var sum = 0.0;
    var count = 0.0;
    for (var q = 1u; q < 19u; q++) {
      let nidx = cellIndex(p + C[q]);
      if (flags[nidx] == 0u) {
        sum += density1(nidx);
        count += 1.0;
      }
    }
    textureStore(macroTex, p, vec4<f32>(0.0, 0.0, 0.0, sum / max(count, 1.0)));
    return;
  }

  // Outlet: constant pressure, velocity taken from the upstream neighbour.
  if (gid.x == dims.x - 1u) {
    var uOut = velocityAt(idx - 1u);
    uOut.x = max(uOut.x, 0.0);
    for (var q = 0u; q < 19u; q++) { dst[q * n + idx] = feqShifted(q, 0.0, uOut); }
    textureStore(macroTex, p, vec4<f32>(uOut, 0.0));
    return;
  }

  // Streaming (pull). A population that would arrive from inside a solid is reflected instead.
  var f: array<f32, 19>;
  var bodyForce = vec3<f32>(0.0);
  var touchesSolid = false;
  var nearWall = false;
  let dHere = wallDistance[idx];
  for (var q = 0u; q < 19u; q++) {
    let sidx = cellIndex(p - C[q]);
    let neighbour = flags[sidx];
    if (neighbour == 0u) {
      f[q] = src[q * n + sidx];
      continue;
    }
    let o = OPP[q];
    let outgoing = src[o * n + idx];
    var v = outgoing;
    if (neighbour == 1u) {
      nearWall = true;
      // Fraction of the link between this cell centre and the wall.
      let dWall = wallDistance[sidx];
      var delta = 0.5;
      if (P.flow.w > 0.5 && dHere >= 0.0 && dWall >= 0.0 && dHere + dWall > 1e-4) {
        delta = clamp(dHere / (dHere + dWall), 0.05, 1.0);
      }
      if (delta < 0.5) {
        let bidx = cellIndex(p + C[q]);
        if (flags[bidx] == 0u) {
          v = 2.0 * delta * outgoing + (1.0 - 2.0 * delta) * src[o * n + bidx];
        }
      } else {
        v = (outgoing + (2.0 * delta - 1.0) * src[q * n + idx]) / (2.0 * delta);
      }
      // Momentum exchange: what left towards the wall plus what came back.
      bodyForce -= (outgoing + v) * vec3<f32>(C[q]);
      touchesSolid = true;
    }
    f[q] = v;
  }

  var rho1 = 0.0;
  var mom = vec3<f32>(0.0);
  for (var q = 0u; q < 19u; q++) {
    rho1 += f[q];
    mom += f[q] * vec3<f32>(C[q]);
  }
  let rho = 1.0 + rho1;
  let u = mom / rho;

  // Non-equilibrium stress for the Smagorinsky eddy viscosity.
  var feq: array<f32, 19>;
  var sxx = 0.0; var syy = 0.0; var szz = 0.0;
  var sxy = 0.0; var sxz = 0.0; var syz = 0.0;
  for (var q = 0u; q < 19u; q++) {
    let e = feqShifted(q, rho1, u);
    feq[q] = e;
    let d = f[q] - e;
    let c = vec3<f32>(C[q]);
    sxx += d * c.x * c.x; syy += d * c.y * c.y; szz += d * c.z * c.z;
    sxy += d * c.x * c.y; sxz += d * c.x * c.z; syz += d * c.y * c.z;
  }
  let stress = sqrt(sxx * sxx + syy * syy + szz * szz + 2.0 * (sxy * sxy + sxz * sxz + syz * syz));
  let tau0 = P.flow.y;
  // 0.76421222 = 18 * sqrt(2) * Cs^2 with Cs = 0.173
  // Plain Smagorinsky overestimates eddy viscosity in the first cell off a wall, which shows up as
  // far too much skin friction. Damp it there.
  let eddy = select(1.0, P.tune.x, nearWall);
  let omega = 2.0 / (tau0 + sqrt(tau0 * tau0 + 0.76421222 * eddy * stress / rho));

  // Regularised collision: rebuild the non-equilibrium part from its stress tensor alone. This drops
  // the non-hydrodynamic modes that otherwise show up as checkerboard pressure noise at low viscosity.
  let keep = 1.0 - omega;
  for (var q = 0u; q < 19u; q++) {
    let c = vec3<f32>(C[q]);
    let neq = 4.5 * W[q] * (
      (c.x * c.x - 1.0 / 3.0) * sxx + (c.y * c.y - 1.0 / 3.0) * syy + (c.z * c.z - 1.0 / 3.0) * szz
      + 2.0 * (c.x * c.y * sxy + c.x * c.z * sxz + c.y * c.z * syz));
    dst[q * n + idx] = feq[q] + keep * neq;
  }
  textureStore(macroTex, p, vec4<f32>(u, rho1));

  if (touchesSolid) {
    let scaled = vec3<i32>(round(bodyForce * P.flow.z));
    atomicAdd(&force[0], scaled.x);
    atomicAdd(&force[1], scaled.y);
    atomicAdd(&force[2], scaled.z);
  }
}
`;

export interface ForceSample {
  /** Mean force on the body over the sampled steps, in lattice units (drag, lift, side). */
  force: [number, number, number];
  /** Number of solver steps the mean covers. */
  steps: number;
  /** Total solver steps completed when the sample ended. */
  atStep: number;
}

export class Lbm3D {
  readonly nx: number;
  readonly ny: number;
  readonly nz: number;
  readonly cells: number;
  readonly macroTex: GPUTexture;
  readonly solidTex: GPUTexture;

  totalSteps = 0;
  onForce: ((sample: ForceSample) => void) | null = null;

  private readonly device: GPUDevice;
  private readonly pops: [GPUBuffer, GPUBuffer];
  private readonly flagBuf: GPUBuffer;
  private readonly forceBuf: GPUBuffer;
  private readonly distanceBuf: GPUBuffer;
  private readonly readback: GPUBuffer;
  private readonly uniform: GPUBuffer;
  private readonly stepPipeline: GPUComputePipeline;
  private readonly initPipeline: GPUComputePipeline;
  private readonly bindGroups: [GPUBindGroup, GPUBindGroup];
  private current = 0;
  private stepsSinceReadback = 0;
  private readbackPending = false;
  private generation = 0;
  private destroyed = false;
  private u0 = 0.06;
  private tau = 0.51;
  private interpolatedWalls = false;
  private wallEddyFactor = 0.15;

  constructor(device: GPUDevice, nx: number, ny: number, nz: number) {
    this.device = device;
    this.nx = nx;
    this.ny = ny;
    this.nz = nz;
    this.cells = nx * ny * nz;

    const popBytes = this.cells * 19 * 4;
    const makePops = (): GPUBuffer => device.createBuffer({ size: popBytes, usage: GPUBufferUsage.STORAGE });
    this.pops = [makePops(), makePops()];
    this.flagBuf = device.createBuffer({
      size: this.cells * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.distanceBuf = device.createBuffer({
      size: this.cells * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.forceBuf = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.readback = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    this.uniform = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    this.macroTex = device.createTexture({
      size: [nx, ny, nz],
      dimension: '3d',
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.solidTex = device.createTexture({
      size: [nx, ny, nz],
      dimension: '3d',
      format: 'r8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });

    const module = device.createShaderModule({ code: STEP_SHADER, label: 'lbm3d' });
    const layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        {
          binding: 4,
          visibility: GPUShaderStage.COMPUTE,
          storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '3d' },
        },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      ],
    });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    this.stepPipeline = device.createComputePipeline({
      layout: pipelineLayout,
      compute: { module, entryPoint: 'step' },
    });
    this.initPipeline = device.createComputePipeline({
      layout: pipelineLayout,
      compute: { module, entryPoint: 'init' },
    });

    const macroView = this.macroTex.createView();
    const makeGroup = (src: GPUBuffer, dst: GPUBuffer): GPUBindGroup =>
      device.createBindGroup({
        layout,
        entries: [
          { binding: 0, resource: { buffer: this.uniform } },
          { binding: 1, resource: { buffer: src } },
          { binding: 2, resource: { buffer: dst } },
          { binding: 3, resource: { buffer: this.flagBuf } },
          { binding: 4, resource: macroView },
          { binding: 5, resource: { buffer: this.forceBuf } },
          { binding: 6, resource: { buffer: this.distanceBuf } },
        ],
      });
    this.bindGroups = [makeGroup(this.pops[0], this.pops[1]), makeGroup(this.pops[1], this.pops[0])];
    this.writeUniform();
  }

  /** Lattice inflow speed and BGK relaxation time. */
  setFlow(u0: number, tau: number): void {
    this.u0 = u0;
    this.tau = tau;
    this.writeUniform();
  }

  /** Scales the sub-grid eddy viscosity in cells that touch a wall. 1 leaves it untouched. */
  setWallEddyFactor(factor: number): void {
    this.wallEddyFactor = factor;
    this.writeUniform();
  }

  get inflowSpeed(): number {
    return this.u0;
  }

  /**
   * Upload a new cell map. `flags` holds CELL_FLUID, CELL_SOLID or CELL_WALL per cell.
   * With `distance` (cells to the surface, -1 where unknown) walls are placed at sub-cell positions.
   */
  setSolid(flags: Uint32Array, distance?: Float32Array): void {
    this.device.queue.writeBuffer(this.flagBuf, 0, flags.buffer, flags.byteOffset, flags.byteLength);
    this.interpolatedWalls = distance !== undefined;
    if (distance) {
      this.device.queue.writeBuffer(this.distanceBuf, 0, distance.buffer, distance.byteOffset, distance.byteLength);
    }
    this.writeUniform();
    const mask = new Uint8Array(this.cells);
    for (let i = 0; i < this.cells; i++) mask[i] = flags[i] === CELL_SOLID ? 255 : 0;
    this.device.queue.writeTexture(
      { texture: this.solidTex },
      mask,
      { bytesPerRow: this.nx, rowsPerImage: this.ny },
      [this.nx, this.ny, this.nz],
    );
    this.discardForce();
  }

  /** Restart the flow from a uniform free stream. */
  reset(): void {
    const encoder = this.device.createCommandEncoder();
    // Initialise both buffers so either can act as the source.
    for (const group of this.bindGroups) {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.initPipeline);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(...this.workgroups());
      pass.end();
    }
    this.device.queue.submit([encoder.finish()]);
    this.current = 0;
    this.totalSteps = 0;
    this.discardForce();
  }

  /** Advance the flow by `steps` time steps. */
  advance(steps: number): void {
    if (this.destroyed || steps <= 0) return;
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.stepPipeline);
    const groups = this.workgroups();
    for (let i = 0; i < steps; i++) {
      pass.setBindGroup(0, this.bindGroups[this.current]);
      pass.dispatchWorkgroups(...groups);
      this.current = 1 - this.current;
    }
    pass.end();
    this.totalSteps += steps;
    this.stepsSinceReadback += steps;

    const takeSample = !this.readbackPending && this.stepsSinceReadback >= 8;
    if (takeSample) {
      encoder.copyBufferToBuffer(this.forceBuf, 0, this.readback, 0, 16);
      encoder.clearBuffer(this.forceBuf);
    }
    this.device.queue.submit([encoder.finish()]);

    if (takeSample) {
      const steps = this.stepsSinceReadback;
      const atStep = this.totalSteps;
      const generation = this.generation;
      this.stepsSinceReadback = 0;
      this.readbackPending = true;
      this.readback
        .mapAsync(GPUMapMode.READ)
        .then(() => {
          const raw = new Int32Array(this.readback.getMappedRange().slice(0));
          this.readback.unmap();
          this.readbackPending = false;
          if (generation !== this.generation || !this.onForce) return;
          const k = 1 / (FORCE_SCALE * steps);
          this.onForce({ force: [raw[0] * k, raw[1] * k, raw[2] * k], steps, atStep });
        })
        .catch(() => {
          this.readbackPending = false;
        });
    }
  }

  destroy(): void {
    this.destroyed = true;
    this.onForce = null;
    for (const b of [...this.pops, this.flagBuf, this.distanceBuf, this.forceBuf, this.uniform]) b.destroy();
    this.macroTex.destroy();
    this.solidTex.destroy();
  }

  /** Drop force accumulated so far, for example after the geometry changed. */
  private discardForce(): void {
    this.generation++;
    this.stepsSinceReadback = 0;
    const encoder = this.device.createCommandEncoder();
    encoder.clearBuffer(this.forceBuf);
    this.device.queue.submit([encoder.finish()]);
  }

  private workgroups(): [number, number, number] {
    return [Math.ceil(this.nx / 4), Math.ceil(this.ny / 4), Math.ceil(this.nz / 4)];
  }

  private writeUniform(): void {
    const data = new ArrayBuffer(48);
    new Float32Array(data, 32, 4).set([this.wallEddyFactor, 0, 0, 0]);
    new Uint32Array(data, 0, 4).set([this.nx, this.ny, this.nz, this.cells]);
    new Float32Array(data, 16, 4).set([this.u0, this.tau, FORCE_SCALE, this.interpolatedWalls ? 1 : 0]);
    this.device.queue.writeBuffer(this.uniform, 0, data);
  }
}
