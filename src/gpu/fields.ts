/**
 * Derived fields used for visualisation, all computed on the GPU from the solver's velocity texture:
 * streakline particles released from a rake, an advected smoke density, and the Q-criterion.
 */
import type { Lbm3D } from './lbm3d';

export interface Rake {
  /** Release plane position along the tunnel, in cells. */
  x: number;
  /** Rake centre, in cells. */
  y: number;
  z: number;
  rows: number;
  cols: number;
  /** Seed spacing in cells. */
  dy: number;
  dz: number;
}

export const MAX_SEEDS = 400;

const FIELD_COMMON = /* wgsl */ `
struct Field {
  dims: vec4<f32>,      // nx, ny, nz, inflow speed
  step: vec4<f32>,      // dt in solver steps, particle life, particles per seed, seed count
  rake: vec4<f32>,      // x, centre y, centre z, smoke source radius
  rakeGrid: vec4<f32>,  // cols, rows, dz, dy
  aux: vec4<f32>,       // Q scale, vorticity scale, smoke decay per step, unused
};
@group(0) @binding(0) var<uniform> F: Field;
@group(0) @binding(1) var macroTex: texture_3d<f32>;
@group(0) @binding(2) var solidTex: texture_3d<f32>;
@group(0) @binding(3) var samp: sampler;

fn velocity(p: vec3<f32>) -> vec3<f32> {
  return textureSampleLevel(macroTex, samp, p / F.dims.xyz, 0.0).xyz;
}
`;

const PARTICLE_SHADER = FIELD_COMMON + /* wgsl */ `
@group(0) @binding(4) var<storage, read_write> parts: array<vec4<f32>>;

@compute @workgroup_size(64)
fn advect(@builtin(global_invocation_id) gid: vec3<u32>) {
  let per = u32(F.step.z);
  let total = per * u32(F.step.w);
  let i = gid.x;
  if (i >= total) { return; }
  let dt = F.step.x;
  let life = F.step.y;
  var pos = parts[i].xyz;
  var age = parts[i].w + dt;

  if (age >= life) {
    age -= life;
    let seed = i / per;
    let cols = u32(F.rakeGrid.x);
    let row = f32(seed / cols);
    let col = f32(seed % cols);
    pos = vec3<f32>(
      F.rake.x,
      F.rake.y + (row - (F.rakeGrid.y - 1.0) * 0.5) * F.rakeGrid.w,
      F.rake.z + (col - (F.rakeGrid.x - 1.0) * 0.5) * F.rakeGrid.z,
    );
    pos += velocity(pos) * age;
  } else if (pos.x > -100.0) {
    let mid = pos + 0.5 * dt * velocity(pos);
    pos += dt * velocity(mid);
    let solid = textureSampleLevel(solidTex, samp, pos / F.dims.xyz, 0.0).r;
    if (solid > 0.6 || any(pos < vec3<f32>(0.5)) || any(pos > F.dims.xyz - vec3<f32>(0.5))) {
      pos = vec3<f32>(-1000.0);
    }
  }
  parts[i] = vec4<f32>(pos, age);
}
`;

const SMOKE_SHADER = FIELD_COMMON + /* wgsl */ `
@group(0) @binding(4) var smokeSrc: texture_3d<f32>;
@group(0) @binding(5) var smokeDst: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(4, 4, 4)
fn advect(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = vec3<u32>(F.dims.xyz);
  if (any(gid >= dims)) { return; }
  let p = vec3<f32>(gid) + vec3<f32>(0.5);
  let uvw = p / F.dims.xyz;
  var density = 0.0;
  let solid = textureSampleLevel(solidTex, samp, uvw, 0.0).r;
  let edge = any(gid == vec3<u32>(0u)) || any(gid == dims - vec3<u32>(1u));
  if (solid < 0.5 && !edge) {
    let dt = F.step.x;
    let mid = p - 0.5 * dt * velocity(p);
    let back = p - dt * velocity(mid);
    density = textureSampleLevel(smokeSrc, samp, back / F.dims.xyz, 0.0).r * pow(F.aux.z, dt);

    // Smoke nozzles sit at the rake seeds.
    if (abs(p.x - F.rake.x) < 2.0) {
      let row = clamp(round((p.y - F.rake.y) / F.rakeGrid.w + (F.rakeGrid.y - 1.0) * 0.5), 0.0, F.rakeGrid.y - 1.0);
      let col = clamp(round((p.z - F.rake.z) / F.rakeGrid.z + (F.rakeGrid.x - 1.0) * 0.5), 0.0, F.rakeGrid.x - 1.0);
      let sy = F.rake.y + (row - (F.rakeGrid.y - 1.0) * 0.5) * F.rakeGrid.w;
      let sz = F.rake.z + (col - (F.rakeGrid.x - 1.0) * 0.5) * F.rakeGrid.z;
      let d = vec2<f32>(p.y - sy, p.z - sz);
      density = max(density, exp(-dot(d, d) / (F.rake.w * F.rake.w)));
    }
  }
  textureStore(smokeDst, vec3<i32>(gid), vec4<f32>(density, 0.0, 0.0, 0.0));
}
`;

const VORTEX_SHADER = FIELD_COMMON + /* wgsl */ `
@group(0) @binding(4) var vortDst: texture_storage_3d<rgba16float, write>;

fn vel(p: vec3<i32>) -> vec3<f32> {
  let hi = vec3<i32>(F.dims.xyz) - vec3<i32>(1);
  return textureLoad(macroTex, clamp(p, vec3<i32>(0), hi), 0).xyz;
}

@compute @workgroup_size(4, 4, 4)
fn derive(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = vec3<u32>(F.dims.xyz);
  if (any(gid >= dims)) { return; }
  let p = vec3<i32>(gid);
  let solid = textureLoad(solidTex, p, 0).r;
  var q = 0.0;
  var w = 0.0;
  let u = vel(p);
  if (solid < 0.5) {
    let dx = (vel(p + vec3<i32>(1, 0, 0)) - vel(p - vec3<i32>(1, 0, 0))) * 0.5;
    let dy = (vel(p + vec3<i32>(0, 1, 0)) - vel(p - vec3<i32>(0, 1, 0))) * 0.5;
    let dz = (vel(p + vec3<i32>(0, 0, 1)) - vel(p - vec3<i32>(0, 0, 1))) * 0.5;
    // Q = -1/2 * trace(J * J), positive where rotation dominates strain.
    q = -0.5 * (dx.x * dx.x + dy.y * dy.y + dz.z * dz.z) - (dy.x * dx.y + dz.x * dx.z + dz.y * dy.z);
    let curl = vec3<f32>(dy.z - dz.y, dz.x - dx.z, dx.y - dy.x);
    w = length(curl);
  }
  textureStore(vortDst, p, vec4<f32>(q * F.aux.x, w * F.aux.y, length(u) / F.dims.w, 0.0));
}
`;

export class Fields {
  readonly particleBuffer: GPUBuffer;
  readonly perSeed: number;
  readonly life: number;
  readonly vortTex: GPUTexture;
  seedCount = 0;

  private readonly device: GPUDevice;
  private readonly lbm: Lbm3D;
  private readonly uniform: GPUBuffer;
  private readonly smokeTex: [GPUTexture, GPUTexture];
  private readonly particlePipeline: GPUComputePipeline;
  private readonly smokePipeline: GPUComputePipeline;
  private readonly vortexPipeline: GPUComputePipeline;
  private readonly particleGroup: GPUBindGroup;
  private readonly smokeGroups: [GPUBindGroup, GPUBindGroup];
  private readonly vortexGroup: GPUBindGroup;
  private smokeCurrent = 0;
  private rake: Rake;
  /** Object length in cells, used to normalise velocity gradients. */
  private refLength = 32;

  constructor(device: GPUDevice, lbm: Lbm3D, sampler: GPUSampler) {
    this.device = device;
    this.lbm = lbm;
    const { nx, ny, nz } = lbm;
    this.perSeed = Math.max(256, nx * 2);
    this.life = (1.5 * nx) / lbm.inflowSpeed;
    this.rake = { x: 2.5, y: ny / 2, z: nz / 2, rows: 9, cols: 1, dy: ny / 16, dz: nz / 16 };

    this.uniform = device.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.particleBuffer = device.createBuffer({
      size: MAX_SEEDS * this.perSeed * 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const volume = (): GPUTexture =>
      device.createTexture({
        size: [nx, ny, nz],
        dimension: '3d',
        format: 'rgba16float',
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      });
    this.smokeTex = [volume(), volume()];
    this.vortTex = volume();

    const base = [
      { binding: 0, resource: { buffer: this.uniform } },
      { binding: 1, resource: lbm.macroTex.createView() },
      { binding: 2, resource: lbm.solidTex.createView() },
      { binding: 3, resource: sampler },
    ];
    const C = GPUShaderStage.COMPUTE;
    const volumeIn: GPUBindGroupLayoutEntry['texture'] = { sampleType: 'float', viewDimension: '3d' };
    const volumeOut: GPUBindGroupLayoutEntry['storageTexture'] = {
      access: 'write-only',
      format: 'rgba16float',
      viewDimension: '3d',
    };
    const shared: GPUBindGroupLayoutEntry[] = [
      { binding: 0, visibility: C, buffer: { type: 'uniform' } },
      { binding: 1, visibility: C, texture: volumeIn },
      { binding: 2, visibility: C, texture: volumeIn },
      { binding: 3, visibility: C, sampler: { type: 'filtering' } },
    ];
    const make = (
      code: string,
      entryPoint: string,
      extra: GPUBindGroupLayoutEntry[],
    ): { pipeline: GPUComputePipeline; layout: GPUBindGroupLayout } => {
      const layout = device.createBindGroupLayout({ entries: [...shared, ...extra] });
      const pipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
        compute: { module: device.createShaderModule({ code }), entryPoint },
      });
      return { pipeline, layout };
    };

    const particles = make(PARTICLE_SHADER, 'advect', [{ binding: 4, visibility: C, buffer: { type: 'storage' } }]);
    this.particlePipeline = particles.pipeline;
    this.particleGroup = device.createBindGroup({
      layout: particles.layout,
      entries: [...base, { binding: 4, resource: { buffer: this.particleBuffer } }],
    });

    const smoke = make(SMOKE_SHADER, 'advect', [
      { binding: 4, visibility: C, texture: volumeIn },
      { binding: 5, visibility: C, storageTexture: volumeOut },
    ]);
    this.smokePipeline = smoke.pipeline;
    const smokeGroup = (src: GPUTexture, dst: GPUTexture): GPUBindGroup =>
      device.createBindGroup({
        layout: smoke.layout,
        entries: [...base, { binding: 4, resource: src.createView() }, { binding: 5, resource: dst.createView() }],
      });
    this.smokeGroups = [smokeGroup(this.smokeTex[0], this.smokeTex[1]), smokeGroup(this.smokeTex[1], this.smokeTex[0])];

    const vortex = make(VORTEX_SHADER, 'derive', [{ binding: 4, visibility: C, storageTexture: volumeOut }]);
    this.vortexPipeline = vortex.pipeline;
    this.vortexGroup = device.createBindGroup({
      layout: vortex.layout,
      entries: [...base, { binding: 4, resource: this.vortTex.createView() }],
    });

    this.setRake(this.rake);
  }

  /** The smoke texture that holds the latest density. */
  get smokeTexture(): GPUTexture {
    return this.smokeTex[this.smokeCurrent];
  }

  get smokeTextures(): [GPUTexture, GPUTexture] {
    return this.smokeTex;
  }

  get smokeIndex(): number {
    return this.smokeCurrent;
  }

  setReferenceLength(cells: number): void {
    this.refLength = Math.max(cells, 4);
  }

  setRake(rake: Rake): void {
    const rows = Math.max(1, Math.round(rake.rows));
    const cols = Math.max(1, Math.min(Math.round(rake.cols), Math.floor(MAX_SEEDS / rows)));
    this.rake = { ...rake, rows, cols };
    this.seedCount = rows * cols;
    this.resetParticles();
  }

  /** Parks every particle off-stage with staggered ages, so streaklines grow out of the rake. */
  resetParticles(): void {
    const total = this.seedCount * this.perSeed;
    const data = new Float32Array(total * 4);
    for (let i = 0; i < total; i++) {
      const k = i % this.perSeed;
      data[i * 4] = -1000;
      data[i * 4 + 1] = -1000;
      data[i * 4 + 2] = -1000;
      data[i * 4 + 3] = (this.life * k) / this.perSeed;
    }
    this.device.queue.writeBuffer(this.particleBuffer, 0, data);
  }

  update(dtSteps: number, want: { particles: boolean; smoke: boolean; vortex: boolean }): void {
    const { nx, ny, nz } = this.lbm;
    const u0 = this.lbm.inflowSpeed;
    const r = this.rake;
    const gradScale = this.refLength / u0;
    const data = new Float32Array([
      nx, ny, nz, u0,
      dtSteps, this.life, this.perSeed, this.seedCount,
      r.x, r.y, r.z, Math.max(0.9, nx / 110),
      r.cols, r.rows, r.dz, r.dy,
      gradScale * gradScale, gradScale, 0.9996, 0,
    ]);
    this.device.queue.writeBuffer(this.uniform, 0, data);

    const encoder = this.device.createCommandEncoder();
    const grid: [number, number, number] = [Math.ceil(nx / 4), Math.ceil(ny / 4), Math.ceil(nz / 4)];
    if (want.particles && dtSteps > 0) {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.particlePipeline);
      pass.setBindGroup(0, this.particleGroup);
      pass.dispatchWorkgroups(Math.ceil((this.seedCount * this.perSeed) / 64));
      pass.end();
    }
    if (want.smoke && dtSteps > 0) {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.smokePipeline);
      pass.setBindGroup(0, this.smokeGroups[this.smokeCurrent]);
      pass.dispatchWorkgroups(...grid);
      pass.end();
      this.smokeCurrent = 1 - this.smokeCurrent;
    }
    if (want.vortex) {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.vortexPipeline);
      pass.setBindGroup(0, this.vortexGroup);
      pass.dispatchWorkgroups(...grid);
      pass.end();
    }
    this.device.queue.submit([encoder.finish()]);
  }

  destroy(): void {
    this.particleBuffer.destroy();
    this.uniform.destroy();
    this.vortTex.destroy();
    for (const t of this.smokeTex) t.destroy();
  }
}
