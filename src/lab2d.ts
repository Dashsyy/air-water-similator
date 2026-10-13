/**
 * Two-dimensional companion to the wind tunnel: a D2Q9 Lattice Boltzmann solver that shows
 * vortex shedding and measures the Strouhal number from the oscillating lift force.
 */
import { initGpu } from './gpu/device';

const NX = 960;
const NY = 360;
const U0 = 0.08;
const FORCE_SCALE = 1 << 20;
const OBSTACLE_X = 200;
const OBSTACLE_SIZE = 30;

const COMPUTE = /* wgsl */ `
struct Params { dims: vec4<u32>, flow: vec4<f32> };  // nx, ny, cells, _ | inflow speed, tau, force scale, _
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<storage, read> solid: array<u32>;
@group(0) @binding(4) var macroTex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var<storage, read_write> force: array<atomic<i32>, 4>;

const C = array<vec2<i32>, 9>(
  vec2<i32>(0, 0), vec2<i32>(1, 0), vec2<i32>(-1, 0), vec2<i32>(0, 1), vec2<i32>(0, -1),
  vec2<i32>(1, 1), vec2<i32>(-1, -1), vec2<i32>(1, -1), vec2<i32>(-1, 1),
);
const W = array<f32, 9>(4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0);
const OPP = array<u32, 9>(0u, 2u, 1u, 4u, 3u, 6u, 5u, 8u, 7u);

fn feq(q: u32, rho1: f32, u: vec2<f32>) -> f32 {
  let cu = dot(vec2<f32>(C[q]), u);
  return W[q] * (rho1 + (1.0 + rho1) * (3.0 * cu + 4.5 * cu * cu - 1.5 * dot(u, u)));
}

fn index(p: vec2<i32>) -> u32 { return u32(p.x) + P.dims.x * u32(p.y); }

@compute @workgroup_size(8, 8)
fn init(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= P.dims.x || gid.y >= P.dims.y) { return; }
  let n = P.dims.z;
  let idx = index(vec2<i32>(gid.xy));
  // A slight vertical ripple breaks the symmetry so shedding starts promptly.
  let u = vec2<f32>(P.flow.x, 0.03 * P.flow.x * sin(6.2831853 * f32(gid.x) / 97.0));
  for (var q = 0u; q < 9u; q++) { dst[q * n + idx] = select(feq(q, 0.0, u), 0.0, solid[idx] == 1u); }
}

@compute @workgroup_size(8, 8)
fn step(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= P.dims.x || gid.y >= P.dims.y) { return; }
  let n = P.dims.z;
  let p = vec2<i32>(gid.xy);
  let idx = index(p);
  let uIn = vec2<f32>(P.flow.x, 0.0);

  if (gid.x == 0u || gid.y == 0u || gid.y == P.dims.y - 1u) {
    for (var q = 0u; q < 9u; q++) { dst[q * n + idx] = feq(q, 0.0, uIn); }
    textureStore(macroTex, p, vec4<f32>(uIn, 0.0, 0.0));
    return;
  }
  if (solid[idx] == 1u) {
    for (var q = 0u; q < 9u; q++) { dst[q * n + idx] = 0.0; }
    textureStore(macroTex, p, vec4<f32>(0.0, 0.0, 0.0, 1.0));
    return;
  }
  if (gid.x == P.dims.x - 1u) {
    var r = 0.0;
    var m = vec2<f32>(0.0);
    for (var q = 0u; q < 9u; q++) { let v = src[q * n + idx - 1u]; r += v; m += v * vec2<f32>(C[q]); }
    var uOut = m / (1.0 + r);
    uOut.x = max(uOut.x, 0.0);
    for (var q = 0u; q < 9u; q++) { dst[q * n + idx] = feq(q, 0.0, uOut); }
    textureStore(macroTex, p, vec4<f32>(uOut, 0.0, 0.0));
    return;
  }

  var f: array<f32, 9>;
  var bodyForce = vec2<f32>(0.0);
  var touches = false;
  for (var q = 0u; q < 9u; q++) {
    let sidx = index(p - C[q]);
    if (solid[sidx] == 1u) {
      let v = src[OPP[q] * n + idx];
      f[q] = v;
      bodyForce -= 2.0 * v * vec2<f32>(C[q]);
      touches = true;
    } else {
      f[q] = src[q * n + sidx];
    }
  }
  var rho1 = 0.0;
  var mom = vec2<f32>(0.0);
  for (var q = 0u; q < 9u; q++) { rho1 += f[q]; mom += f[q] * vec2<f32>(C[q]); }
  let u = mom / (1.0 + rho1);
  let omega = 1.0 / P.flow.y;
  for (var q = 0u; q < 9u; q++) { dst[q * n + idx] = f[q] + omega * (feq(q, rho1, u) - f[q]); }
  textureStore(macroTex, p, vec4<f32>(u, rho1, 0.0));
  if (touches) {
    let s = vec2<i32>(round(bodyForce * P.flow.z));
    atomicAdd(&force[0], s.x);
    atomicAdd(&force[1], s.y);
  }
}
`;

const RENDER = /* wgsl */ `
struct View { dims: vec4<f32> };  // nx, ny, field, inflow speed
@group(0) @binding(0) var<uniform> V: View;
@group(0) @binding(1) var macroTex: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
  let p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  return vec4<f32>(p[vi], 0.0, 1.0);
}

fn load(p: vec2<i32>) -> vec4<f32> {
  return textureLoad(macroTex, clamp(p, vec2<i32>(0), vec2<i32>(V.dims.xy) - vec2<i32>(1)), 0);
}

fn turbo(t: f32) -> vec3<f32> {
  let x = clamp(t, 0.0, 1.0);
  let v4 = vec4<f32>(1.0, x, x * x, x * x * x);
  let v2 = v4.zw * v4.z;
  return clamp(vec3<f32>(
    dot(v4, vec4<f32>(0.13572138, 4.61539260, -42.66032258, 132.13108234)) + dot(v2, vec2<f32>(-152.94239396, 59.28637943)),
    dot(v4, vec4<f32>(0.09140261, 2.19418839, 4.84296658, -14.18503333)) + dot(v2, vec2<f32>(4.27729857, 2.82956604)),
    dot(v4, vec4<f32>(0.10667330, 12.64194608, -60.58204836, 110.36276771)) + dot(v2, vec2<f32>(-89.90310912, 27.34824973)),
  ), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn diverging(t: f32) -> vec3<f32> {
  let x = clamp(t, -1.0, 1.0);
  let mid = vec3<f32>(0.06, 0.07, 0.10);
  return mix(mid, select(vec3<f32>(1.0, 0.42, 0.20), vec3<f32>(0.25, 0.60, 1.0), x < 0.0), pow(abs(x), 0.7));
}

struct Canvas { size: vec4<f32> };
@group(0) @binding(2) var<uniform> Cv: Canvas;

@fragment
fn fs(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
  let uv = vec2<f32>(frag.x / Cv.size.x, 1.0 - frag.y / Cv.size.y);
  let p = vec2<i32>(uv * V.dims.xy);
  let m = load(p);
  if (m.w > 0.5) { return vec4<f32>(0.78, 0.80, 0.84, 1.0); }
  var rgb: vec3<f32>;
  if (V.dims.z < 0.5) {
    let curl = (load(p + vec2<i32>(1, 0)).y - load(p - vec2<i32>(1, 0)).y - load(p + vec2<i32>(0, 1)).x + load(p - vec2<i32>(0, 1)).x) * 0.5;
    rgb = diverging(curl / V.dims.w * 9.0);
  } else if (V.dims.z < 1.5) {
    rgb = turbo(length(m.xy) / V.dims.w / 1.7);
  } else {
    let cp = (m.z / 3.0) / (0.5 * V.dims.w * V.dims.w);
    rgb = mix(vec3<f32>(0.93, 0.92, 0.90), select(vec3<f32>(0.80, 0.10, 0.12), vec3<f32>(0.13, 0.30, 0.78), cp < 0.0), pow(min(abs(cp), 1.0), 0.8));
  }
  return vec4<f32>(rgb, 1.0);
}
`;

type Shape = 'cylinder' | 'square' | 'plate' | 'wedge' | 'none';

function buildShape(shape: Shape): Uint32Array {
  const mask = new Uint32Array(NX * NY);
  const cy = NY / 2 + 0.5;
  const r = OBSTACLE_SIZE / 2;
  for (let y = 1; y < NY - 1; y++) {
    for (let x = 1; x < NX - 1; x++) {
      const dx = x + 0.5 - OBSTACLE_X;
      const dy = y + 0.5 - cy;
      let inside = false;
      if (shape === 'cylinder') inside = dx * dx + dy * dy < r * r;
      else if (shape === 'square') inside = Math.abs(dx) < r && Math.abs(dy) < r;
      else if (shape === 'plate') inside = Math.abs(dx) < 2 && Math.abs(dy) < r;
      else if (shape === 'wedge') inside = dx > -r && dx < r && Math.abs(dy) < ((dx + r) / (2 * r)) * r;
      if (inside) mask[x + NX * y] = 1;
    }
  }
  return mask;
}

/** Strouhal number of a circular cylinder (Williamson's fit), valid while shedding is laminar. */
function textbookStrouhal(re: number): number | null {
  if (re < 47) return null;
  if (re <= 190) return 0.2175 - 5.1064 / re;
  return 0.2;
}

async function start(): Promise<void> {
  const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;
  let device: GPUDevice;
  try {
    device = (await initGpu()).device;
  } catch (error) {
    const box = $('fatal');
    box.hidden = false;
    box.textContent = error instanceof Error ? error.message : String(error);
    return;
  }
  const canvas = $<HTMLCanvasElement>('lab-canvas');
  const context = canvas.getContext('webgpu') as GPUCanvasContext;
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'opaque' });

  const cells = NX * NY;
  const pops = [0, 1].map(() => device.createBuffer({ size: cells * 9 * 4, usage: GPUBufferUsage.STORAGE }));
  const solidBuf = device.createBuffer({ size: cells * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const forceBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const readback = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const params = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const viewBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const canvasBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const macroTex = device.createTexture({
    size: [NX, NY],
    format: 'rgba16float',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  });

  const computeModule = device.createShaderModule({ code: COMPUTE });
  const computeLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  });
  const computePipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [computeLayout] });
  const stepPipeline = device.createComputePipeline({ layout: computePipelineLayout, compute: { module: computeModule, entryPoint: 'step' } });
  const initPipeline = device.createComputePipeline({ layout: computePipelineLayout, compute: { module: computeModule, entryPoint: 'init' } });
  const macroView = macroTex.createView();
  const groups = [0, 1].map((i) =>
    device.createBindGroup({
      layout: computeLayout,
      entries: [
        { binding: 0, resource: { buffer: params } },
        { binding: 1, resource: { buffer: pops[i] } },
        { binding: 2, resource: { buffer: pops[1 - i] } },
        { binding: 3, resource: { buffer: solidBuf } },
        { binding: 4, resource: macroView },
        { binding: 5, resource: { buffer: forceBuf } },
      ],
    }),
  );

  const renderModule = device.createShaderModule({ code: RENDER });
  const renderPipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: renderModule, entryPoint: 'vs' },
    fragment: { module: renderModule, entryPoint: 'fs', targets: [{ format }] },
  });
  const renderGroup = device.createBindGroup({
    layout: renderPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: viewBuf } },
      { binding: 1, resource: macroView },
      { binding: 2, resource: { buffer: canvasBuf } },
    ],
  });

  // ----- State -----
  let mask = buildShape('cylinder');
  let reynolds = 120;
  let current = 0;
  let totalSteps = 0;
  let stepsSinceSample = 0;
  let pending = false;
  let generation = 0;
  let lift: { t: number; fy: number }[] = [];
  let drag: { fx: number; steps: number }[] = [];
  let measureFrom = 0;

  const frontalRows = (): number => {
    let rows = 0;
    for (let y = 0; y < NY; y++) {
      for (let x = 0; x < NX; x++) {
        if (mask[x + NX * y]) {
          rows++;
          break;
        }
      }
    }
    return rows;
  };
  let width = frontalRows();

  const writeParams = (): void => {
    const size = Math.max(width, 4);
    const viscosity = (U0 * size) / reynolds;
    const data = new ArrayBuffer(32);
    new Uint32Array(data, 0, 4).set([NX, NY, cells, 0]);
    new Float32Array(data, 16, 4).set([U0, 3 * viscosity + 0.5, FORCE_SCALE, 0]);
    device.queue.writeBuffer(params, 0, data);
  };
  const restartMeasurement = (): void => {
    generation++;
    lift = [];
    drag = [];
    measureFrom = totalSteps;
    stepsSinceSample = 0;
    const encoder = device.createCommandEncoder();
    encoder.clearBuffer(forceBuf);
    device.queue.submit([encoder.finish()]);
  };
  const uploadMask = (): void => {
    device.queue.writeBuffer(solidBuf, 0, mask);
    width = frontalRows();
    writeParams();
    restartMeasurement();
  };
  const resetFlow = (): void => {
    const encoder = device.createCommandEncoder();
    for (const g of groups) {
      const pass = encoder.beginComputePass();
      pass.setPipeline(initPipeline);
      pass.setBindGroup(0, g);
      pass.dispatchWorkgroups(Math.ceil(NX / 8), Math.ceil(NY / 8));
      pass.end();
    }
    device.queue.submit([encoder.finish()]);
    current = 0;
    restartMeasurement();
  };

  const advance = (steps: number): void => {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(stepPipeline);
    for (let i = 0; i < steps; i++) {
      pass.setBindGroup(0, groups[current]);
      pass.dispatchWorkgroups(Math.ceil(NX / 8), Math.ceil(NY / 8));
      current = 1 - current;
    }
    pass.end();
    totalSteps += steps;
    stepsSinceSample += steps;
    const take = !pending && stepsSinceSample >= 16;
    if (take) {
      encoder.copyBufferToBuffer(forceBuf, 0, readback, 0, 16);
      encoder.clearBuffer(forceBuf);
    }
    device.queue.submit([encoder.finish()]);
    if (!take) return;
    const n = stepsSinceSample;
    const at = totalSteps;
    const gen = generation;
    stepsSinceSample = 0;
    pending = true;
    readback
      .mapAsync(GPUMapMode.READ)
      .then(() => {
        const raw = new Int32Array(readback.getMappedRange().slice(0));
        readback.unmap();
        pending = false;
        if (gen !== generation) return;
        const k = 1 / (FORCE_SCALE * n);
        lift.push({ t: at, fy: raw[1] * k });
        drag.push({ fx: raw[0] * k, steps: n });
        if (lift.length > 4000) {
          lift = lift.slice(-3000);
          drag = drag.slice(-3000);
        }
      })
      .catch((error) => {
        pending = false;
        console.error('Force read-back failed', error);
      });
  };

  /** Shedding frequency from upward zero crossings of the lift signal, in cycles per step. */
  const sheddingFrequency = (): number | null => {
    const warm = lift.filter((s) => s.t - measureFrom > (6 * NX) / U0 / 10);
    if (warm.length < 40) return null;
    const mean = warm.reduce((a, s) => a + s.fy, 0) / warm.length;
    const amplitude = Math.sqrt(warm.reduce((a, s) => a + (s.fy - mean) ** 2, 0) / warm.length);
    const reference = 0.5 * U0 * U0 * Math.max(width, 1);
    if (amplitude < reference * 0.01) return null;
    const crossings: number[] = [];
    for (let i = 1; i < warm.length; i++) {
      const a = warm[i - 1].fy - mean;
      const b = warm[i].fy - mean;
      if (a < 0 && b >= 0) crossings.push(warm[i - 1].t + ((warm[i].t - warm[i - 1].t) * -a) / (b - a));
    }
    if (crossings.length < 4) return null;
    const recent = crossings.slice(-9);
    return (recent.length - 1) / (recent[recent.length - 1] - recent[0]);
  };

  // ----- Controls -----
  const shape = $<HTMLSelectElement>('shape');
  shape.addEventListener('change', () => {
    mask = buildShape(shape.value as Shape);
    uploadMask();
    resetFlow();
  });
  const field = $<HTMLSelectElement>('field');
  const reInput = $<HTMLInputElement>('reynolds');
  const reOut = reInput.parentElement?.querySelector('output') as HTMLOutputElement;
  reOut.textContent = reInput.value;
  reInput.addEventListener('input', () => {
    reynolds = Number(reInput.value);
    reOut.textContent = reInput.value;
    writeParams();
    restartMeasurement();
  });
  $('lab-reset').addEventListener('click', resetFlow);

  const tool = $<HTMLSelectElement>('tool');
  let painting = false;
  const paint = (e: PointerEvent): void => {
    const rect = canvas.getBoundingClientRect();
    const cx = Math.floor(((e.clientX - rect.left) / rect.width) * NX);
    const cy = Math.floor((1 - (e.clientY - rect.top) / rect.height) * NY);
    const value = tool.value === 'draw' ? 1 : 0;
    const r = 5;
    for (let y = cy - r; y <= cy + r; y++) {
      for (let x = cx - r; x <= cx + r; x++) {
        if (x < 40 || x > NX - 40 || y < 4 || y > NY - 5) continue;
        if ((x - cx) ** 2 + (y - cy) ** 2 <= r * r) mask[x + NX * y] = value;
      }
    }
    uploadMask();
  };
  canvas.addEventListener('pointerdown', (e) => {
    painting = true;
    canvas.setPointerCapture(e.pointerId);
    paint(e);
  });
  canvas.addEventListener('pointermove', (e) => {
    if (painting) paint(e);
  });
  canvas.addEventListener('pointerup', () => (painting = false));
  canvas.addEventListener('pointercancel', () => (painting = false));

  // ----- Loop -----
  uploadMask();
  resetFlow();
  let lastHud = 0;
  const frame = (now: number): void => {
    advance(24);
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(2, Math.floor(canvas.clientWidth * dpr));
    const h = Math.max(2, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    const fieldIndex = field.value === 'vorticity' ? 0 : field.value === 'speed' ? 1 : 2;
    device.queue.writeBuffer(viewBuf, 0, new Float32Array([NX, NY, fieldIndex, U0]));
    device.queue.writeBuffer(canvasBuf, 0, new Float32Array([w, h, 0, 0]));
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: context.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
    });
    pass.setPipeline(renderPipeline);
    pass.setBindGroup(0, renderGroup);
    pass.draw(3);
    pass.end();
    device.queue.submit([encoder.finish()]);

    if (now - lastHud > 300) {
      lastHud = now;
      const reference = 0.5 * U0 * U0 * Math.max(width, 1);
      const recent = drag.slice(-400);
      const steps = recent.reduce((a, s) => a + s.steps, 0);
      const cd = steps ? recent.reduce((a, s) => a + s.fx * s.steps, 0) / steps / reference : NaN;
      $('lab-cd').textContent = width > 0 && Number.isFinite(cd) ? cd.toFixed(2) : '–';
      const f = sheddingFrequency();
      const st = f === null ? null : (f * width) / U0;
      $('lab-st').textContent = st === null ? '–' : st.toFixed(3);
      const ref = shape.value === 'cylinder' ? textbookStrouhal(reynolds) : null;
      $('lab-st-ref').textContent = shape.value !== 'cylinder' ? 'cylinder only' : ref === null ? 'no shedding' : ref.toFixed(3);
      $('lab-status').textContent =
        width === 0 ? 'Draw an obstacle' : st === null ? 'Waiting for a steady rhythm…' : 'Shedding vortices';
      (window as unknown as { __lab: unknown }).__lab = { cd, st, ref, totalSteps, width };
    }
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

void start();
