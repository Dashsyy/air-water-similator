import type { Fields } from './fields';
import type { Lbm3D } from './lbm3d';
import type { Model } from '../geometry/model';
import type { Mat4, Vec3 } from '../math';
import {
  LINE_HELPER_SHADER,
  MESH_SHADER,
  SCENE_FLOATS,
  SLICE_SHADER,
  STREAK_SHADER,
  VOLUME_SHADER,
} from './shaders';

export type SliceAxis = 'x' | 'y' | 'z';
export type SliceField = 'speed' | 'pressure' | 'vorticity';

export interface ViewSettings {
  smokeLines: boolean;
  lineColor: 'smoke' | 'speed';
  lineWidth: number;
  volumetricSmoke: boolean;
  smokeDensity: number;
  pressureMap: boolean;
  vortices: boolean;
  vortexThreshold: number;
  slice: boolean;
  sliceAxis: SliceAxis;
  /** Slice position as a fraction of the tunnel along the slice axis. */
  slicePosition: number;
  sliceField: SliceField;
  showModel: boolean;
}

export interface FrameInput {
  viewProj: Mat4;
  invViewProj: Mat4;
  cameraPosition: Vec3;
  modelMatrix: Mat4;
  settings: ViewSettings;
}

const SAMPLES = 4;
const BACKGROUND = { r: 0.035, g: 0.045, b: 0.07, a: 1 };

export class Renderer {
  readonly sampler: GPUSampler;
  private readonly context: GPUCanvasContext;
  private readonly format: GPUTextureFormat;
  private readonly layout: GPUBindGroupLayout;
  private readonly sceneBuffer: GPUBuffer;
  private readonly meshPipeline: GPURenderPipeline;
  private readonly helperPipeline: GPURenderPipeline;
  private readonly streakPipeline: GPURenderPipeline;
  private readonly slicePipeline: GPURenderPipeline;
  private readonly volumePipeline: GPURenderPipeline;

  private colorTarget: GPUTexture | null = null;
  private depthTarget: GPUTexture | null = null;
  private meshBuffer: GPUBuffer | null = null;
  private meshVertices = 0;
  private boxBuffer: GPUBuffer | null = null;
  private boxVertices = 0;
  private groups: [GPUBindGroup, GPUBindGroup] | null = null;
  private lbm: Lbm3D | null = null;
  private fields: Fields | null = null;

  constructor(
    private readonly device: GPUDevice,
    private readonly canvas: HTMLCanvasElement,
  ) {
    const context = canvas.getContext('webgpu');
    if (!context) throw new Error('Could not create a WebGPU canvas context.');
    this.context = context;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format: this.format, alphaMode: 'opaque' });

    this.sampler = device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
      addressModeW: 'clamp-to-edge',
    });
    this.sceneBuffer = device.createBuffer({
      size: SCENE_FLOATS * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const stages = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT;
    const volumeTexture: GPUBindGroupLayoutEntry['texture'] = { sampleType: 'float', viewDimension: '3d' };
    this.layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: stages, buffer: { type: 'uniform' } },
        { binding: 1, visibility: stages, sampler: { type: 'filtering' } },
        { binding: 2, visibility: stages, texture: volumeTexture },
        { binding: 3, visibility: stages, texture: volumeTexture },
        { binding: 4, visibility: stages, texture: volumeTexture },
        { binding: 5, visibility: stages, texture: volumeTexture },
        { binding: 6, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      ],
    });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.layout] });

    const alphaBlend: GPUBlendState = {
      color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' },
      alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
    };
    const premultiplied: GPUBlendState = {
      color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
      alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
    };
    const make = (
      code: string,
      opts: {
        buffers?: GPUVertexBufferLayout[];
        topology?: GPUPrimitiveTopology;
        blend?: GPUBlendState;
        depthWrite: boolean;
        depthCompare?: GPUCompareFunction;
      },
    ): GPURenderPipeline => {
      const module = device.createShaderModule({ code });
      return device.createRenderPipeline({
        layout: pipelineLayout,
        vertex: { module, entryPoint: 'vs', buffers: opts.buffers ?? [] },
        fragment: { module, entryPoint: 'fs', targets: [{ format: this.format, blend: opts.blend }] },
        primitive: { topology: opts.topology ?? 'triangle-list', cullMode: 'none' },
        depthStencil: {
          format: 'depth24plus',
          depthWriteEnabled: opts.depthWrite,
          depthCompare: opts.depthCompare ?? 'less',
        },
        multisample: { count: SAMPLES },
      });
    };

    this.meshPipeline = make(MESH_SHADER, {
      depthWrite: true,
      buffers: [
        {
          arrayStride: 36,
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x3' },
            { shaderLocation: 1, offset: 12, format: 'float32x3' },
            { shaderLocation: 2, offset: 24, format: 'float32x3' },
          ],
        },
      ],
    });
    this.helperPipeline = make(LINE_HELPER_SHADER, {
      depthWrite: false,
      topology: 'line-list',
      blend: alphaBlend,
      buffers: [{ arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] }],
    });
    this.streakPipeline = make(STREAK_SHADER, { depthWrite: false, blend: alphaBlend });
    this.slicePipeline = make(SLICE_SHADER, { depthWrite: true, blend: alphaBlend });
    this.volumePipeline = make(VOLUME_SHADER, { depthWrite: false, depthCompare: 'always', blend: premultiplied });
  }

  /** Points the renderer at a (new) solver and its derived fields. */
  attach(lbm: Lbm3D, fields: Fields): void {
    this.lbm = lbm;
    this.fields = fields;
    const smoke = fields.smokeTextures;
    const group = (smokeTex: GPUTexture): GPUBindGroup =>
      this.device.createBindGroup({
        layout: this.layout,
        entries: [
          { binding: 0, resource: { buffer: this.sceneBuffer } },
          { binding: 1, resource: this.sampler },
          { binding: 2, resource: lbm.macroTex.createView() },
          { binding: 3, resource: lbm.solidTex.createView() },
          { binding: 4, resource: smokeTex.createView() },
          { binding: 5, resource: fields.vortTex.createView() },
          { binding: 6, resource: { buffer: fields.particleBuffer } },
        ],
      });
    this.groups = [group(smoke[0]), group(smoke[1])];
    this.buildBox(lbm.nx, lbm.ny, lbm.nz);
  }

  setModel(model: Model): void {
    this.meshBuffer?.destroy();
    const count = model.triangleCount * 3;
    const data = new Float32Array(count * 9);
    for (let i = 0; i < count; i++) {
      data.set(model.positions.subarray(i * 3, i * 3 + 3), i * 9);
      data.set(model.normals.subarray(i * 3, i * 3 + 3), i * 9 + 3);
      data.set(model.colors.subarray(i * 3, i * 3 + 3), i * 9 + 6);
    }
    this.meshBuffer = this.device.createBuffer({
      size: data.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.meshBuffer, 0, data);
    this.meshVertices = count;
  }

  /** Resizes the drawing buffer to the canvas' CSS size. Returns the aspect ratio. */
  resize(): number {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(2, Math.floor(this.canvas.clientWidth * dpr));
    const h = Math.max(2, Math.floor(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h || !this.colorTarget) {
      this.canvas.width = w;
      this.canvas.height = h;
      this.colorTarget?.destroy();
      this.depthTarget?.destroy();
      const usage = GPUTextureUsage.RENDER_ATTACHMENT;
      this.colorTarget = this.device.createTexture({ size: [w, h], format: this.format, sampleCount: SAMPLES, usage });
      this.depthTarget = this.device.createTexture({ size: [w, h], format: 'depth24plus', sampleCount: SAMPLES, usage });
    }
    return w / h;
  }

  render(frame: FrameInput): void {
    if (!this.lbm || !this.fields || !this.groups || !this.colorTarget || !this.depthTarget) return;
    const { lbm, fields } = this;
    const s = frame.settings;
    const dims = [lbm.nx, lbm.ny, lbm.nz];
    const axisIndex = s.sliceAxis === 'x' ? 0 : s.sliceAxis === 'y' ? 1 : 2;
    const sliceAt = 1 + s.slicePosition * (dims[axisIndex] - 2);
    const fieldIndex = s.sliceField === 'speed' ? 0 : s.sliceField === 'pressure' ? 1 : 2;

    const scene = new Float32Array(SCENE_FLOATS);
    scene.set(frame.viewProj, 0);
    scene.set(frame.invViewProj, 16);
    scene.set(frame.modelMatrix, 32);
    scene.set([...frame.cameraPosition, 1], 48);
    scene.set([lbm.nx, lbm.ny, lbm.nz, lbm.inflowSpeed], 52);
    scene.set([this.canvas.width, this.canvas.height, s.lineWidth * Math.min(window.devicePixelRatio || 1, 2), 0], 56);
    scene.set([axisIndex, sliceAt, fieldIndex, 0.94], 60);
    scene.set([s.volumetricSmoke ? 1 : 0, s.vortices ? 1 : 0, s.vortexThreshold, s.smokeDensity], 64);
    scene.set([s.pressureMap ? 1 : 0, 1.0, 1.6, s.lineColor === 'speed' ? 1 : 0], 68);
    scene.set([fields.perSeed, fields.seedCount, fields.life, 0], 72);
    scene.set([0.55, 0.65, 0.8, 0.35], 76);
    this.device.queue.writeBuffer(this.sceneBuffer, 0, scene);

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.colorTarget.createView(),
          resolveTarget: this.context.getCurrentTexture().createView(),
          clearValue: BACKGROUND,
          loadOp: 'clear',
          storeOp: 'discard',
        },
      ],
      depthStencilAttachment: {
        view: this.depthTarget.createView(),
        depthClearValue: 1,
        depthLoadOp: 'clear',
        depthStoreOp: 'discard',
      },
    });
    pass.setBindGroup(0, this.groups[fields.smokeIndex]);

    if (s.showModel && this.meshBuffer) {
      pass.setPipeline(this.meshPipeline);
      pass.setVertexBuffer(0, this.meshBuffer);
      pass.draw(this.meshVertices);
    }
    if (this.boxBuffer) {
      pass.setPipeline(this.helperPipeline);
      pass.setVertexBuffer(0, this.boxBuffer);
      pass.draw(this.boxVertices);
    }
    if (s.slice) {
      pass.setPipeline(this.slicePipeline);
      pass.draw(6);
    }
    if (s.volumetricSmoke || s.vortices) {
      pass.setPipeline(this.volumePipeline);
      pass.draw(3);
    }
    if (s.smokeLines && fields.seedCount > 0) {
      pass.setPipeline(this.streakPipeline);
      pass.draw(6, fields.seedCount * fields.perSeed);
    }
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  /** Image of what was just rendered. Call straight after render() in the same task. */
  capture(type = 'image/jpeg', quality = 0.9): string {
    return this.canvas.toDataURL(type, quality);
  }

  private buildBox(nx: number, ny: number, nz: number): void {
    this.boxBuffer?.destroy();
    const v: number[] = [];
    const line = (a: Vec3, b: Vec3): void => {
      v.push(...a, ...b);
    };
    for (const y of [0, ny]) {
      line([0, y, 0], [nx, y, 0]);
      line([0, y, nz], [nx, y, nz]);
      line([0, y, 0], [0, y, nz]);
      line([nx, y, 0], [nx, y, nz]);
    }
    for (const x of [0, nx]) {
      for (const z of [0, nz]) line([x, 0, z], [x, ny, z]);
    }
    const spacing = nx / 16;
    for (let x = spacing; x < nx; x += spacing) line([x, 0, 0], [x, 0, nz]);
    for (let z = spacing; z < nz; z += spacing) line([0, 0, z], [nx, 0, z]);
    const data = new Float32Array(v);
    this.boxBuffer = this.device.createBuffer({
      size: data.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.boxBuffer, 0, data);
    this.boxVertices = data.length / 3;
  }
}
