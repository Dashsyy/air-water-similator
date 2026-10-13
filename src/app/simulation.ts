import type { GpuContext } from '../gpu/device';
import { BYTES_PER_CELL } from '../gpu/device';
import { Fields, MAX_SEEDS, type Rake } from '../gpu/fields';
import { Lbm3D } from '../gpu/lbm3d';
import { Renderer, type ViewSettings } from '../gpu/renderer';
import { OrbitCamera } from '../gpu/camera';
import { placeModel, transformedTriangles, type Model, type Orientation, type Placement } from '../geometry/model';
import { addFloor, voxelize, type VoxelGrid } from '../geometry/voxelize';
import {
  ForceRecorder,
  flowNumbers,
  realWorld,
  type FlowNumbers,
  type Measurement,
  type RealWorld,
  type TunnelSetup,
} from '../physics/aero';

export interface QualityTier {
  id: string;
  label: string;
  nx: number;
  ny: number;
  nz: number;
}

export const QUALITY_TIERS: QualityTier[] = [
  { id: 'low', label: 'Low', nx: 96, ny: 48, nz: 48 },
  { id: 'medium', label: 'Medium', nx: 128, ny: 64, nz: 64 },
  { id: 'high', label: 'High', nx: 192, ny: 96, nz: 96 },
  { id: 'ultra', label: 'Ultra', nx: 256, ny: 128, nz: 128 },
];

export interface Conditions {
  /** Wind speed in km/h. */
  windKmh: number;
  /** Longest dimension of the object in metres. */
  realSize: number;
  massKg: number;
}

export type RakeLayout = 'vertical' | 'horizontal' | 'grid';

export interface RakeSettings {
  layout: RakeLayout;
  /** Seeds along the main direction of the layout. */
  count: number;
  /** Centre height and sideways position as fractions of the tunnel. */
  height: number;
  sideways: number;
  /** Extent of the rake as a fraction of the tunnel. */
  spread: number;
}

export interface Snapshot {
  measurement: Measurement;
  realWorld: RealWorld;
  flow: FlowNumbers;
  setup: TunnelSetup;
  grid: VoxelGrid;
  tier: QualityTier;
  stepsPerSecond: number;
  totalSteps: number;
}

const MAX_STEPS_PER_FRAME = 24;

export class Simulation {
  readonly camera: OrbitCamera;
  readonly renderer: Renderer;
  paused = false;
  view: ViewSettings = {
    smokeLines: true,
    lineColor: 'speed',
    lineWidth: 1.4,
    volumetricSmoke: false,
    smokeDensity: 0.4,
    pressureMap: true,
    vortices: false,
    vortexThreshold: 0.6,
    slice: false,
    sliceAxis: 'z',
    slicePosition: 0.5,
    sliceField: 'speed',
    showModel: true,
  };

  private readonly device: GPUDevice;
  private lbm!: Lbm3D;
  private fields!: Fields;
  private tier!: QualityTier;
  private model: Model | null = null;
  private orientation: Orientation = { yaw: 0, pitch: 0, roll: 0, onGround: false };
  private conditions: Conditions = { windKmh: 100, realSize: 1, massKg: 10 };
  private rake: RakeSettings = { layout: 'vertical', count: 13, height: 0.5, sideways: 0.5, spread: 0.6 };
  private placement: Placement | null = null;
  private grid: VoxelGrid | null = null;
  private recorder: ForceRecorder | null = null;
  private stepsPerFrame = 6;
  private frameTimeAvg = 16.7;
  private framesInFlight = 0;
  private stepRate = 0;
  private rateWindowStart = 0;
  private rateWindowSteps = 0;
  private geometryDirty = false;

  constructor(
    private readonly gpu: GpuContext,
    canvas: HTMLCanvasElement,
  ) {
    this.device = gpu.device;
    this.renderer = new Renderer(gpu.device, canvas);
    this.camera = new OrbitCamera(canvas);
  }

  /** Tiers this device has enough GPU buffer space for. */
  availableTiers(): QualityTier[] {
    return QUALITY_TIERS.filter((t) => t.nx * t.ny * t.nz <= this.gpu.maxCells);
  }

  get currentTier(): QualityTier {
    return this.tier;
  }

  get currentOrientation(): Orientation {
    return { ...this.orientation };
  }

  get currentConditions(): Conditions {
    return { ...this.conditions };
  }

  get currentModel(): Model | null {
    return this.model;
  }

  setQuality(tier: QualityTier, keepCamera = false): void {
    if (tier.nx * tier.ny * tier.nz * BYTES_PER_CELL > this.gpu.maxCells * BYTES_PER_CELL) {
      throw new Error(`The ${tier.label} grid does not fit in this GPU's buffer limits.`);
    }
    this.fields?.destroy();
    this.lbm?.destroy();
    this.tier = tier;
    this.lbm = new Lbm3D(this.device, tier.nx, tier.ny, tier.nz);
    this.lbm.onForce = (sample) => this.recorder?.add(sample);
    this.fields = new Fields(this.device, this.lbm, this.renderer.sampler);
    this.renderer.attach(this.lbm, this.fields);
    if (!keepCamera) this.camera.frame(tier.nx, tier.ny, tier.nz);
    this.stepsPerFrame = 6;
    this.applyRake();
    this.rebuildGeometry(true);
  }

  setModel(model: Model, orientation?: Partial<Orientation>): void {
    this.model = model;
    this.orientation = { yaw: 0, pitch: 0, roll: 0, onGround: model.grounded, ...orientation };
    this.conditions = { ...this.conditions, realSize: model.realSize, massKg: model.mass };
    this.renderer.setModel(model);
    this.rebuildGeometry(true);
  }

  setOrientation(o: Partial<Orientation>): void {
    this.orientation = { ...this.orientation, ...o };
    this.geometryDirty = true;
  }

  setConditions(c: Partial<Conditions>): void {
    this.conditions = { ...this.conditions, ...c };
    if (!this.grid) return;
    const before = this.flow().tau;
    this.applyFlow();
    // A different viscosity changes the flow itself, so the average has to start again.
    if (Math.abs(this.flow().tau - before) > 1e-9) this.restartMeasurement();
  }

  setRake(r: Partial<RakeSettings>): void {
    this.rake = { ...this.rake, ...r };
    this.applyRake();
  }

  get rakeSettings(): RakeSettings {
    return { ...this.rake };
  }

  /** Restarts the flow from a uniform stream and clears the measurement. */
  restartFlow(): void {
    this.lbm.reset();
    this.fields.resetParticles();
    this.restartMeasurement();
  }

  frame(now: number, lastFrame: number): void {
    if (this.geometryDirty) {
      this.geometryDirty = false;
      this.rebuildGeometry(false);
    }
    const dt = Math.min(100, Math.max(1, now - lastFrame));
    this.frameTimeAvg += (dt - this.frameTimeAvg) * 0.1;

    let steps = 0;
    if (!this.paused && this.model && this.framesInFlight < 3) {
      // Hold roughly 60 frames per second: spend spare GPU time on more solver steps.
      if (this.frameTimeAvg < 18 && this.stepsPerFrame < MAX_STEPS_PER_FRAME) this.stepsPerFrame += 0.05;
      else if (this.frameTimeAvg > 21 && this.stepsPerFrame > 1) this.stepsPerFrame -= 0.25;
      steps = Math.max(1, Math.round(this.stepsPerFrame));
      this.lbm.advance(steps);
      this.framesInFlight++;
      this.device.queue.onSubmittedWorkDone().then(() => {
        this.framesInFlight--;
      });
      this.rateWindowSteps += steps;
    }
    if (now - this.rateWindowStart > 500) {
      this.stepRate = (this.rateWindowSteps * 1000) / (now - this.rateWindowStart);
      this.rateWindowStart = now;
      this.rateWindowSteps = 0;
    }

    const v = this.view;
    this.fields.update(steps, {
      particles: v.smokeLines,
      smoke: v.volumetricSmoke,
      vortex: v.vortices || (v.slice && v.sliceField === 'vorticity'),
    });
    this.draw();
  }

  /** Renders one frame without advancing the flow and returns it as an image. */
  capture(): string {
    this.draw();
    return this.renderer.capture();
  }

  snapshot(): Snapshot | null {
    if (!this.grid || !this.recorder) return null;
    const setup = this.setup();
    const measurement = this.recorder.measure();
    return {
      measurement,
      realWorld: realWorld(setup, measurement),
      flow: this.flow(),
      setup,
      grid: this.grid,
      tier: this.tier,
      stepsPerSecond: this.stepRate,
      totalSteps: this.lbm.totalSteps,
    };
  }

  history(): ForceRecorder['points'] {
    return this.recorder?.points ?? [];
  }

  private draw(): void {
    const aspect = this.renderer.resize();
    const m = this.camera.matrices(aspect);
    if (!this.placement) return;
    this.renderer.render({
      viewProj: m.viewProj,
      invViewProj: m.invViewProj,
      cameraPosition: m.position,
      modelMatrix: this.placement.matrix,
      settings: this.view,
    });
  }

  private rebuildGeometry(resetFlow: boolean): void {
    if (!this.model || !this.lbm) return;
    const { nx, ny, nz } = this.tier;
    this.placement = placeModel(this.model, this.orientation, nx, ny, nz);
    this.grid = voxelize(transformedTriangles(this.model, this.placement.matrix), nx, ny, nz);
    if (this.orientation.onGround) addFloor(this.grid);
    this.lbm.setSolid(this.grid.flags, this.grid.distance);
    this.fields.setReferenceLength(this.grid.extent[0]);
    this.applyFlow();
    if (resetFlow) {
      this.lbm.reset();
      this.fields.resetParticles();
    }
    this.restartMeasurement();
  }

  private restartMeasurement(): void {
    if (!this.grid) return;
    this.recorder = new ForceRecorder(
      { lengthCells: this.grid.extent[0], frontalCells: this.grid.frontalCells, blockage: this.grid.blockage },
      this.lbm.totalSteps,
    );
  }

  private applyFlow(): void {
    const f = this.flow();
    this.lbm.setFlow(f.latticeSpeed, f.tau);
  }

  private applyRake(): void {
    if (!this.fields) return;
    const { ny, nz } = this.tier;
    const r = this.rake;
    const count = Math.max(1, Math.round(r.count));
    let rows = 1, cols = 1;
    if (r.layout === 'vertical') rows = count;
    else if (r.layout === 'horizontal') cols = count;
    else {
      rows = count;
      cols = Math.max(1, Math.min(count, Math.floor(MAX_SEEDS / count)));
    }
    const spacing = (extent: number, n: number): number => (n > 1 ? (r.spread * extent) / (n - 1) : 0);
    const rake: Rake = {
      x: 2.5,
      y: 1.5 + r.height * (ny - 3),
      z: 1.5 + r.sideways * (nz - 3),
      rows,
      cols,
      dy: spacing(ny - 3, rows),
      dz: spacing(nz - 3, cols),
    };
    this.fields.setRake(rake);
  }

  private setup(): TunnelSetup {
    const grid = this.grid!;
    const placement = this.placement!;
    return {
      lengthCells: grid.extent[0],
      frontalCells: grid.frontalCells,
      blockage: grid.blockage,
      // The model's longest side spans `placement.scale` cells and `realSize` metres.
      metresPerCell: this.conditions.realSize / placement.scale,
      windSpeed: this.conditions.windKmh / 3.6,
      massKg: this.conditions.massKg,
    };
  }

  private flow(): FlowNumbers {
    return flowNumbers(this.setup());
  }
}
