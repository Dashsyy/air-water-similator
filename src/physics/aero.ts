/**
 * Converts raw lattice forces into aerodynamic coefficients and real-world numbers.
 *
 * Coefficients use the frontal area the solver actually sees (the voxel silhouette) and are
 * corrected for tunnel blockage: the flow past the model speeds up by roughly 1 / (1 - B),
 * so measured forces are scaled by (1 - B)^2.
 */
import type { ForceSample } from '../gpu/lbm3d';

export const AIR_DENSITY = 1.225;
export const AIR_KINEMATIC_VISCOSITY = 1.5e-5;
export const GRAVITY = 9.81;

/** Lattice inflow speed. Mach number is about 0.1, low enough for near-incompressible flow. */
export const LATTICE_SPEED = 0.06;
/** Smallest lattice viscosity the solver is run at. Caps the simulated Reynolds number. */
export const MIN_LATTICE_VISCOSITY = 3e-4;
/** Passes of air over the model's length that are discarded before averaging starts. */
export const WARMUP_FLOW_THROUGHS = 5;
/** Flow-throughs of averaging after which the measurement counts as settled. */
export const SETTLE_FLOW_THROUGHS = 8;

export interface TunnelSetup {
  /** Characteristic model length in cells: its longest side as placed in the tunnel. */
  lengthCells: number;
  frontalCells: number;
  blockage: number;
  /** Real-world size of one cell in metres. */
  metresPerCell: number;
  /** Real-world wind speed in metres per second. */
  windSpeed: number;
  massKg: number;
}

export interface FlowNumbers {
  latticeSpeed: number;
  tau: number;
  realReynolds: number;
  simulatedReynolds: number;
}

export function flowNumbers(setup: TunnelSetup): FlowNumbers {
  const realLength = setup.lengthCells * setup.metresPerCell;
  const realReynolds = (setup.windSpeed * realLength) / AIR_KINEMATIC_VISCOSITY;
  const wanted = (LATTICE_SPEED * setup.lengthCells) / Math.max(realReynolds, 1);
  const viscosity = Math.max(wanted, MIN_LATTICE_VISCOSITY);
  return {
    latticeSpeed: LATTICE_SPEED,
    tau: 3 * viscosity + 0.5,
    realReynolds,
    simulatedReynolds: (LATTICE_SPEED * setup.lengthCells) / viscosity,
  };
}

export interface CoefficientPoint {
  /** Time in flow-throughs of the model length. */
  t: number;
  cd: number;
  cl: number;
  cs: number;
  steps: number;
}

export interface Measurement {
  cd: number;
  cl: number;
  cs: number;
  /** Standard deviation of the drag coefficient over the averaging window. */
  cdSpread: number;
  liftToDrag: number;
  /** Flow-throughs simulated so far. */
  time: number;
  /** Flow-throughs that went into the average. */
  averagedTime: number;
  /** 0 to 1: how far the measurement is towards being settled. */
  progress: number;
  settled: boolean;
}

/** Collects force samples for one geometry and averages them after a warm-up. */
export class ForceRecorder {
  readonly points: CoefficientPoint[] = [];
  private readonly dynamicForce: number;
  private readonly correction: number;
  private readonly stepsPerFlowThrough: number;

  /**
   * @param originStep Solver step count at which this geometry started being measured.
   */
  constructor(
    setup: Pick<TunnelSetup, 'lengthCells' | 'frontalCells' | 'blockage'>,
    private readonly originStep = 0,
    latticeSpeed = LATTICE_SPEED,
  ) {
    this.dynamicForce = 0.5 * latticeSpeed * latticeSpeed * Math.max(setup.frontalCells, 1);
    this.correction = (1 - Math.min(setup.blockage, 0.5)) ** 2;
    this.stepsPerFlowThrough = Math.max(setup.lengthCells, 1) / latticeSpeed;
  }

  add(sample: ForceSample): void {
    const k = this.correction / this.dynamicForce;
    const [fx, fy, fz] = sample.force;
    if (!Number.isFinite(fx + fy + fz)) return;
    this.points.push({
      t: Math.max(0, sample.atStep - this.originStep) / this.stepsPerFlowThrough,
      cd: fx * k,
      cl: fy * k,
      cs: fz * k,
      steps: sample.steps,
    });
  }

  measure(): Measurement {
    const last = this.points[this.points.length - 1];
    const time = last ? last.t : 0;
    let weight = 0, cd = 0, cl = 0, cs = 0, cd2 = 0;
    for (const p of this.points) {
      if (p.t < WARMUP_FLOW_THROUGHS) continue;
      weight += p.steps;
      cd += p.cd * p.steps;
      cl += p.cl * p.steps;
      cs += p.cs * p.steps;
      cd2 += p.cd * p.cd * p.steps;
    }
    const averagedTime = Math.max(0, time - WARMUP_FLOW_THROUGHS);
    const progress = Math.min(1, time / (WARMUP_FLOW_THROUGHS + SETTLE_FLOW_THROUGHS));
    if (weight === 0) {
      // Still warming up: show the latest instantaneous values so the read-out is alive.
      return {
        cd: last?.cd ?? 0, cl: last?.cl ?? 0, cs: last?.cs ?? 0,
        cdSpread: 0, liftToDrag: 0, time, averagedTime: 0, progress, settled: false,
      };
    }
    cd /= weight; cl /= weight; cs /= weight;
    const variance = Math.max(0, cd2 / weight - cd * cd);
    return {
      cd, cl, cs,
      cdSpread: Math.sqrt(variance),
      liftToDrag: Math.abs(cd) > 1e-6 ? cl / cd : 0,
      time, averagedTime, progress,
      settled: averagedTime >= SETTLE_FLOW_THROUGHS,
    };
  }
}

export interface RealWorld {
  lengthM: number;
  frontalAreaM2: number;
  dragN: number;
  liftN: number;
  sideN: number;
  powerW: number;
  weightN: number;
  /** Speed at which drag balances weight, in m/s. Null when drag is not positive. */
  terminalSpeed: number | null;
  /** Speed at which lift balances weight, in m/s. Null when the shape makes no upward lift. */
  takeoffSpeed: number | null;
}

export function realWorld(setup: TunnelSetup, m: Pick<Measurement, 'cd' | 'cl' | 'cs'>): RealWorld {
  const area = setup.frontalCells * setup.metresPerCell ** 2;
  const q = 0.5 * AIR_DENSITY * setup.windSpeed ** 2 * area;
  const weight = setup.massKg * GRAVITY;
  const speedFor = (coefficient: number): number | null =>
    coefficient > 1e-4 && area > 0 ? Math.sqrt((2 * weight) / (AIR_DENSITY * area * coefficient)) : null;
  return {
    lengthM: setup.lengthCells * setup.metresPerCell,
    frontalAreaM2: area,
    dragN: q * m.cd,
    liftN: q * m.cl,
    sideN: q * m.cs,
    powerW: q * m.cd * setup.windSpeed,
    weightN: weight,
    terminalSpeed: speedFor(m.cd),
    takeoffSpeed: speedFor(m.cl),
  };
}
