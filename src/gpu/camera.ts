import { invert, lookAt, multiply, perspective, type Mat4, type Vec3 } from '../math';

/** Orbit camera: drag to rotate, wheel or pinch to zoom, right-drag or shift-drag to pan. */
export class OrbitCamera {
  target: Vec3 = [0, 0, 0];
  distance = 100;
  azimuth = -0.75;
  elevation = 0.32;
  private readonly fovY = (38 * Math.PI) / 180;
  private readonly pointers = new Map<number, { x: number; y: number }>();
  private pinchStart = 0;

  constructor(private readonly element: HTMLElement) {
    element.addEventListener('pointerdown', this.onDown);
    element.addEventListener('pointermove', this.onMove);
    element.addEventListener('pointerup', this.onUp);
    element.addEventListener('pointercancel', this.onUp);
    element.addEventListener('wheel', this.onWheel, { passive: false });
    element.addEventListener('contextmenu', (e) => e.preventDefault());
  }

  /** Frames a tunnel of the given size. */
  frame(nx: number, ny: number, nz: number): void {
    this.target = [nx * 0.5, ny * 0.4, nz * 0.5];
    this.distance = nx * 1.6;
  }

  get position(): Vec3 {
    const ce = Math.cos(this.elevation);
    return [
      this.target[0] + this.distance * ce * Math.sin(this.azimuth),
      this.target[1] + this.distance * Math.sin(this.elevation),
      this.target[2] + this.distance * ce * Math.cos(this.azimuth),
    ];
  }

  matrices(aspect: number): { viewProj: Mat4; invViewProj: Mat4; position: Vec3 } {
    const position = this.position;
    const view = lookAt(position, this.target, [0, 1, 0]);
    const proj = perspective(this.fovY, aspect, this.distance * 0.02, this.distance * 20);
    const viewProj = multiply(proj, view);
    return { viewProj, invViewProj: invert(viewProj), position };
  }

  private readonly onDown = (e: PointerEvent): void => {
    this.element.setPointerCapture(e.pointerId);
    this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (this.pointers.size === 2) this.pinchStart = this.pinchDistance();
  };

  private readonly onMove = (e: PointerEvent): void => {
    const last = this.pointers.get(e.pointerId);
    if (!last) return;
    const dx = e.clientX - last.x;
    const dy = e.clientY - last.y;
    this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });

    if (this.pointers.size === 2) {
      const d = this.pinchDistance();
      if (this.pinchStart > 0 && d > 0) this.zoom(this.pinchStart / d);
      this.pinchStart = d;
      return;
    }
    if (e.buttons === 2 || e.shiftKey) {
      this.pan(dx, dy);
      return;
    }
    this.azimuth -= dx * 0.006;
    this.elevation = Math.min(1.5, Math.max(-1.5, this.elevation + dy * 0.006));
  };

  private readonly onUp = (e: PointerEvent): void => {
    this.pointers.delete(e.pointerId);
    this.pinchStart = 0;
  };

  private readonly onWheel = (e: WheelEvent): void => {
    e.preventDefault();
    this.zoom(Math.exp(e.deltaY * 0.0012));
  };

  private zoom(factor: number): void {
    this.distance = Math.min(4000, Math.max(8, this.distance * factor));
  }

  private pan(dx: number, dy: number): void {
    const k = this.distance * 0.0012;
    const right: Vec3 = [Math.cos(this.azimuth), 0, -Math.sin(this.azimuth)];
    this.target = [
      this.target[0] - right[0] * dx * k,
      this.target[1] + dy * k,
      this.target[2] - right[2] * dx * k,
    ];
  }

  private pinchDistance(): number {
    const [a, b] = [...this.pointers.values()];
    return a && b ? Math.hypot(a.x - b.x, a.y - b.y) : 0;
  }
}
