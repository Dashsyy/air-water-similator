import {
  BoxGeometry,
  Color,
  ConeGeometry,
  CylinderGeometry,
  ExtrudeGeometry,
  Group,
  LatheGeometry,
  Mesh,
  MeshStandardMaterial,
  Shape,
  SphereGeometry,
  Vector2,
  type BufferGeometry,
} from 'three';
import { modelFromObject, type Model } from './model.ts';

export interface Preset {
  id: string;
  label: string;
  emoji: string;
  /** Textbook drag coefficient for shapes that serve as validation cases. */
  expectedCd?: number;
  expectedNote?: string;
  pitch?: number;
  build: () => Model;
}

function part(geometry: BufferGeometry, color: number, x = 0, y = 0, z = 0): Mesh {
  const mesh = new Mesh(geometry, new MeshStandardMaterial({ color: new Color(color) }));
  mesh.position.set(x, y, z);
  return mesh;
}

/** NACA four-digit half thickness at chord fraction x, with a closed trailing edge. */
function nacaThickness(x: number, t: number): number {
  return 5 * t * (0.2969 * Math.sqrt(x) - 0.126 * x - 0.3516 * x * x + 0.2843 * x ** 3 - 0.1036 * x ** 4);
}

function buildSphere(): Model {
  const g = new Group();
  g.add(part(new SphereGeometry(0.5, 64, 40), 0xd9dde3));
  return modelFromObject(g, 'Sphere', { realSize: 0.22, mass: 0.43 });
}

function buildCube(): Model {
  const g = new Group();
  g.add(part(new BoxGeometry(1, 1, 1), 0xc8a26a));
  return modelFromObject(g, 'Cube', { realSize: 0.3, mass: 5 });
}

function buildCar(): Model {
  const g = new Group();
  const s = new Shape();
  s.moveTo(-2.1, 0.22);
  s.lineTo(2.1, 0.22);
  s.quadraticCurveTo(2.25, 0.3, 2.22, 0.62);
  s.lineTo(2.15, 0.9);
  s.lineTo(1.55, 0.96);
  s.quadraticCurveTo(1.15, 1.36, 0.75, 1.4);
  s.lineTo(-0.2, 1.42);
  s.quadraticCurveTo(-0.6, 1.36, -1.05, 0.98);
  s.lineTo(-1.95, 0.8);
  s.quadraticCurveTo(-2.28, 0.7, -2.25, 0.42);
  s.quadraticCurveTo(-2.24, 0.24, -2.1, 0.22);
  const body = new ExtrudeGeometry(s, {
    depth: 1.5,
    bevelEnabled: true,
    bevelSize: 0.12,
    bevelThickness: 0.14,
    bevelSegments: 4,
    curveSegments: 10,
  });
  body.translate(0, 0, -0.75);
  g.add(part(body, 0xc0392b));
  for (const x of [-1.4, 1.35]) {
    for (const z of [-0.8, 0.8]) {
      const wheel = new CylinderGeometry(0.34, 0.34, 0.26, 28);
      wheel.rotateX(Math.PI / 2);
      g.add(part(wheel, 0x1b1d21, x, 0.34, z));
    }
  }
  return modelFromObject(g, 'Car', { realSize: 4.5, mass: 1400, grounded: true });
}

function buildWing(): Model {
  const camber = 0.04, camberPos = 0.4, thickness = 0.18;
  const upper: Vector2[] = [];
  const lower: Vector2[] = [];
  const n = 40;
  for (let i = 0; i <= n; i++) {
    const x = 0.5 * (1 - Math.cos((Math.PI * i) / n));
    const yc = x < camberPos
      ? (camber / camberPos ** 2) * (2 * camberPos * x - x * x)
      : (camber / (1 - camberPos) ** 2) * (1 - 2 * camberPos + 2 * camberPos * x - x * x);
    const yt = nacaThickness(x, thickness);
    upper.push(new Vector2(x, yc + yt));
    lower.push(new Vector2(x, yc - yt));
  }
  const s = new Shape();
  s.moveTo(upper[0].x, upper[0].y);
  for (let i = 1; i <= n; i++) s.lineTo(upper[i].x, upper[i].y);
  for (let i = n - 1; i >= 1; i--) s.lineTo(lower[i].x, lower[i].y);
  s.closePath();
  const wing = new ExtrudeGeometry(s, { depth: 1.5, bevelEnabled: false });
  wing.translate(-0.5, 0, -0.75);
  const g = new Group();
  g.add(part(wing, 0x8fb3d9));
  return modelFromObject(g, 'Wing section', { realSize: 3, mass: 60 });
}

function buildCow(): Model {
  const g = new Group();
  const white = 0xf2efe9, dark = 0x2c2a29, pink = 0xe8a7a1;
  const ellipsoid = (r: number, sx: number, sy: number, sz: number): SphereGeometry => {
    const geo = new SphereGeometry(r, 36, 24);
    geo.scale(sx, sy, sz);
    return geo;
  };
  g.add(part(ellipsoid(1, 1.05, 0.56, 0.5), white, 0, 1.3, 0));
  g.add(part(ellipsoid(0.5, 1, 1, 1), dark, 0.35, 1.52, 0.18));
  const neck = new CylinderGeometry(0.26, 0.34, 0.6, 20);
  neck.rotateZ(-0.9);
  g.add(part(neck, white, -1.0, 1.6, 0));
  g.add(part(ellipsoid(0.33, 1.25, 1, 0.9), white, -1.38, 1.86, 0));
  g.add(part(ellipsoid(0.2, 1.1, 0.85, 1), pink, -1.74, 1.74, 0));
  for (const z of [-0.2, 0.2]) {
    const horn = new ConeGeometry(0.05, 0.24, 12);
    g.add(part(horn, 0xd8cfae, -1.26, 2.24, z));
    g.add(part(ellipsoid(0.12, 0.5, 0.8, 1.3), dark, -1.18, 2.02, z * 2));
  }
  for (const x of [-0.68, 0.68]) {
    for (const z of [-0.26, 0.26]) {
      g.add(part(new CylinderGeometry(0.115, 0.1, 0.95, 16), white, x, 0.52, z));
      g.add(part(new CylinderGeometry(0.105, 0.11, 0.12, 16), dark, x, 0.06, z));
    }
  }
  g.add(part(ellipsoid(0.2, 1.1, 0.8, 1), pink, 0.42, 0.78, 0));
  const tail = new CylinderGeometry(0.035, 0.035, 0.95, 10);
  tail.rotateZ(0.22);
  g.add(part(tail, white, 1.14, 1.05, 0));
  g.add(part(ellipsoid(0.08, 1, 1.5, 1), dark, 1.25, 0.56, 0));
  return modelFromObject(g, 'Cow', { realSize: 2.4, mass: 650, grounded: true });
}

function buildChair(): Model {
  const g = new Group();
  const wood = 0xa9743b;
  g.add(part(new BoxGeometry(0.46, 0.05, 0.46), wood, 0, 0.45, 0));
  g.add(part(new BoxGeometry(0.05, 0.5, 0.46), wood, 0.205, 0.725, 0));
  for (const x of [-0.2, 0.2]) {
    for (const z of [-0.2, 0.2]) {
      g.add(part(new BoxGeometry(0.05, 0.45, 0.05), wood, x, 0.225, z));
    }
  }
  return modelFromObject(g, 'Chair', { realSize: 0.98, mass: 6, grounded: true });
}

function buildTeardrop(): Model {
  const pts: Vector2[] = [];
  const n = 60;
  for (let i = 0; i <= n; i++) {
    const x = 0.5 * (1 - Math.cos((Math.PI * i) / n));
    pts.push(new Vector2(Math.max(nacaThickness(x, 0.28), 0), x));
  }
  const geo = new LatheGeometry(pts, 48);
  geo.rotateZ(-Math.PI / 2);
  const g = new Group();
  g.add(part(geo, 0x6fbf9a));
  return modelFromObject(g, 'Teardrop', { realSize: 1, mass: 8 });
}

function buildPlate(): Model {
  const g = new Group();
  g.add(part(new BoxGeometry(0.07, 1, 1), 0x9aa3ad));
  return modelFromObject(g, 'Flat plate', { realSize: 1, mass: 12 });
}

export const PRESETS: Preset[] = [
  { id: 'car', label: 'Car', emoji: '🚗', build: buildCar },
  { id: 'cow', label: 'Cow', emoji: '🐄', build: buildCow },
  { id: 'wing', label: 'Wing', emoji: '✈️', pitch: 8, build: buildWing },
  { id: 'chair', label: 'Chair', emoji: '🪑', build: buildChair },
  {
    id: 'sphere',
    label: 'Sphere',
    emoji: '⚽',
    expectedCd: 0.47,
    expectedNote: 'smooth sphere below the drag crisis',
    build: buildSphere,
  },
  { id: 'cube', label: 'Cube', emoji: '📦', expectedCd: 1.05, expectedNote: 'cube facing the flow', build: buildCube },
  {
    id: 'plate',
    label: 'Flat plate',
    emoji: '🧱',
    expectedCd: 1.17,
    expectedNote: 'square plate normal to the flow',
    build: buildPlate,
  },
  {
    id: 'teardrop',
    label: 'Teardrop',
    emoji: '💧',
    expectedCd: 0.05,
    expectedNote: 'streamlined body; coarse grids overpredict this by a wide margin',
    build: buildTeardrop,
  },
];
