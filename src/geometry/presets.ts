import {
  BoxGeometry,
  BufferGeometry,
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
  Float32BufferAttribute,
  Vector2,
  Vector3,
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

/**
 * Joins closed rings of equal length into a capped tube. The rings may run in either direction;
 * the faces are flipped if needed so they point outward, which keeps the voxeliser's winding
 * numbers from cancelling where parts overlap.
 */
function loft(rings: Vector3[][]): BufferGeometry {
  const n = rings[0].length;
  const pos: number[] = [];
  for (const ring of rings) for (const p of ring) pos.push(p.x, p.y, p.z);
  const index: number[] = [];
  for (let r = 0; r + 1 < rings.length; r++) {
    for (let i = 0; i < n; i++) {
      const a = r * n + i, b = r * n + ((i + 1) % n), c = a + n, d = b + n;
      index.push(a, b, d, a, d, c);
    }
  }
  for (const [r, flip] of [[0, true], [rings.length - 1, false]] as const) {
    const centre = new Vector3();
    for (const p of rings[r]) centre.add(p);
    centre.divideScalar(n);
    const k = pos.length / 3;
    pos.push(centre.x, centre.y, centre.z);
    for (let i = 0; i < n; i++) {
      const a = r * n + i, b = r * n + ((i + 1) % n);
      index.push(...(flip ? [k, b, a] : [k, a, b]));
    }
  }
  let volume = 0;
  for (let t = 0; t < index.length; t += 3) {
    const [a, b, c] = [index[t] * 3, index[t + 1] * 3, index[t + 2] * 3];
    volume +=
      pos[a] * (pos[b + 1] * pos[c + 2] - pos[b + 2] * pos[c + 1]) -
      pos[a + 1] * (pos[b] * pos[c + 2] - pos[b + 2] * pos[c]) +
      pos[a + 2] * (pos[b] * pos[c + 1] - pos[b + 1] * pos[c]);
  }
  if (volume < 0) {
    for (let t = 0; t < index.length; t += 3) [index[t + 1], index[t + 2]] = [index[t + 2], index[t + 1]];
  }
  const geo = new BufferGeometry();
  geo.setAttribute('position', new Float32BufferAttribute(pos, 3));
  geo.setIndex(index);
  geo.computeVertexNormals();
  return geo;
}

/** Symmetric NACA 00xx outline as (chord fraction, half thickness) pairs, leading edge first round the top. */
function airfoilOutline(thickness: number, n = 16): Array<[number, number]> {
  const out: Array<[number, number]> = [];
  for (let i = 0; i <= n; i++) {
    const x = 0.5 * (1 - Math.cos((Math.PI * i) / n));
    out.push([x, nacaThickness(x, thickness)]);
  }
  for (let i = n - 1; i >= 1; i--) {
    const x = 0.5 * (1 - Math.cos((Math.PI * i) / n));
    out.push([x, -nacaThickness(x, thickness)]);
  }
  return out;
}

/**
 * A lifting surface between two stations. Each station gives its leading edge, chord and the
 * direction its thickness points (normal to the surface); the chord always runs along +x.
 */
function surface(
  stations: Array<{ le: [number, number, number]; chord: number }>,
  thickness: number,
  up: [number, number, number] = [0, 1, 0],
): BufferGeometry {
  const outline = airfoilOutline(thickness);
  const u = new Vector3(...up).normalize();
  return loft(
    stations.map(({ le, chord }) =>
      outline.map(([x, t]) => new Vector3(le[0] + x * chord, le[1], le[2]).addScaledVector(u, t * chord)),
    ),
  );
}

/** A strut of airfoil section between two points, with its thickness lying in the horizontal plane. */
function strut(from: Vector3, to: Vector3, chord: number, thickness: number): BufferGeometry {
  const along = to.clone().sub(from).normalize();
  const side = new Vector3(1, 0, 0).cross(along).normalize();
  const outline = airfoilOutline(thickness, 8);
  return loft(
    [from, to].map((p) =>
      outline.map(([x, t]) => p.clone().add(new Vector3((x - 0.4) * chord, 0, 0)).addScaledVector(side, t * chord)),
    ),
  );
}

/**
 * NASA and Boeing's Transonic Truss-Braced Wing (SUGAR / X-66 family), sized like the
 * 737-class concept: a high, very slender wing held up by struts from the lower fuselage.
 * The nose points upstream (-x), the wing spans z.
 */
function buildTrussJet(): Model {
  const g = new Group();
  const body = 0xdfe3e8, wingGrey = 0xc3c9d0, dark = 0x5d6670, blue = 0x1f4f9e;
  const length = 40, radius = 1.95;

  // Fuselage: rounded nose, straight barrel, tapered tail cone.
  const profile: Vector2[] = [];
  for (let i = 0; i <= 64; i++) {
    const x = (length * i) / 64;
    let r = radius;
    if (x < 6) r = radius * Math.sqrt(1 - ((6 - x) / 6) ** 2);
    else if (x > 28) r = radius * (1 - 0.82 * ((x - 28) / 12) ** 1.4);
    profile.push(new Vector2(Math.max(r, 0.001), x));
  }
  profile[0].x = 0.001;
  const fuselage = new LatheGeometry(profile, 40);
  fuselage.rotateZ(-Math.PI / 2);
  g.add(part(fuselage, body));

  // Main gear fairings on the lower fuselage.
  for (const z of [-1.45, 1.45]) {
    const pod = new SphereGeometry(1, 28, 16);
    pod.scale(3.4, 0.75, 0.8);
    g.add(part(pod, body, 18.5, -1.35, z));
  }

  // High wing: about 52 m span, aspect ratio near 19, light sweep, root above the fuselage.
  const semi = 26, sweep = Math.tan((14 * Math.PI) / 180), dihedral = Math.tan((1.5 * Math.PI) / 180);
  const rootLe = 14, rootChord = 4.4, tipChord = 1.4, wingY = radius - 0.05;
  const wingAt = (z: number): { le: [number, number, number]; chord: number } => {
    const f = Math.abs(z) / semi;
    const chord = rootChord + (tipChord - rootChord) * f;
    // Sweep the quarter-chord line so the taper does not skew it.
    const quarter = rootLe + 0.25 * rootChord + Math.abs(z) * sweep;
    return { le: [quarter - 0.25 * chord, wingY + Math.abs(z) * dihedral, z], chord };
  };
  const spanStations = [-26, -20, -14, -8, -2, 0, 2, 8, 14, 20, 26];
  g.add(part(surface(spanStations.map(wingAt), 0.12), wingGrey));

  // Struts from the gear fairings to 55% semi-span, with a short jury strut near their upper end.
  for (const side of [-1, 1]) {
    const zs = 0.55 * semi * side;
    const top = wingAt(zs);
    const upper = new Vector3(top.le[0] + 0.4 * top.chord, top.le[1] - 0.1, zs);
    const lower = new Vector3(19.2, -1.3, 1.6 * side);
    g.add(part(strut(lower, upper, 1.3, 0.16), wingGrey));
    const juryBase = lower.clone().lerp(upper, 0.62);
    const juryTop = wingAt(juryBase.z);
    g.add(
      part(
        strut(juryBase, new Vector3(juryBase.x, juryTop.le[1] - 0.05, juryBase.z), 0.7, 0.2),
        wingGrey,
      ),
    );

    // Underwing turbofans on short pylons.
    const ze = 6.2 * side;
    const w = wingAt(ze);
    const nacelleProfile = [
      [0.95, 0], [1.12, 0.3], [1.15, 1.4], [1.05, 3.2], [0.8, 4.6], [0.45, 5.4], [0.001, 5.6],
    ].map(([r, x]) => new Vector2(r, x));
    nacelleProfile.unshift(new Vector2(0.001, 0));
    const nacelle = new LatheGeometry(nacelleProfile, 28);
    nacelle.rotateZ(-Math.PI / 2);
    const ex = w.le[0] - 2.4, ey = w.le[1] - 1.75;
    g.add(part(nacelle, dark, ex, ey, ze));
    g.add(part(surface([{ le: [ex + 1.2, ey + 0.9, ze], chord: 3.6 }, { le: [w.le[0] - 0.2, w.le[1], ze], chord: 3.2 }], 0.1, [0, 0, 1]), wingGrey));
  }

  // T-tail: swept fin with the tailplane on top.
  const finRoot = { le: [31.5, 0.3, 0] as [number, number, number], chord: 6.2 };
  const finTip = { le: [37.2, 7.6, 0] as [number, number, number], chord: 3.4 };
  g.add(part(surface([finRoot, finTip], 0.11, [0, 0, 1]), blue));
  const tailSweep = Math.tan((28 * Math.PI) / 180);
  const tailAt = (z: number): { le: [number, number, number]; chord: number } => {
    const f = Math.abs(z) / 6.5;
    return { le: [37.0 + Math.abs(z) * tailSweep, 7.55, z], chord: 3.2 - 1.6 * f };
  };
  g.add(part(surface([-6.5, -2, 0, 2, 6.5].map(tailAt), 0.1), blue));

  return modelFromObject(g, 'Truss-braced wing jet', { realSize: 52, mass: 68000 });
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
  { id: 'truss-jet', label: 'Truss jet', emoji: '🛩️', pitch: 2, build: buildTrussJet },
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
