import type { Object3D, Mesh, BufferGeometry, Material, MeshStandardMaterial } from 'three';
import { Matrix3, Vector3 } from 'three';
import { multiply, rotationYPR, translationScale, type Mat4 } from '../math.ts';

/** A model reduced to a triangle soup, centred on the origin with its longest side scaled to 1. */
export interface Model {
  name: string;
  /** xyz per vertex, three vertices per triangle. */
  positions: Float32Array;
  normals: Float32Array;
  colors: Float32Array;
  triangleCount: number;
  /** Bounding box size in the normalised frame. The largest component is 1. */
  size: [number, number, number];
  /** Suggested real-world longest dimension in metres. */
  realSize: number;
  /** Suggested mass in kilograms. */
  mass: number;
  /** Whether the model normally rests on the ground. */
  grounded: boolean;
}

export interface ModelDefaults {
  realSize?: number;
  mass?: number;
  grounded?: boolean;
}

export interface Orientation {
  yaw: number;
  pitch: number;
  roll: number;
  onGround: boolean;
}

export interface Placement {
  /** Maps normalised model space to lattice coordinates. */
  matrix: Mat4;
  /** Lattice cells per unit of normalised model space. */
  scale: number;
}

/** Largest share of the tunnel length and of its height and width that a model may fill. */
const FIT_LENGTH = 0.42;
const FIT_CROSS = 0.5;
/** Largest share of the tunnel cross-section that the model's bounding box may block. */
const FIT_BLOCKAGE = 0.12;

const DEFAULT_COLOR: [number, number, number] = [0.72, 0.74, 0.78];

export function modelFromObject(root: Object3D, name: string, defaults: ModelDefaults = {}): Model {
  root.updateMatrixWorld(true);
  const meshes: Mesh[] = [];
  root.traverse((node) => {
    const mesh = node as Mesh;
    if (mesh.isMesh && mesh.geometry?.attributes?.position) meshes.push(mesh);
  });

  let vertexTotal = 0;
  for (const mesh of meshes) {
    const g = mesh.geometry as BufferGeometry;
    const count = g.index ? g.index.count : g.attributes.position.count;
    vertexTotal += count - (count % 3);
  }
  if (vertexTotal === 0) throw new Error('The file contains no triangle meshes.');

  const positions = new Float32Array(vertexTotal * 3);
  const normals = new Float32Array(vertexTotal * 3);
  const colors = new Float32Array(vertexTotal * 3);
  const v = new Vector3();
  const normalMatrix = new Matrix3();
  let cursor = 0;

  for (const mesh of meshes) {
    const g = mesh.geometry as BufferGeometry;
    if (!g.attributes.normal) g.computeVertexNormals();
    const pos = g.attributes.position;
    const nrm = g.attributes.normal;
    const index = g.index;
    const count = index ? index.count : pos.count;
    const usable = count - (count % 3);
    const color = materialColor(mesh.material);
    normalMatrix.getNormalMatrix(mesh.matrixWorld);

    for (let i = 0; i < usable; i++) {
      const vi = index ? index.getX(i) : i;
      v.fromBufferAttribute(pos, vi).applyMatrix4(mesh.matrixWorld);
      positions[cursor] = v.x; positions[cursor + 1] = v.y; positions[cursor + 2] = v.z;
      v.fromBufferAttribute(nrm, vi).applyMatrix3(normalMatrix).normalize();
      normals[cursor] = v.x; normals[cursor + 1] = v.y; normals[cursor + 2] = v.z;
      colors[cursor] = color[0]; colors[cursor + 1] = color[1]; colors[cursor + 2] = color[2];
      cursor += 3;
    }
  }

  const size = normaliseInPlace(positions);
  return {
    name,
    positions,
    normals,
    colors,
    triangleCount: vertexTotal / 3,
    size,
    realSize: defaults.realSize ?? 1,
    mass: defaults.mass ?? 10,
    grounded: defaults.grounded ?? false,
  };
}

function materialColor(material: Material | Material[]): [number, number, number] {
  const m = (Array.isArray(material) ? material[0] : material) as MeshStandardMaterial | undefined;
  if (!m?.color) return DEFAULT_COLOR;
  return [m.color.r, m.color.g, m.color.b];
}

/** Centres the points on their bounding box and scales the longest side to 1. Returns the new size. */
function normaliseInPlace(p: Float32Array): [number, number, number] {
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < p.length; i += 3) {
    for (let a = 0; a < 3; a++) {
      const c = p[i + a];
      if (!Number.isFinite(c)) continue;
      if (c < lo[a]) lo[a] = c;
      if (c > hi[a]) hi[a] = c;
    }
  }
  const ext = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
  const longest = Math.max(ext[0], ext[1], ext[2]);
  if (!(longest > 0) || !Number.isFinite(longest)) throw new Error('The model has no measurable size.');
  const s = 1 / longest;
  for (let i = 0; i < p.length; i += 3) {
    for (let a = 0; a < 3; a++) {
      const c = p[i + a];
      p[i + a] = Number.isFinite(c) ? (c - (lo[a] + hi[a]) / 2) * s : 0;
    }
  }
  return [ext[0] * s, ext[1] * s, ext[2] * s];
}

/** Places the rotated model in the tunnel: scaled to fit, upstream of centre, optionally on the floor. */
export function placeModel(model: Model, o: Orientation, nx: number, ny: number, nz: number): Placement {
  const rot = rotationYPR(o.yaw, o.pitch, o.roll);
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  const p = model.positions;
  // Sampling every vertex is cheap, but very dense models only need a subset for a bounding box.
  const step = Math.max(1, Math.floor(p.length / 3 / 200000)) * 3;
  for (let i = 0; i < p.length; i += step) {
    const x = p[i], y = p[i + 1], z = p[i + 2];
    for (let a = 0; a < 3; a++) {
      const c = rot[a] * x + rot[4 + a] * y + rot[8 + a] * z;
      if (c < lo[a]) lo[a] = c;
      if (c > hi[a]) hi[a] = c;
    }
  }
  const ext = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]].map((e) => Math.max(e, 1e-3));
  const scale = Math.min(
    (FIT_LENGTH * nx) / ext[0],
    (FIT_CROSS * ny) / ext[1],
    (FIT_CROSS * nz) / ext[2],
    Math.sqrt((FIT_BLOCKAGE * ny * nz) / (ext[1] * ext[2])),
  );
  const cx = (lo[0] + hi[0]) / 2, cy = (lo[1] + hi[1]) / 2, cz = (lo[2] + hi[2]) / 2;
  const tx = 0.36 * nx - cx * scale;
  const ty = o.onGround ? 1.02 - lo[1] * scale : ny / 2 - cy * scale;
  const tz = nz / 2 - cz * scale;
  return { matrix: multiply(translationScale([tx, ty, tz], scale), rot), scale };
}

/** Transforms the model's triangles into lattice coordinates for voxelisation. */
export function transformedTriangles(model: Model, m: Mat4): Float32Array {
  const p = model.positions;
  const out = new Float32Array(p.length);
  for (let i = 0; i < p.length; i += 3) {
    const x = p[i], y = p[i + 1], z = p[i + 2];
    out[i] = m[0] * x + m[4] * y + m[8] * z + m[12];
    out[i + 1] = m[1] * x + m[5] * y + m[9] * z + m[13];
    out[i + 2] = m[2] * x + m[6] * y + m[10] * z + m[14];
  }
  return out;
}
