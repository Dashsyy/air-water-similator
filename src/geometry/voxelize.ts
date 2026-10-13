/**
 * Turns a triangle soup into a solid/fluid voxel grid.
 *
 * For each of the three axes, every triangle is rasterised onto the plane normal to that axis.
 * Each crossing adds a signed count to all cells behind it, which yields a winding number per cell.
 * A cell is solid when at least two axes agree it is inside, which tolerates holes and stray faces.
 * Cells that contain a surface crossing but have no inside neighbour are also solid, so parts thinner
 * than a cell survive without fattening the rest of the body.
 */
export interface VoxelGrid {
  nx: number;
  ny: number;
  nz: number;
  /** 0 for fluid, 1 for the model, 2 for a tunnel wall. Indexed x + nx * (y + ny * z). */
  flags: Uint32Array;
  solidCells: number;
  /** Number of (y, z) columns that contain solid: the frontal area in cell faces. */
  frontalCells: number;
  /** Solid extent in cells along each axis. */
  extent: [number, number, number];
  /** Frontal area divided by the tunnel cross-section. */
  blockage: number;
  /**
   * Unsigned distance from each cell centre to the model surface, in cells, where it is known
   * (within about two cells of the surface). -1 elsewhere. The solver uses it to place the wall
   * between cell centres instead of always halfway.
   */
  distance: Float32Array;
}

const EPS_U = 1.37e-5;
const EPS_V = 2.91e-5;

export function voxelize(triangles: Float32Array, nx: number, ny: number, nz: number): VoxelGrid {
  const dims = [nx, ny, nz];
  const stride = [1, nx, nx * ny];
  const cells = nx * ny * nz;
  const votes = new Uint8Array(cells);
  const surface = new Uint8Array(cells);
  const diff = new Int16Array(cells);
  // Distance from each cell centre to the nearest surface crossing along each axis.
  const reach = [new Float32Array(cells), new Float32Array(cells), new Float32Array(cells)];
  for (const r of reach) r.fill(Infinity);
  const triCount = Math.floor(triangles.length / 9);

  for (let axis = 0; axis < 3; axis++) {
    const ua = (axis + 1) % 3;
    const va = (axis + 2) % 3;
    const na = dims[axis], nu = dims[ua], nv = dims[va];
    const sa = stride[axis], su = stride[ua], sv = stride[va];
    const axisReach = reach[axis];
    diff.fill(0);

    for (let t = 0; t < triCount; t++) {
      const o = t * 9;
      const u0 = triangles[o + ua], v0 = triangles[o + va], a0 = triangles[o + axis];
      const u1 = triangles[o + 3 + ua], v1 = triangles[o + 3 + va], a1 = triangles[o + 3 + axis];
      const u2 = triangles[o + 6 + ua], v2 = triangles[o + 6 + va], a2 = triangles[o + 6 + axis];
      const area = (u1 - u0) * (v2 - v0) - (u2 - u0) * (v1 - v0);
      if (Math.abs(area) < 1e-12) continue;
      const sign = area > 0 ? 1 : -1;
      const invArea = 1 / area;

      const iMin = Math.max(0, Math.ceil(Math.min(u0, u1, u2) - 0.5 - EPS_U));
      const iMax = Math.min(nu - 1, Math.floor(Math.max(u0, u1, u2) - 0.5 - EPS_U));
      const jMin = Math.max(0, Math.ceil(Math.min(v0, v1, v2) - 0.5 - EPS_V));
      const jMax = Math.min(nv - 1, Math.floor(Math.max(v0, v1, v2) - 0.5 - EPS_V));

      for (let j = jMin; j <= jMax; j++) {
        const pv = j + 0.5 + EPS_V;
        for (let i = iMin; i <= iMax; i++) {
          const pu = i + 0.5 + EPS_U;
          const w0 = ((u1 - pu) * (v2 - pv) - (u2 - pu) * (v1 - pv)) * invArea;
          const w1 = ((u2 - pu) * (v0 - pv) - (u0 - pu) * (v2 - pv)) * invArea;
          const w2 = 1 - w0 - w1;
          if (w0 < 0 || w1 < 0 || w2 < 0) continue;
          const depth = w0 * a0 + w1 * a1 + w2 * a2;
          const column = i * su + j * sv;

          const cell = Math.floor(depth);
          if (cell >= 0 && cell < na) surface[column + cell * sa] = 1;

          const kFrom = Math.max(0, Math.floor(depth - 2.5));
          const kTo = Math.min(na - 1, Math.floor(depth + 1.5));
          for (let k = kFrom; k <= kTo; k++) {
            const gap = Math.abs(depth - (k + 0.5));
            const at = column + k * sa;
            if (gap < axisReach[at]) axisReach[at] = gap;
          }

          // Every cell whose centre lies before this crossing gets the signed count.
          let last = Math.ceil(depth - 0.5) - 1;
          if (last < 0) continue;
          if (last > na - 1) last = na - 1;
          diff[column + last * sa] += sign;
        }
      }
    }

    for (let j = 0; j < nv; j++) {
      for (let i = 0; i < nu; i++) {
        const column = i * su + j * sv;
        let winding = 0;
        for (let k = na - 1; k >= 0; k--) {
          const idx = column + k * sa;
          winding += diff[idx];
          if (winding !== 0) votes[idx]++;
        }
      }
    }
  }

  // For a locally flat surface with axis intercepts a, b, c the true distance d obeys
  // 1/d^2 = 1/a^2 + 1/b^2 + 1/c^2.
  const distance = new Float32Array(cells);
  for (let i = 0; i < cells; i++) {
    const a = reach[0][i], b = reach[1][i], c = reach[2][i];
    if (a === Infinity && b === Infinity && c === Infinity) {
      distance[i] = -1;
      continue;
    }
    const inv = 1 / (a * a + 1e-12) + 1 / (b * b + 1e-12) + 1 / (c * c + 1e-12);
    distance[i] = 1 / Math.sqrt(inv);
  }

  const flags = new Uint32Array(cells);
  const frontal = new Uint8Array(ny * nz);
  let solidCells = 0;
  const lo = [nx, ny, nz];
  const hi = [-1, -1, -1];
  for (let z = 1; z < nz - 1; z++) {
    for (let y = 1; y < ny - 1; y++) {
      const row = nx * (y + ny * z);
      for (let x = 1; x < nx - 1; x++) {
        const idx = row + x;
        if (votes[idx] < 2) {
          if (!surface[idx]) continue;
          const attached =
            votes[idx - 1] >= 2 || votes[idx + 1] >= 2 ||
            votes[idx - nx] >= 2 || votes[idx + nx] >= 2 ||
            votes[idx - nx * ny] >= 2 || votes[idx + nx * ny] >= 2;
          if (attached) continue;
          // A thin shell: the surface runs through this cell, so its wall position is unknown.
          distance[idx] = -1;
        }
        flags[idx] = 1;
        solidCells++;
        frontal[y + ny * z] = 1;
        if (x < lo[0]) lo[0] = x;
        if (y < lo[1]) lo[1] = y;
        if (z < lo[2]) lo[2] = z;
        if (x > hi[0]) hi[0] = x;
        if (y > hi[1]) hi[1] = y;
        if (z > hi[2]) hi[2] = z;
      }
    }
  }
  let frontalCells = 0;
  for (let i = 0; i < frontal.length; i++) frontalCells += frontal[i];

  const extent: [number, number, number] = solidCells
    ? [hi[0] - lo[0] + 1, hi[1] - lo[1] + 1, hi[2] - lo[2] + 1]
    : [0, 0, 0];
  return { nx, ny, nz, flags, solidCells, frontalCells, extent, blockage: frontalCells / (ny * nz), distance };
}

/** Marks the bottom layer of cells as a fixed floor, downstream of the inlet plane. */
export function addFloor(grid: VoxelGrid): void {
  const { nx, ny, nz, flags } = grid;
  for (let z = 1; z < nz - 1; z++) {
    for (let x = 1; x < nx; x++) flags[x + nx * ny * z] = 2;
  }
}
