// Quick sanity check of the voxeliser against shapes with known volume and frontal area.
import { PRESETS } from '../src/geometry/presets.ts';
import { placeModel, transformedTriangles } from '../src/geometry/model.ts';
import { voxelize } from '../src/geometry/voxelize.ts';

const [nx, ny, nz] = [128, 64, 64];
for (const preset of PRESETS) {
  const model = preset.build();
  const t0 = performance.now();
  const place = placeModel(model, { yaw: 0, pitch: preset.pitch ?? 0, roll: 0, onGround: model.grounded }, nx, ny, nz);
  const grid = voxelize(transformedTriangles(model, place.matrix), nx, ny, nz);
  const ms = (performance.now() - t0).toFixed(1);
  let note = '';
  if (preset.id === 'sphere') {
    const r = place.scale / 2;
    note = ` expected volume ${((4 / 3) * Math.PI * r ** 3).toFixed(0)}, frontal ${(Math.PI * r * r).toFixed(0)}`;
  }
  if (preset.id === 'cube') note = ` expected volume ${(place.scale ** 3).toFixed(0)}, frontal ${(place.scale ** 2).toFixed(0)}`;
  console.log(
    `${preset.id.padEnd(9)} tris=${String(model.triangleCount).padStart(6)} scale=${place.scale.toFixed(1)} ` +
      `solid=${grid.solidCells} frontal=${grid.frontalCells} extent=${grid.extent} blockage=${(grid.blockage * 100).toFixed(1)}% ${ms}ms${note}`,
  );
}
