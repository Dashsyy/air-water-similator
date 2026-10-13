// Writes one small model in each supported text/binary format, for exercising the importers.
// Usage: node scripts/make-test-models.mjs <outDir>
import { mkdirSync, writeFileSync } from 'node:fs';
import { TorusKnotGeometry } from 'three';

const out = process.argv[2];
mkdirSync(out, { recursive: true });
const g = new TorusKnotGeometry(1, 0.32, 96, 14);
const pos = g.attributes.position.array;
const idx = g.index.array;
const vCount = pos.length / 3;

let obj = '# torus knot\n';
for (let i = 0; i < vCount; i++) obj += `v ${pos[i * 3]} ${pos[i * 3 + 1]} ${pos[i * 3 + 2]}\n`;
for (let i = 0; i < idx.length; i += 3) obj += `f ${idx[i] + 1} ${idx[i + 1] + 1} ${idx[i + 2] + 1}\n`;
writeFileSync(`${out}/knot.obj`, obj);

let stl = 'solid knot\n';
for (let i = 0; i < idx.length; i += 3) {
  stl += 'facet normal 0 0 0\n outer loop\n';
  for (let k = 0; k < 3; k++) { const v = idx[i + k] * 3; stl += `  vertex ${pos[v]} ${pos[v + 1]} ${pos[v + 2]}\n`; }
  stl += ' endloop\nendfacet\n';
}
writeFileSync(`${out}/knot.stl`, stl + 'endsolid knot\n');

let ply = `ply\nformat ascii 1.0\nelement vertex ${vCount}\nproperty float x\nproperty float y\nproperty float z\nelement face ${idx.length / 3}\nproperty list uchar int vertex_indices\nend_header\n`;
for (let i = 0; i < vCount; i++) ply += `${pos[i * 3]} ${pos[i * 3 + 1]} ${pos[i * 3 + 2]}\n`;
for (let i = 0; i < idx.length; i += 3) ply += `3 ${idx[i]} ${idx[i + 1]} ${idx[i + 2]}\n`;
writeFileSync(`${out}/knot.ply`, ply);

// Minimal binary glTF: one mesh, positions and 32-bit indices.
const posBytes = Buffer.from(new Float32Array(pos).buffer);
const idxBytes = Buffer.from(new Uint32Array(idx).buffer);
const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
for (let i = 0; i < pos.length; i++) { min[i % 3] = Math.min(min[i % 3], pos[i]); max[i % 3] = Math.max(max[i % 3], pos[i]); }
const json = {
  asset: { version: '2.0' }, scene: 0, scenes: [{ nodes: [0] }], nodes: [{ mesh: 0 }],
  meshes: [{ primitives: [{ attributes: { POSITION: 0 }, indices: 1, material: 0 }] }],
  materials: [{ pbrMetallicRoughness: { baseColorFactor: [0.2, 0.6, 0.9, 1] } }],
  buffers: [{ byteLength: posBytes.length + idxBytes.length }],
  bufferViews: [
    { buffer: 0, byteOffset: 0, byteLength: posBytes.length, target: 34962 },
    { buffer: 0, byteOffset: posBytes.length, byteLength: idxBytes.length, target: 34963 },
  ],
  accessors: [
    { bufferView: 0, componentType: 5126, count: vCount, type: 'VEC3', min, max },
    { bufferView: 1, componentType: 5125, count: idx.length, type: 'SCALAR' },
  ],
};
let jsonBytes = Buffer.from(JSON.stringify(json));
jsonBytes = Buffer.concat([jsonBytes, Buffer.alloc((4 - (jsonBytes.length % 4)) % 4, 0x20)]);
const bin = Buffer.concat([posBytes, idxBytes]);
const header = Buffer.alloc(12);
header.writeUInt32LE(0x46546c67, 0); header.writeUInt32LE(2, 4); header.writeUInt32LE(12 + 8 + jsonBytes.length + 8 + bin.length, 8);
const chunk = (type, data) => { const h = Buffer.alloc(8); h.writeUInt32LE(data.length, 0); h.writeUInt32LE(type, 4); return Buffer.concat([h, data]); };
writeFileSync(`${out}/knot.glb`, Buffer.concat([header, chunk(0x4e4f534a, jsonBytes), chunk(0x004e4942, bin)]));
writeFileSync(`${out}/broken.obj`, 'this is not a model\n');
writeFileSync(`${out}/notes.txt`, 'hello\n');
console.log('wrote', out);
