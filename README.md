# Virtual Wind Tunnel

An interactive 3D wind tunnel that runs in the browser. Drop in any 3D model, watch the air flow
around it, and generate a "Science & Fun" report with an estimated drag coefficient, lift, and
real-world comparisons. Everything runs on the user's GPU through WebGPU. There is no server.

## Run it

```bash
npm install
npm run dev          # open the printed URL in Chrome, Edge, or Safari
npm run build        # type-check and build to dist/
npm run test:browser # solver validation in headless Chrome (see below)
```

Requires a browser with WebGPU. `npm run test:browser` expects Chrome at the default macOS path;
set `CHROME_PATH` to override.

## What is in the app

- **Objects:** nine built-in shapes (including NASA and Boeing's truss-braced wing jet), plus import of GLB, glTF, OBJ, STL, PLY, and FBX by file
  picker or drag and drop.
- **Orientation:** yaw, pitch, and roll sliders. The flow adapts live. Grounded objects sit on a
  fixed floor.
- **Views:** smoke lines from a movable rake, a pressure map on the surface, volumetric smoke, a
  slice plane (speed, pressure, or vorticity), and vortex cores.
- **Live read-out:** drag coefficient, lift coefficient, lift-to-drag ratio, and a progress bar
  that shows when the average has settled.
- **Report:** printable page with screenshots, headline numbers, fun comparisons, a full table, a
  convergence chart, and plain-language accuracy notes. Save it to the in-browser gallery,
  download it as HTML, or print it to PDF.
- **Challenges:** nine goals such as "get below a sphere" that unlock from settled measurements.
- **2D lab** (`/lab2d.html`): vortex shedding behind a cylinder with a measured Strouhal number,
  and a brush to draw your own obstacles.

## How it works

| Stage | File | Notes |
|---|---|---|
| Import | `src/geometry/loaders.ts`, `model.ts` | Three.js loaders, flattened to a triangle soup, centred and scaled |
| Voxelise | `src/geometry/voxelize.ts` | Signed ray-crossing counts on three axes with a two-of-three vote; also outputs sub-cell wall distances |
| Solve | `src/gpu/lbm3d.ts` | D3Q19 Lattice Boltzmann in a WebGPU compute shader |
| Derived fields | `src/gpu/fields.ts` | Streakline particles, smoke advection, Q-criterion |
| Render | `src/gpu/renderer.ts`, `shaders.ts` | Raw WebGPU: mesh, lines, slice, and a ray-marched volume pass |
| Measure | `src/physics/aero.ts` | Coefficients, averaging, real-world conversion |
| Report | `src/report/report.ts` | Self-contained HTML with inline SVG charts |
| Orchestration | `src/app/simulation.ts`, `src/main.ts` | Simulation controller and UI wiring |

Solver details that matter for accuracy:

- **Collision:** BGK with a Smagorinsky sub-grid model. The eddy viscosity is damped in cells that
  touch the model, because plain Smagorinsky gives far too much skin friction there.
- **Walls:** interpolated (Bouzidi) bounce-back on the model, using the voxeliser's wall distances.
  Halfway bounce-back on the tunnel floor.
- **Boundaries:** fixed free-stream velocity on the inlet and sides, constant pressure at the outlet.
- **Start-up:** the wind is eased in from rest. An impulsive start fills the tunnel with ripples.
- **Forces:** momentum exchange, accumulated on the GPU in fixed point and read back asynchronously.
- **Blockage:** coefficients are scaled by `(1 - blockage)^2`. This was checked against a sphere at
  two blockage ratios.
- **Reynolds number:** the lattice viscosity has a floor (`MIN_LATTICE_VISCOSITY`), so the simulated
  Reynolds number is capped near 10 000. The report shows both the real and simulated values.

## Validation

`npm run test:browser` runs these cases and fails if a result leaves its tolerance.

| Case | Textbook | 128 grid | 192 grid |
|---|---|---|---|
| Sphere, Re 100 | 1.09 | 1.12 | 1.12 |
| Sphere, Re 20 000 | 0.47 | 0.45 | 0.37 |
| Cube, face on | 1.05 | 1.16 | 1.14 |
| Square plate, face on | 1.17 | 1.01 | 1.16 |
| 2D cylinder Strouhal number, Re 120 | 0.175 | 0.178 | |

Known weak spots, stated in the report too:

- **Streamlined shapes read far too high.** The built-in teardrop gives about 0.41 against a
  textbook 0.05, and the car gives about 0.64 against roughly 0.3 for a real saloon. The grid
  cannot resolve the thin boundary layer that keeps real flow attached.
- **The sphere result depends on the grid** because its separation point is sensitive to
  resolution. Treat agreement at one grid as partly fortunate.
- **FBX import is untested.** The other five formats were exercised with generated files.

## Not built

The original plan listed an optional Laravel back end for accounts and a shared gallery. It is not
here. Saved reports live in the browser's IndexedDB instead. The report is a single HTML string, so
a back end only needs to store that string and a thumbnail.

## Development scripts

`scripts/` holds the headless-Chrome tooling used during development: `browser-test.mjs`
(validation), `run-page.mjs` and `shot.mjs` (run a test page), `app-shot.mjs`, `report-shot.mjs`,
`lab-shot.mjs`, and `import-test.mjs` (drive the real app), `make-test-models.mjs` (sample files),
and `voxel-check.ts` (voxeliser sanity check, run with `node`).
