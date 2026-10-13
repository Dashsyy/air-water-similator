// Validation suite: runs the solver on reference shapes in headless Chrome (WebGPU) and compares
// the blockage-corrected drag coefficient with textbook values.
// Usage: npm run test:browser            (medium grid)
//        npm run test:browser -- 192     (grid length in cells)
import puppeteer from 'puppeteer-core';
import { createServer } from 'vite';

const CHROME = process.env.CHROME_PATH ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const nx = Number(process.argv[2] ?? 128);

// tolerance is the accepted relative error; streamlined shapes are known to be far off on coarse grids.
const CASES = [
  { name: 'Sphere, Re 100', query: 'preset=sphere&re=100&ft=40', expected: 1.09, tolerance: 0.15 },
  { name: 'Sphere, Re 20 000', query: 'preset=sphere&re=20000&ft=60', expected: 0.47, tolerance: 0.25 },
  { name: 'Cube, face on', query: 'preset=cube&re=1000000&ft=40', expected: 1.05, tolerance: 0.25 },
  { name: 'Square plate, face on', query: 'preset=plate&re=1000000&ft=40', expected: 1.17, tolerance: 0.25 },
  { name: 'Car on the floor (stability)', query: 'preset=car&ground=1&re=1000000&ft=20', expected: null, tolerance: null },
  { name: 'Cow on the floor (stability)', query: 'preset=cow&ground=1&re=1000000&ft=20', expected: null, tolerance: null },
];

const server = await createServer({ server: { port: 5198, strictPort: true }, logLevel: 'error' });
await server.listen();
const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: true,
  protocolTimeout: 0,
  args: ['--enable-unsafe-webgpu', '--use-angle=metal', '--no-first-run'],
});

let failed = 0;
try {
  console.log(`Grid ${nx} x ${nx / 2} x ${nx / 2}\n`);
  console.log('case'.padEnd(32), 'Cd'.padStart(7), 'textbook'.padStart(9), 'error'.padStart(8), ' result');
  for (const c of CASES) {
    const page = await browser.newPage();
    await page.goto(`http://localhost:5198/scripts/pages/solver-test.html?nx=${nx}&${c.query}`, { waitUntil: 'load' });
    await page.waitForFunction('window.__result !== undefined || window.__error !== undefined', { timeout: 0 });
    const { result, error } = await page.evaluate(() => ({ result: window.__result, error: window.__error }));
    await page.close();
    if (error || !result) {
      failed++;
      console.log(c.name.padEnd(32), 'page error:', String(error).split('\n')[0]);
      continue;
    }
    const cd = result.cd * (1 - result.blockage) ** 2;
    const stable = !result.blewUp && Number.isFinite(cd) && cd > 0;
    let ok = stable;
    let err = '';
    if (c.expected !== null) {
      const rel = (cd - c.expected) / c.expected;
      err = `${rel >= 0 ? '+' : ''}${(rel * 100).toFixed(0)}%`;
      ok = stable && Math.abs(rel) <= c.tolerance;
    }
    if (!ok) failed++;
    console.log(
      c.name.padEnd(32),
      cd.toFixed(3).padStart(7),
      (c.expected === null ? '-' : c.expected.toFixed(2)).padStart(9),
      err.padStart(8),
      ok ? ' pass' : stable ? ' FAIL (outside tolerance)' : ' FAIL (unstable)',
    );
  }
} finally {
  await browser.close();
  await server.close();
}
console.log(failed ? `\n${failed} case(s) failed.` : '\nAll cases passed.');
process.exit(failed ? 1 : 0);
