// Uploads each test model through the app's file picker and reports what the app did with it.
// Usage: node scripts/import-test.mjs <modelsDir> <shotDir>
import { mkdirSync } from 'node:fs';
import puppeteer from 'puppeteer-core';
const [dir, shots] = process.argv.slice(2);
mkdirSync(shots, { recursive: true });
const browser = await puppeteer.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: true, protocolTimeout: 0,
  args: ['--enable-unsafe-webgpu', '--use-angle=metal', '--no-first-run'],
  defaultViewport: { width: 1440, height: 900 },
});
try {
  const page = await browser.newPage();
  page.on('pageerror', (e) => console.log('[pageerror]', e.message));
  await page.goto(process.env.APP_URL ?? 'http://localhost:5199/', { waitUntil: 'load' });
  await new Promise((r) => setTimeout(r, 2000));
  for (const name of ['knot.obj', 'knot.stl', 'knot.ply', 'knot.glb', 'broken.obj', 'notes.txt']) {
    const input = await page.$('#file');
    await input.uploadFile(`${dir}/${name}`);
    await new Promise((r) => setTimeout(r, 4000));
    const state = await page.evaluate(() => ({
      model: document.getElementById('model-info').textContent,
      toasts: [...document.querySelectorAll('.toast')].map((t) => `${t.className.replace('toast ', '')}: ${t.textContent}`),
      cd: document.getElementById('hud-cd').textContent,
    }));
    console.log(name.padEnd(11), JSON.stringify(state));
    if (name === 'knot.glb') await page.screenshot({ path: `${shots}/import.png` });
  }
} finally { await browser.close(); }
