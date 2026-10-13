// Drives the real app in headless Chrome and saves screenshots.
// Usage: node scripts/app-shot.mjs <outDir> [seconds] [jsToRunBeforeShot]
import { mkdirSync } from 'node:fs';
import puppeteer from 'puppeteer-core';

const [outDir, secondsArg, script] = process.argv.slice(2);
const base = process.env.APP_URL ?? 'http://localhost:5199/';
mkdirSync(outDir, { recursive: true });
const browser = await puppeteer.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: true,
  protocolTimeout: 0,
  args: ['--enable-unsafe-webgpu', '--use-angle=metal', '--no-first-run'],
  defaultViewport: { width: Number(process.env.W ?? 1440), height: Number(process.env.H ?? 900) },
});
try {
  const page = await browser.newPage();
  page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warn') console.log(`[${m.type()}]`, m.text().slice(0, 500)); });
  page.on('pageerror', (e) => console.log('[pageerror]', e.message));
  await page.goto(base, { waitUntil: 'load' });
  await new Promise((r) => setTimeout(r, Number(secondsArg ?? 8) * 1000));
  if (script) {
    const result = await page.evaluate(script);
    if (result !== undefined) console.log('script result:', JSON.stringify(result));
    await new Promise((r) => setTimeout(r, 1500));
  }
  await page.screenshot({ path: `${outDir}/app.png` });
  const hud = await page.evaluate(() => ({
    cd: document.getElementById('hud-cd')?.textContent,
    status: document.getElementById('hud-status')?.textContent,
    sim: document.getElementById('sim-info')?.textContent,
    model: document.getElementById('model-info')?.textContent,
  }));
  console.log(JSON.stringify(hud));
} finally {
  await browser.close();
}
