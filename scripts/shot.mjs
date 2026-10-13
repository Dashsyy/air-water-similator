// Renders a test page in headless Chrome and saves window.__image to a file.
// Usage: node scripts/shot.mjs "<url>" <output.jpg> [timeoutSeconds]
import { writeFileSync, mkdirSync } from 'node:fs';
import { dirname } from 'node:path';
import puppeteer from 'puppeteer-core';

const [url, output, timeoutArg] = process.argv.slice(2);
const browser = await puppeteer.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: true,
  protocolTimeout: 0,
  args: ['--enable-unsafe-webgpu', '--use-angle=metal', '--no-first-run'],
  defaultViewport: { width: 1280, height: 720 },
});
try {
  const page = await browser.newPage();
  page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warn') console.log(`[${m.type()}]`, m.text().slice(0, 600)); });
  page.on('pageerror', (e) => console.log('[pageerror]', e.message));
  await page.goto(url, { waitUntil: 'load' });
  await page.waitForFunction('window.__result !== undefined || window.__error !== undefined', { timeout: Number(timeoutArg ?? 120) * 1000 });
  const out = await page.evaluate(() => ({ result: window.__result, error: window.__error, image: window.__image }));
  if (out.image) {
    mkdirSync(dirname(output), { recursive: true });
    writeFileSync(output, Buffer.from(out.image.split(',')[1], 'base64'));
  }
  console.log(JSON.stringify({ result: out.result, error: out.error }, null, 1));
} finally {
  await browser.close();
}
