// Opens a page in headless Chrome with WebGPU and prints window.__result (or window.__error).
// Usage: node scripts/run-page.mjs "<url>" [timeoutSeconds]
import puppeteer from 'puppeteer-core';

const url = process.argv[2];
const timeout = Number(process.argv[3] ?? 180) * 1000;
const browser = await puppeteer.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: true,
  protocolTimeout: 0,
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=metal', '--no-first-run'],
});
try {
  const page = await browser.newPage();
  page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warn') console.log(`[${m.type()}]`, m.text()); });
  page.on('pageerror', (e) => console.log('[pageerror]', e.message));
  await page.goto(url, { waitUntil: 'load' });
  await page.waitForFunction('window.__result !== undefined || window.__error !== undefined', { timeout });
  const out = await page.evaluate(() => ({ result: window.__result, error: window.__error }));
  console.log(JSON.stringify(out, null, 1));
} finally {
  await browser.close();
}
