// Screenshots the 2D lab after it has run for a while. Usage: node scripts/lab-shot.mjs <out.png> [seconds]
import puppeteer from 'puppeteer-core';
const [out, secondsArg] = process.argv.slice(2);
const browser = await puppeteer.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: true, protocolTimeout: 0,
  args: ['--enable-unsafe-webgpu', '--use-angle=metal', '--no-first-run'],
  defaultViewport: { width: 1280, height: 900 },
});
try {
  const page = await browser.newPage();
  page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warn') console.log(`[${m.type()}]`, m.text().slice(0, 500)); });
  page.on('pageerror', (e) => console.log('[pageerror]', e.message));
  await page.goto((process.env.APP_URL ?? 'http://localhost:5199/') + 'lab2d.html', { waitUntil: 'load' });
  await new Promise((r) => setTimeout(r, Number(secondsArg ?? 30) * 1000));
  await page.screenshot({ path: out });
  console.log(JSON.stringify(await page.evaluate(() => window.__lab)));
} finally { await browser.close(); }
