// Opens the app, waits, generates the report, and saves the report HTML plus a full-page screenshot.
// Usage: node scripts/report-shot.mjs <outDir> [seconds] [presetIndex]
import { mkdirSync, writeFileSync } from 'node:fs';
import puppeteer from 'puppeteer-core';

const [outDir, secondsArg, presetArg] = process.argv.slice(2);
mkdirSync(outDir, { recursive: true });
const browser = await puppeteer.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: true,
  protocolTimeout: 0,
  args: ['--enable-unsafe-webgpu', '--use-angle=metal', '--no-first-run'],
  defaultViewport: { width: 1440, height: 900 },
});
try {
  const page = await browser.newPage();
  page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warn') console.log(`[${m.type()}]`, m.text().slice(0, 500)); });
  page.on('pageerror', (e) => console.log('[pageerror]', e.message));
  await page.goto(process.env.APP_URL ?? 'http://localhost:5199/', { waitUntil: 'load' });
  await new Promise((r) => setTimeout(r, 1500));
  if (presetArg) await page.evaluate((i) => document.querySelectorAll('.preset')[i].click(), Number(presetArg));
  await new Promise((r) => setTimeout(r, Number(secondsArg ?? 14) * 1000));
  await page.screenshot({ path: `${outDir}/app.png` });
  await page.evaluate(() => document.getElementById('make-report').click());
  await new Promise((r) => setTimeout(r, 800));
  const html = await page.evaluate(() => document.getElementById('report-frame').srcdoc);
  writeFileSync(`${outDir}/report.html`, html);
  const report = await browser.newPage();
  await report.setViewport({ width: 900, height: 1000 });
  await report.setContent(html, { waitUntil: 'load' });
  await report.screenshot({ path: `${outDir}/report.png`, fullPage: true });
  console.log('report bytes:', html.length, '| hud:', await page.evaluate(() => document.getElementById('hud').innerText.replace(/\n/g, ' ')));
} finally {
  await browser.close();
}
