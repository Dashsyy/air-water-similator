import type { Conditions, Snapshot } from '../app/simulation';
import type { Orientation } from '../geometry/model';
import { SETTLE_FLOW_THROUGHS, WARMUP_FLOW_THROUGHS, type CoefficientPoint } from '../physics/aero';
import { REFERENCE_SHAPES, closestReference, gradeFor, type Grade } from '../physics/references';

export interface ReportImage {
  caption: string;
  src: string;
}

export interface ReportInput {
  modelName: string;
  imported: boolean;
  triangleCount: number;
  snapshot: Snapshot;
  conditions: Conditions;
  orientation: Orientation;
  history: CoefficientPoint[];
  images: ReportImage[];
  expected?: { cd: number; note: string };
  createdAt: Date;
}

export interface BuiltReport {
  html: string;
  grade: Grade;
  title: string;
}

const esc = (s: string): string =>
  s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c] as string);

/** Formats with a sensible number of significant digits and thousands separators. */
function num(value: number, digits = 3): string {
  if (!Number.isFinite(value)) return 'n/a';
  const abs = Math.abs(value);
  if (abs >= 1000) return Math.round(value).toLocaleString('en-US');
  if (abs === 0) return '0';
  const decimals = Math.max(0, digits - 1 - Math.floor(Math.log10(abs)));
  return value.toFixed(Math.min(decimals, 4));
}

const kmh = (metresPerSecond: number): number => metresPerSecond * 3.6;

function powerPhrase(watts: number): string {
  const w = Math.abs(watts);
  if (w < 1) return `${num(w * 1000)} milliwatts, less than a small LED uses`;
  if (w < 120) return `${num(w)} W, about what ${num(w / 10, 2)} LED bulbs draw`;
  if (w < 2000) return `${num(w)} W, the output of ${num(w / 250, 2)} hard-pedalling cyclists`;
  return `${num(w / 1000)} kW, about ${num(w / 745.7, 2)} horsepower or ${num(w / 250, 2)} hard-pedalling cyclists`;
}

function weightPhrase(newtons: number): string {
  const kg = Math.abs(newtons) / 9.81;
  const things: [number, string, string][] = [
    [1200, 'small car', 'small cars'],
    [400, 'grand piano', 'grand pianos'],
    [70, 'adult', 'adults'],
    [4.5, 'house cat', 'house cats'],
    [0.1, 'apple', 'apples'],
    [0.003, 'sheet of paper', 'sheets of paper'],
  ];
  for (const [mass, one, many] of things) {
    const count = kg / mass;
    if (count >= 1) return `the weight of ${num(count, 2)} ${count < 1.5 ? one : many}`;
  }
  return 'less than the weight of a sheet of paper';
}

function funFacts(input: ReportInput): string[] {
  const { measurement: m, realWorld: rw } = input.snapshot;
  const wind = input.conditions.windKmh;
  const facts: string[] = [];
  const ref = closestReference(m.cd);
  facts.push(`Its drag coefficient sits closest to a <b>${esc(ref.name.toLowerCase())}</b> ${ref.emoji} (about ${ref.cd}).`);
  facts.push(`At ${num(wind)} km/h the wind pushes it back with <b>${num(rw.dragN)} N</b>, ${weightPhrase(rw.dragN)}.`);
  facts.push(`Holding it steady in that wind takes <b>${powerPhrase(rw.powerW)}</b>.`);
  if (rw.terminalSpeed !== null) {
    const v = kmh(rw.terminalSpeed);
    const versus = v > 220 ? 'faster than' : v < 180 ? 'slower than' : 'about the same as';
    facts.push(
      `Dropped facing the same way, it would stop speeding up near <b>${num(v)} km/h</b>, ${versus} a belly-down skydiver at roughly 200 km/h.`,
    );
  }
  if (rw.takeoffSpeed !== null) {
    const v = kmh(rw.takeoffSpeed);
    let verdict: string;
    if (v > 1235) verdict = 'That is faster than sound, so it is staying on the ground.';
    else if (v > 252) verdict = 'That is beyond the strongest hurricanes, so it is safe outdoors.';
    else if (v > 120) verdict = 'Only hurricane-force wind could manage that.';
    else verdict = 'A strong storm could do it. Tie it down.';
    facts.push(`Lift would match its weight at <b>${num(v)} km/h</b>. ${verdict}`);
  } else {
    facts.push('It makes no upward lift in this position, so no wind will make it fly. It gets pressed down instead.');
  }
  if (m.cd > 0 && m.cdSpread / m.cd > 0.08) {
    facts.push(
      `Its drag wobbles by about ±${num((100 * m.cdSpread) / m.cd, 2)}%. That is the wake shedding swirls of air, the same effect that makes flags flap.`,
    );
  }
  return facts;
}

/** Bucket-averages the history so the chart stays light however long the run was. */
function downsample(points: CoefficientPoint[], target: number): { t: number; cd: number }[] {
  if (points.length <= target) return points.map((p) => ({ t: p.t, cd: p.cd }));
  const out: { t: number; cd: number }[] = [];
  const size = points.length / target;
  for (let i = 0; i < target; i++) {
    const from = Math.floor(i * size);
    const to = Math.max(from + 1, Math.floor((i + 1) * size));
    let t = 0, cd = 0;
    for (let j = from; j < to; j++) {
      t += points[j].t;
      cd += points[j].cd;
    }
    out.push({ t: t / (to - from), cd: cd / (to - from) });
  }
  return out;
}

function historyChart(input: ReportInput): string {
  const pts = downsample(input.history, 240);
  if (pts.length < 2) return '<p class="muted">Not enough data yet for a history chart.</p>';
  const mean = input.snapshot.measurement.cd;
  const W = 680, H = 240, L = 46, R = 76, T = 14, B = 34;
  const tMax = Math.max(pts[pts.length - 1].t, 1);
  const settledPts = pts.filter((p) => p.t >= WARMUP_FLOW_THROUGHS);
  const basis = (settledPts.length > 4 ? settledPts : pts).map((p) => p.cd).sort((a, b) => a - b);
  const lo = Math.min(0, basis[Math.floor(basis.length * 0.02)]);
  const hi = Math.max(basis[Math.ceil(basis.length * 0.98) - 1] * 1.25, mean * 1.4, 0.1);
  const x = (t: number): number => L + (t / tMax) * (W - L - R);
  const y = (v: number): number => T + (1 - (Math.min(Math.max(v, lo), hi) - lo) / (hi - lo)) * (H - T - B);

  const path = pts.map((p, i) => `${i ? 'L' : 'M'}${x(p.t).toFixed(1)},${y(p.cd).toFixed(1)}`).join('');
  const ticks = 4;
  let grid = '';
  for (let i = 0; i <= ticks; i++) {
    const v = lo + ((hi - lo) * i) / ticks;
    grid += `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" class="grid"/>`;
    grid += `<text x="${L - 8}" y="${y(v) + 4}" text-anchor="end" class="tick">${num(v, 2)}</text>`;
  }
  const xStep = tMax > 40 ? 10 : tMax > 16 ? 5 : 2;
  for (let t = 0; t <= tMax; t += xStep) {
    grid += `<text x="${x(t)}" y="${H - B + 16}" text-anchor="middle" class="tick">${t}</text>`;
  }
  const warmEnd = Math.min(WARMUP_FLOW_THROUGHS, tMax);
  const warm =
    `<rect x="${x(0)}" y="${T}" width="${x(warmEnd) - x(0)}" height="${H - T - B}" class="warm"/>` +
    `<text x="${x(0) + 6}" y="${T + 14}" class="tick">warm-up, not averaged</text>`;
  const meanLine = input.snapshot.measurement.averagedTime > 0
    ? `<line x1="${x(warmEnd)}" x2="${W - R}" y1="${y(mean)}" y2="${y(mean)}" class="mean"/>` +
      `<text x="${W - R + 6}" y="${y(mean) + 4}" class="label">mean ${num(mean)}</text>`
    : '';
  const data = JSON.stringify(pts.map((p) => [Number(p.t.toFixed(2)), Number(p.cd.toFixed(4))]));

  return `
<figure class="chart" id="history" data-points='${data}' data-x0="${L}" data-x1="${W - R}" data-tmax="${tMax}">
  <figcaption>Drag coefficient while the flow developed</figcaption>
  <div class="plot">
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Line chart of drag coefficient against time in flow-throughs">
      ${warm}${grid}
      <line x1="${L}" x2="${W - R}" y1="${H - B}" y2="${H - B}" class="axis"/>
      ${meanLine}
      <path d="${path}" class="line"/>
      <line id="cross" y1="${T}" y2="${H - B}" class="cross" visibility="hidden"/>
      <circle id="dot" r="4" class="dot" visibility="hidden"/>
      <text x="${(L + W - R) / 2}" y="${H - 4}" text-anchor="middle" class="tick">time, in passes of air over the object's length</text>
    </svg>
    <div class="tip" id="tip" hidden></div>
  </div>
</figure>`;
}

function comparisonChart(cd: number, name: string): string {
  const rows = [...REFERENCE_SHAPES.map((r) => ({ label: `${r.emoji} ${r.name}`, cd: r.cd, mine: false }))];
  rows.push({ label: `➤ ${name}`, cd, mine: true });
  rows.sort((a, b) => a.cd - b.cd);
  const max = Math.max(1.5, ...rows.map((r) => r.cd)) * 1.05;
  const body = rows
    .map((r) => {
      const width = Math.max(0.6, (Math.max(r.cd, 0) / max) * 100);
      return `<div class="bar-row${r.mine ? ' mine' : ''}" title="${esc(r.label)}: ${num(r.cd)}">
  <span class="bar-label">${esc(r.label)}</span>
  <span class="bar-track"><span class="bar" style="width:${width.toFixed(1)}%"></span><span class="bar-value">${num(r.cd)}</span></span>
</div>`;
    })
    .join('\n');
  return `<figure class="chart"><figcaption>Drag coefficient next to familiar shapes (lower is more streamlined)</figcaption>
<div class="bars">${body}</div>
<p class="muted small">Reference values are commonly quoted approximations, not measurements from this tunnel.</p></figure>`;
}

const STYLE = `
:root{--surface:#fcfcfb;--page:#f3f2ee;--ink:#0b0b0b;--ink2:#52514e;--muted:#7b7a75;--rule:#e2e0da;--blue:#2a78d6;--accent:#eb6834;--neutral:#b9b7ae}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
main{max-width:860px;margin:0 auto;padding:28px 24px 48px;background:var(--surface)}
h1{font-size:28px;line-height:1.2;margin:0 0 4px}
h2{font-size:19px;margin:34px 0 12px;padding-top:18px;border-top:1px solid var(--rule)}
p{margin:0 0 10px}
.muted{color:var(--muted)}.small{font-size:12.5px}
.head{display:flex;gap:18px;align-items:center;justify-content:space-between}
.grade{flex:none;width:92px;height:92px;border-radius:50%;border:3px solid var(--ink);display:grid;place-items:center;text-align:center}
.grade b{font-size:40px;line-height:1}.grade span{font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink2);margin-top:-14px}
.notice{margin:16px 0;padding:10px 14px;border-left:4px solid #fab219;background:#fff7df;font-size:14px}
.shots{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px;margin:18px 0}
.shots figure{margin:0}.shots img{width:100%;display:block;border-radius:6px;background:#090b12}
.shots figcaption{font-size:12.5px;color:var(--muted);margin-top:4px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:14px 0}
.tile{border:1px solid var(--rule);border-radius:8px;padding:12px 14px}
.tile .k{font-size:12px;color:var(--ink2)}.tile .v{font-size:27px;font-weight:650;line-height:1.2}.tile .u{font-size:12px;color:var(--muted)}
table{width:100%;border-collapse:collapse;font-size:14px}
td,th{padding:7px 8px;border-bottom:1px solid var(--rule);text-align:left;vertical-align:top}
td:nth-child(2){text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
td:nth-child(3){color:var(--muted);font-size:12.5px}
.chart{margin:18px 0}.chart figcaption{font-weight:600;margin-bottom:8px}
.plot{position:relative}.plot svg{width:100%;height:auto;display:block}
.grid{stroke:var(--rule);stroke-width:1}.axis{stroke:var(--neutral);stroke-width:1}
.tick{font-size:11px;fill:var(--muted)}.label{font-size:12px;fill:var(--ink2)}
.warm{fill:#efeee9}.line{fill:none;stroke:var(--blue);stroke-width:2;stroke-linejoin:round}
.mean{stroke:var(--ink2);stroke-width:1;stroke-dasharray:4 4}
.cross{stroke:var(--neutral);stroke-width:1}.dot{fill:var(--blue);stroke:var(--surface);stroke-width:2}
.tip{position:absolute;pointer-events:none;background:var(--ink);color:#fff;font-size:12px;padding:5px 8px;border-radius:5px;white-space:nowrap;transform:translate(-50%,-130%)}
.bars{display:grid;gap:6px}
.bar-row{display:grid;grid-template-columns:200px 1fr;gap:10px;align-items:center;font-size:13.5px}
.bar-label{color:var(--ink2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.bar-track{display:flex;align-items:center;gap:8px}
.bar{height:12px;border-radius:0 4px 4px 0;background:var(--neutral)}
.bar-value{font-variant-numeric:tabular-nums;color:var(--ink2);font-size:12.5px}
.bar-row.mine .bar{background:var(--accent)}.bar-row.mine .bar-label,.bar-row.mine .bar-value{color:var(--ink);font-weight:650}
.fun{list-style:none;padding:0;margin:0;display:grid;gap:8px}
.fun li{padding:10px 14px;border:1px solid var(--rule);border-radius:8px}
.verdict{font-size:17px}
footer{margin-top:30px;font-size:12.5px;color:var(--muted)}
@media (max-width:560px){.bar-row{grid-template-columns:120px 1fr}.head{align-items:flex-start}}
@media print{body{background:#fff}main{padding:0;max-width:none}h2{break-after:avoid}figure,.tile,.fun li,tr{break-inside:avoid}.tip{display:none}}
`;

const HOVER_SCRIPT = `
(function(){var f=document.getElementById('history');if(!f)return;var pts=JSON.parse(f.dataset.points);
var svg=f.querySelector('svg'),cross=document.getElementById('cross'),dot=document.getElementById('dot'),tip=document.getElementById('tip');
var path=f.querySelector('.line'),x0=+f.dataset.x0,x1=+f.dataset.x1,tmax=+f.dataset.tmax,vb=svg.viewBox.baseVal;
function hide(){cross.setAttribute('visibility','hidden');dot.setAttribute('visibility','hidden');tip.hidden=true}
svg.addEventListener('pointerleave',hide);
svg.addEventListener('pointermove',function(e){var r=svg.getBoundingClientRect(),sx=(e.clientX-r.left)/r.width*vb.width;
if(sx<x0||sx>x1){hide();return}var t=(sx-x0)/(x1-x0)*tmax,best=0;
for(var i=1;i<pts.length;i++){if(Math.abs(pts[i][0]-t)<Math.abs(pts[best][0]-t))best=i}
var px=x0+pts[best][0]/tmax*(x1-x0),len=path.getTotalLength(),lo=0,hi=len,p;
for(var k=0;k<24;k++){var mid=(lo+hi)/2;p=path.getPointAtLength(mid);if(p.x<px)lo=mid;else hi=mid}
cross.setAttribute('x1',px);cross.setAttribute('x2',px);cross.setAttribute('visibility','visible');
dot.setAttribute('cx',px);dot.setAttribute('cy',p.y);dot.setAttribute('visibility','visible');
tip.hidden=false;tip.style.left=(px/vb.width*r.width)+'px';tip.style.top=(p.y/vb.height*r.height)+'px';
tip.textContent='Cd '+pts[best][1].toFixed(3)+' at '+pts[best][0].toFixed(1)+' passes'})})();
`;

export function buildReport(input: ReportInput): BuiltReport {
  const { measurement: m, realWorld: rw, flow, setup, grid, tier } = input.snapshot;
  const grade = gradeFor(m.cd);
  const name = input.modelName || 'Mystery object';
  const title = `Science & Fun Report: ${name}`;
  const wind = input.conditions.windKmh;
  const o = input.orientation;
  const row = (k: string, v: string, note = ''): string => `<tr><td>${k}</td><td>${v}</td><td>${note}</td></tr>`;

  const settleNote = m.settled
    ? ''
    : `<div class="notice"><b>Still settling.</b> This report was made ${Math.round(m.progress * 100)}% of the way through the measurement. ` +
      `The tunnel discards the first ${WARMUP_FLOW_THROUGHS} passes of air and wants ${SETTLE_FLOW_THROUGHS} more before it trusts the average. Let it run longer for steadier numbers.</div>`;

  let validation = '';
  if (input.expected) {
    const diff = ((m.cd - input.expected.cd) / input.expected.cd) * 100;
    validation = `<p><b>Check against the textbook.</b> The accepted value for a ${esc(input.expected.note)} is about ${input.expected.cd}. ` +
      `This run gave ${num(m.cd)}, which is ${diff >= 0 ? '+' : ''}${num(diff, 2)}% away.</p>`;
  }

  const reynoldsNote = flow.simulatedReynolds < flow.realReynolds * 0.98
    ? `The real flow has a Reynolds number near ${num(flow.realReynolds, 2)}. The grid can only resolve about ${num(flow.simulatedReynolds, 2)}, so the simulated air is effectively stickier than real air.`
    : `The simulation ran at the real Reynolds number of about ${num(flow.realReynolds, 2)}.`;

  const html = `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>${esc(title)}</title><style>${STYLE}</style></head><body><main>
<div class="head">
  <div>
    <p class="muted small">Virtual wind tunnel · ${esc(input.createdAt.toLocaleString('en-GB', { dateStyle: 'long', timeStyle: 'short' }))}</p>
    <h1>${esc(title)}</h1>
    <p class="verdict"><b>${esc(grade.title)}.</b> ${esc(grade.blurb)}</p>
  </div>
  <div class="grade"><b>${grade.letter}</b><span>aero grade</span></div>
</div>
${settleNote}
<div class="shots">${input.images
    .map((i) => `<figure><img src="${i.src}" alt="${esc(i.caption)}"><figcaption>${esc(i.caption)}</figcaption></figure>`)
    .join('')}</div>

<div class="tiles">
  <div class="tile"><div class="k">Drag coefficient</div><div class="v">${num(m.cd)}</div><div class="u">Cd, estimated</div></div>
  <div class="tile"><div class="k">Lift coefficient</div><div class="v">${num(m.cl)}</div><div class="u">Cl, positive is up</div></div>
  <div class="tile"><div class="k">Lift-to-drag ratio</div><div class="v">${num(m.liftToDrag)}</div><div class="u">aerodynamic efficiency</div></div>
  <div class="tile"><div class="k">Drag at ${num(wind)} km/h</div><div class="v">${num(rw.dragN)}</div><div class="u">newtons</div></div>
</div>

<h2>The fun part</h2>
<ul class="fun">${funFacts(input).map((f) => `<li>${f}</li>`).join('')}</ul>
${comparisonChart(m.cd, name)}

<h2>The science part</h2>
<p>Air pushes on anything it flows around. The <b>drag coefficient</b> says how hard, after removing the effect of size and speed, so a small toy and a full-size car can be compared fairly. <b>Lift</b> is the part of the push that acts upward, and the <b>lift-to-drag ratio</b> is how much lift a shape earns for each unit of drag it pays.</p>
${validation}
<table>
<tr><th>Quantity</th><th>Value</th><th>Notes</th></tr>
${row('Drag coefficient, Cd', num(m.cd), `fluctuates by ±${num(m.cdSpread, 2)}`)}
${row('Lift coefficient, Cl', num(m.cl), 'based on frontal area')}
${row('Side-force coefficient', num(m.cs))}
${row('Lift-to-drag ratio', num(m.liftToDrag))}
${row('Frontal area', `${num(rw.frontalAreaM2)} m²`, 'silhouette seen by the wind')}
${row('Length along the wind', `${num(rw.lengthM)} m`)}
${row('Wind speed', `${num(wind)} km/h`, `${num(wind / 3.6)} m/s`)}
${row('Drag force', `${num(rw.dragN)} N`)}
${row('Lift force', `${num(rw.liftN)} N`, `weight is ${num(rw.weightN)} N`)}
${row('Power to overcome drag', `${num(rw.powerW)} W`)}
${row('Terminal velocity', rw.terminalSpeed === null ? 'n/a' : `${num(kmh(rw.terminalSpeed))} km/h`, `for a mass of ${num(input.conditions.massKg)} kg`)}
${row('Reynolds number, real', num(flow.realReynolds, 2))}
${row('Reynolds number, simulated', num(flow.simulatedReynolds, 2))}
${row('Orientation', `yaw ${num(o.yaw)}°, pitch ${num(o.pitch)}°, roll ${num(o.roll)}°`, o.onGround ? 'resting on a fixed floor' : 'floating in mid-stream')}
${row('Grid', `${grid.nx} × ${grid.ny} × ${grid.nz}`, `${esc(tier.label)} quality, ${num(grid.solidCells)} solid cells`)}
${row('Tunnel blockage', `${num(grid.blockage * 100, 2)}%`, 'corrected for in the coefficients')}
${row('Averaging time', `${num(m.averagedTime)} passes`, `after ${WARMUP_FLOW_THROUGHS} warm-up passes`)}
${row('Source model', `${num(input.triangleCount)} triangles`, input.imported ? 'imported file' : 'built-in shape')}
</table>
${historyChart(input)}

<h2>How much to trust this</h2>
<p>These are <b>estimates from a fast, coarse simulation</b>, good for comparing shapes and seeing why they behave as they do. They are not engineering data.</p>
<p>${reynoldsNote} The object is built from small cubes about ${num(setup.metresPerCell * 100, 2)} cm across, so fine details and the thin layer of air hugging the surface are not resolved.</p>
<p>Blunt shapes such as boxes, chairs and animals come out reasonably close. Streamlined shapes such as cars, wings and teardrops usually come out with too much drag, sometimes two or three times too much, because their real performance depends on that thin surface layer staying attached.</p>
<p class="small muted">Method: Lattice Boltzmann solver (D3Q19, Smagorinsky sub-grid model) on the GPU. Forces from momentum exchange on the surface, averaged over time. Coefficients use the frontal area and are scaled by (1 − blockage)² to offset the tunnel walls. Air density 1.225 kg/m³.</p>
<footer>Generated by the Virtual Wind Tunnel.</footer>
</main><script>${HOVER_SCRIPT}</script></body></html>`;
  return { html, grade, title };
}
