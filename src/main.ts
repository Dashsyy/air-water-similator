import { initGpu } from './gpu/device';
import { Simulation, type QualityTier, type RakeLayout } from './app/simulation';
import { CHALLENGES, evaluate, loadCompleted, saveCompleted } from './app/challenges';
import { PRESETS, type Preset } from './geometry/presets';
import { loadModelFile, SUPPORTED_EXTENSIONS } from './geometry/loaders';
import type { Model } from './geometry/model';
import type { SliceAxis, SliceField } from './gpu/renderer';
import { buildReport, type BuiltReport, type ReportImage } from './report/report';
import { deleteReport, listReports, saveReport } from './report/gallery';

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Missing element #${id}`);
  return el as T;
};

function toast(message: string, kind: 'info' | 'error' | 'win' = 'info'): void {
  const el = document.createElement('div');
  el.className = `toast ${kind}`;
  el.textContent = message;
  $('toasts').append(el);
  setTimeout(() => el.remove(), kind === 'error' ? 9000 : 5000);
}

function fatal(title: string, detail: string): void {
  const box = $('fatal');
  box.hidden = false;
  box.replaceChildren();
  const inner = document.createElement('div');
  const h = document.createElement('h2');
  h.textContent = title;
  const p = document.createElement('p');
  p.textContent = detail;
  inner.append(h, p);
  box.append(inner);
}

const fmt = (v: number, digits = 2): string => (Number.isFinite(v) ? v.toFixed(digits) : '–');

/** CSS gradient matching the shader's turbo colour map. */
function turboGradient(): string {
  const stops: string[] = [];
  for (let i = 0; i <= 10; i++) {
    const x = i / 10;
    const v = [1, x, x * x, x ** 3, x ** 4, x ** 5];
    const dot = (c: number[]): number => Math.round(255 * Math.min(1, Math.max(0, c.reduce((s, k, j) => s + k * v[j], 0))));
    const r = dot([0.13572138, 4.6153926, -42.66032258, 132.13108234, -152.94239396, 59.28637943]);
    const g = dot([0.09140261, 2.19418839, 4.84296658, -14.18503333, 4.27729857, 2.82956604]);
    const b = dot([0.1066733, 12.64194608, -60.58204836, 110.36276771, -89.90310912, 27.34824973]);
    stops.push(`rgb(${r},${g},${b}) ${x * 100}%`);
  }
  return `linear-gradient(90deg, ${stops.join(', ')})`;
}

async function start(): Promise<void> {
  let gpu;
  try {
    gpu = await initGpu();
  } catch (error) {
    fatal('This device cannot run the wind tunnel', error instanceof Error ? error.message : String(error));
    return;
  }
  gpu.device.lost.then((info) => {
    if (info.reason !== 'destroyed') fatal('The graphics device was lost', 'Reload the page to restart the tunnel. A lower quality setting uses less GPU memory.');
  });
  gpu.device.addEventListener('uncapturederror', (event) => {
    console.error('WebGPU error:', (event as GPUUncapturedErrorEvent).error.message);
  });

  const sim = new Simulation(gpu, $<HTMLCanvasElement>('view'));
  const tiers = sim.availableTiers();
  if (tiers.length === 0) {
    fatal('Not enough GPU memory', 'This device cannot allocate even the smallest simulation grid.');
    return;
  }

  let activePreset: Preset | null = null;
  let imported = false;
  let settledHandled = false;
  const firstCd = new Map<string, number>();
  const completed = loadCompleted();
  let currentReport: (BuiltReport & { thumbnail: string; cd: number }) | null = null;

  // ----- Quality -----
  const quality = $<HTMLSelectElement>('quality');
  for (const t of tiers) quality.add(new Option(`${t.label} (${t.nx} × ${t.ny} × ${t.nz})`, t.id));
  const defaultTier = tiers.find((t) => t.id === 'medium') ?? tiers[tiers.length - 1];
  quality.value = defaultTier.id;
  sim.setQuality(defaultTier);
  quality.addEventListener('change', () => {
    const tier = tiers.find((t) => t.id === quality.value) as QualityTier;
    try {
      sim.setQuality(tier, true);
      sim.camera.frame(tier.nx, tier.ny, tier.nz);
    } catch (error) {
      toast(error instanceof Error ? error.message : String(error), 'error');
    }
  });

  // ----- Sliders with live read-outs -----
  const bindSlider = (id: string, format: (v: number) => string, onInput: (v: number) => void): HTMLInputElement => {
    const input = $<HTMLInputElement>(id);
    const output = input.parentElement?.querySelector('output');
    const refresh = (): void => {
      if (output) output.textContent = format(Number(input.value));
    };
    input.addEventListener('input', () => {
      refresh();
      onInput(Number(input.value));
    });
    input.addEventListener('refresh', refresh);
    refresh();
    return input;
  };
  const setSlider = (input: HTMLInputElement, value: number): void => {
    input.value = String(value);
    input.dispatchEvent(new Event('refresh'));
  };
  const deg = (v: number): string => `${v}°`;
  const pct = (v: number): string => `${Math.round(v * 100)}%`;

  const yaw = bindSlider('yaw', deg, (v) => sim.setOrientation({ yaw: v }));
  const pitch = bindSlider('pitch', deg, (v) => sim.setOrientation({ pitch: v }));
  const roll = bindSlider('roll', deg, (v) => sim.setOrientation({ roll: v }));
  const ground = $<HTMLInputElement>('ground');
  ground.addEventListener('change', () => sim.setOrientation({ onGround: ground.checked }));

  bindSlider('wind', (v) => `${v} km/h`, (v) => {
    sim.setConditions({ windKmh: v });
    refreshLegend();
  });
  const size = $<HTMLInputElement>('size');
  const mass = $<HTMLInputElement>('mass');
  size.addEventListener('change', () => {
    const v = Number(size.value);
    if (v > 0) sim.setConditions({ realSize: v });
  });
  mass.addEventListener('change', () => {
    const v = Number(mass.value);
    if (v > 0) sim.setConditions({ massKg: v });
  });

  const syncModelControls = (): void => {
    const o = sim.currentOrientation;
    setSlider(yaw, o.yaw);
    setSlider(pitch, o.pitch);
    setSlider(roll, o.roll);
    ground.checked = o.onGround;
    const c = sim.currentConditions;
    size.value = String(c.realSize);
    mass.value = String(c.massKg);
  };

  // ----- Models -----
  const presetBox = $('presets');
  const presetButtons = new Map<string, HTMLButtonElement>();
  const markPreset = (): void => {
    for (const [id, b] of presetButtons) b.setAttribute('aria-pressed', String(activePreset?.id === id));
  };
  const useModel = (model: Model, preset: Preset | null): void => {
    activePreset = preset;
    imported = preset === null;
    settledHandled = false;
    sim.setModel(model, { pitch: preset?.pitch ?? 0 });
    // Grounded objects get the rake lowered so the smoke meets them.
    setRakeUi(model.grounded ? { height: 0.3, spread: 0.5 } : { height: 0.5, spread: 0.6 });
    syncModelControls();
    markPreset();
  };
  for (const preset of PRESETS) {
    const b = document.createElement('button');
    b.className = 'preset';
    const emoji = document.createElement('span');
    emoji.className = 'emoji';
    emoji.textContent = preset.emoji;
    b.append(emoji, preset.label);
    b.addEventListener('click', () => useModel(preset.build(), preset));
    presetButtons.set(preset.id, b);
    presetBox.append(b);
  }

  const importFile = async (file: File): Promise<void> => {
    toast(`Reading ${file.name}…`);
    try {
      const model = await loadModelFile(file);
      useModel(model, null);
      toast(`${model.name}: ${model.triangleCount.toLocaleString('en-US')} triangles. Set its real size and mass for accurate forces.`);
    } catch (error) {
      console.error('Model import failed', { file: file.name, error });
      toast(error instanceof Error ? error.message : 'The file could not be read.', 'error');
    }
  };
  const fileInput = $<HTMLInputElement>('file');
  fileInput.addEventListener('change', () => {
    const file = fileInput.files?.[0];
    if (file) void importFile(file);
    fileInput.value = '';
  });
  const stage = $('stage');
  const overlay = $('drop-overlay');
  let dragDepth = 0;
  stage.addEventListener('dragenter', (e) => {
    e.preventDefault();
    dragDepth++;
    overlay.hidden = false;
  });
  stage.addEventListener('dragover', (e) => e.preventDefault());
  stage.addEventListener('dragleave', () => {
    dragDepth = Math.max(0, dragDepth - 1);
    if (dragDepth === 0) overlay.hidden = true;
  });
  stage.addEventListener('drop', (e) => {
    e.preventDefault();
    dragDepth = 0;
    overlay.hidden = true;
    const file = e.dataTransfer?.files?.[0];
    if (!file) return;
    const ext = file.name.split('.').pop()?.toLowerCase() ?? '';
    if (!SUPPORTED_EXTENSIONS.includes(ext)) {
      toast(`".${ext}" files are not supported. Use ${SUPPORTED_EXTENSIONS.map((x) => `.${x}`).join(', ')}.`, 'error');
      return;
    }
    void importFile(file);
  });

  $('reset-orientation').addEventListener('click', () => {
    const model = sim.currentModel;
    sim.setOrientation({ yaw: 0, pitch: activePreset?.pitch ?? 0, roll: 0, onGround: model?.grounded ?? false });
    syncModelControls();
  });

  // ----- View toggles -----
  const bindCheck = (id: string, get: () => boolean, set: (v: boolean) => void): HTMLInputElement => {
    const input = $<HTMLInputElement>(id);
    input.checked = get();
    input.addEventListener('change', () => {
      set(input.checked);
      refreshNested();
      refreshLegend();
    });
    return input;
  };
  bindCheck('v-lines', () => sim.view.smokeLines, (v) => (sim.view.smokeLines = v));
  bindCheck('v-line-color', () => sim.view.lineColor === 'speed', (v) => (sim.view.lineColor = v ? 'speed' : 'smoke'));
  bindCheck('v-pressure', () => sim.view.pressureMap, (v) => (sim.view.pressureMap = v));
  bindCheck('v-smoke', () => sim.view.volumetricSmoke, (v) => (sim.view.volumetricSmoke = v));
  bindCheck('v-slice', () => sim.view.slice, (v) => (sim.view.slice = v));
  bindCheck('v-vortex', () => sim.view.vortices, (v) => (sim.view.vortices = v));

  const sliceAxis = $<HTMLSelectElement>('slice-axis');
  const sliceField = $<HTMLSelectElement>('slice-field');
  sliceAxis.value = sim.view.sliceAxis;
  sliceField.value = sim.view.sliceField;
  sliceAxis.addEventListener('change', () => (sim.view.sliceAxis = sliceAxis.value as SliceAxis));
  sliceField.addEventListener('change', () => {
    sim.view.sliceField = sliceField.value as SliceField;
    refreshLegend();
  });
  bindSlider('slice-pos', pct, (v) => (sim.view.slicePosition = v));
  bindSlider('vortex-threshold', (v) => v.toFixed(2), (v) => (sim.view.vortexThreshold = v));

  function refreshNested(): void {
    $('slice-options').hidden = !sim.view.slice;
    $('vortex-options').hidden = !sim.view.vortices;
  }

  // ----- Rake -----
  const rakeLayout = $<HTMLSelectElement>('rake-layout');
  rakeLayout.value = sim.rakeSettings.layout;
  rakeLayout.addEventListener('change', () => sim.setRake({ layout: rakeLayout.value as RakeLayout }));
  bindSlider('rake-count', (v) => String(v), (v) => sim.setRake({ count: v }));
  const rakeHeight = bindSlider('rake-height', pct, (v) => sim.setRake({ height: v }));
  bindSlider('rake-side', pct, (v) => sim.setRake({ sideways: v }));
  const rakeSpread = bindSlider('rake-spread', pct, (v) => sim.setRake({ spread: v }));
  function setRakeUi(r: { height: number; spread: number }): void {
    sim.setRake(r);
    setSlider(rakeHeight, r.height);
    setSlider(rakeSpread, r.spread);
  }

  // ----- Legend -----
  function refreshLegend(): void {
    const v = sim.view;
    const legend = $('legend');
    legend.replaceChildren();
    const add = (title: string, gradient: string, lo: string, hi: string): void => {
      const item = document.createElement('div');
      item.className = 'legend-item';
      const bar = document.createElement('div');
      bar.className = 'legend-bar';
      bar.style.background = gradient;
      const ends = document.createElement('div');
      ends.className = 'legend-ends';
      const a = document.createElement('span');
      a.textContent = lo;
      const b = document.createElement('span');
      b.textContent = hi;
      ends.append(a, b);
      item.append(title, bar, ends);
      legend.append(item);
    };
    const speedShown = (v.smokeLines && v.lineColor === 'speed') || (v.slice && v.sliceField === 'speed') || v.vortices;
    const pressureShown = v.pressureMap || (v.slice && v.sliceField === 'pressure');
    const kmh = sim.currentConditions.windKmh;
    if (speedShown) add('Air speed', turboGradient(), '0', `${Math.round(kmh * 1.6)} km/h`);
    if (pressureShown) add('Pressure', 'linear-gradient(90deg, rgb(33,77,199), rgb(237,235,230), rgb(204,26,31))', 'suction', 'push');
    if (v.slice && v.sliceField === 'vorticity') add('Swirl', turboGradient(), 'calm', 'spinning');
  }

  // ----- Simulation controls -----
  const pause = $<HTMLButtonElement>('pause');
  pause.addEventListener('click', () => {
    sim.paused = !sim.paused;
    pause.textContent = sim.paused ? 'Resume' : 'Pause';
  });
  $('restart').addEventListener('click', () => {
    sim.restartFlow();
    settledHandled = false;
  });

  // ----- Challenges -----
  const challengeDialog = $<HTMLDialogElement>('challenge-dialog');
  const renderChallenges = (): void => {
    $('challenge-count').textContent = `${completed.size}/${CHALLENGES.length}`;
    const list = $('challenge-list');
    list.replaceChildren();
    for (const c of CHALLENGES) {
      const li = document.createElement('li');
      if (completed.has(c.id)) li.className = 'done';
      const emoji = document.createElement('span');
      emoji.className = 'emoji';
      emoji.textContent = completed.has(c.id) ? '✅' : c.emoji;
      const body = document.createElement('div');
      body.className = 'body';
      const title = document.createElement('div');
      title.className = 'title';
      title.textContent = c.title;
      const desc = document.createElement('div');
      desc.className = 'desc';
      desc.textContent = c.hint;
      body.append(title, desc);
      li.append(emoji, body);
      list.append(li);
    }
  };
  $('open-challenges').addEventListener('click', () => {
    renderChallenges();
    challengeDialog.showModal();
  });

  // ----- Gallery -----
  const galleryDialog = $<HTMLDialogElement>('gallery-dialog');
  const reportDialog = $<HTMLDialogElement>('report-dialog');
  const reportFrame = $<HTMLIFrameElement>('report-frame');
  const showReportHtml = (title: string, html: string, canSave: boolean): void => {
    $('report-title').textContent = title;
    reportFrame.srcdoc = html;
    $('report-save').hidden = !canSave;
    if (!reportDialog.open) reportDialog.showModal();
  };
  let shownHtml = '';
  let shownTitle = 'report';

  const renderGallery = async (): Promise<void> => {
    const list = $('gallery-list');
    list.replaceChildren();
    let reports;
    try {
      reports = await listReports();
    } catch (error) {
      console.error('Could not read the gallery', error);
      toast('Saved reports are unavailable in this browser session.', 'error');
      return;
    }
    if (reports.length === 0) {
      const li = document.createElement('li');
      li.className = 'empty';
      li.textContent = 'Nothing saved yet. Make a report and press “Save to gallery”.';
      list.append(li);
      return;
    }
    for (const r of reports) {
      const li = document.createElement('li');
      const img = document.createElement('img');
      img.src = r.thumbnail;
      img.alt = '';
      const body = document.createElement('div');
      body.className = 'body';
      const title = document.createElement('div');
      title.className = 'title';
      title.textContent = `${r.name} · grade ${r.grade}`;
      const desc = document.createElement('div');
      desc.className = 'desc';
      desc.textContent = `Cd ${fmt(r.cd)} · ${new Date(r.savedAt).toLocaleString('en-GB', { dateStyle: 'medium', timeStyle: 'short' })}`;
      body.append(title, desc);
      const open = document.createElement('button');
      open.className = 'small';
      open.textContent = 'Open';
      open.addEventListener('click', () => {
        galleryDialog.close();
        shownHtml = r.html;
        shownTitle = r.name;
        showReportHtml(`Science & Fun Report: ${r.name}`, r.html, false);
      });
      const remove = document.createElement('button');
      remove.className = 'small';
      remove.textContent = 'Delete';
      remove.addEventListener('click', async () => {
        if (!confirm(`Delete the saved report for “${r.name}”? This cannot be undone.`)) return;
        try {
          await deleteReport(r.id);
        } catch (error) {
          console.error('Could not delete the report', { id: r.id, error });
          toast('The report could not be deleted.', 'error');
        }
        void renderGallery();
      });
      li.append(img, body, open, remove);
      list.append(li);
    }
  };
  $('open-gallery').addEventListener('click', () => {
    void renderGallery();
    galleryDialog.showModal();
  });
  for (const d of [challengeDialog, galleryDialog]) {
    d.querySelector('[data-close]')?.addEventListener('click', () => d.close());
  }

  // ----- Report -----
  const captureImages = (): ReportImage[] => {
    const images: ReportImage[] = [{ caption: 'The view when the report was made', src: sim.capture() }];
    const cam = sim.camera;
    const savedCam = { target: cam.target, distance: cam.distance, azimuth: cam.azimuth, elevation: cam.elevation };
    const savedView = { ...sim.view };
    const tier = sim.currentTier;
    cam.frame(tier.nx, tier.ny, tier.nz);
    cam.azimuth = 0;
    cam.elevation = 0.06;
    cam.distance = tier.nx * 1.05;
    Object.assign(sim.view, {
      slice: true, sliceAxis: 'z', sliceField: 'speed', slicePosition: 0.5,
      smokeLines: false, volumetricSmoke: false, vortices: false, pressureMap: false,
    });
    images.push({ caption: 'Side view: air speed on a slice through the middle (blue is slow, red is fast)', src: sim.capture() });
    Object.assign(sim.view, savedView);
    Object.assign(cam, savedCam);
    sim.capture();
    return images;
  };

  $('make-report').addEventListener('click', () => {
    const snapshot = sim.snapshot();
    const model = sim.currentModel;
    if (!snapshot || !model) return;
    const images = captureImages();
    const report = buildReport({
      modelName: model.name,
      imported,
      triangleCount: model.triangleCount,
      snapshot,
      conditions: sim.currentConditions,
      orientation: sim.currentOrientation,
      history: sim.history(),
      images,
      expected:
        activePreset?.expectedCd !== undefined && isDefaultPose()
          ? { cd: activePreset.expectedCd, note: activePreset.expectedNote ?? activePreset.label.toLowerCase() }
          : undefined,
      createdAt: new Date(),
    });
    currentReport = { ...report, thumbnail: images[0].src, cd: snapshot.measurement.cd };
    shownHtml = report.html;
    shownTitle = model.name;
    showReportHtml(report.title, report.html, true);
  });
  const isDefaultPose = (): boolean => {
    const o = sim.currentOrientation;
    return o.yaw === 0 && o.roll === 0 && o.pitch === (activePreset?.pitch ?? 0);
  };
  $('report-close').addEventListener('click', () => reportDialog.close());
  $('report-print').addEventListener('click', () => reportFrame.contentWindow?.print());
  $('report-download').addEventListener('click', () => {
    const url = URL.createObjectURL(new Blob([shownHtml], { type: 'text/html' }));
    const a = document.createElement('a');
    a.href = url;
    a.download = `wind-tunnel-report-${shownTitle.replace(/[^\w-]+/g, '-').toLowerCase() || 'object'}.html`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  $('report-save').addEventListener('click', async () => {
    if (!currentReport) return;
    try {
      await saveReport({
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        name: shownTitle,
        savedAt: Date.now(),
        cd: currentReport.cd,
        grade: currentReport.grade.letter,
        thumbnail: currentReport.thumbnail,
        html: currentReport.html,
      });
      toast('Report saved to the gallery.', 'win');
    } catch (error) {
      console.error('Could not save the report', error);
      toast('The report could not be saved in this browser.', 'error');
    }
  });

  // ----- Head-up display -----
  const hudCd = $('hud-cd'), hudCl = $('hud-cl'), hudLd = $('hud-ld');
  const hudProgress = $('hud-progress'), hudStatus = $('hud-status');
  const updateHud = (): void => {
    const s = sim.snapshot();
    const model = sim.currentModel;
    if (!s || !model) return;
    const m = s.measurement;
    hudCd.textContent = fmt(m.cd);
    hudCl.textContent = fmt(m.cl);
    hudLd.textContent = m.averagedTime > 0 ? fmt(m.liftToDrag) : '–';
    hudProgress.style.width = `${Math.round(m.progress * 100)}%`;
    hudProgress.classList.toggle('done', m.settled);
    hudStatus.textContent = sim.paused
      ? 'Paused'
      : m.settled
        ? 'Settled. Averaging continues.'
        : m.averagedTime > 0
          ? `Measuring… ${Math.round(m.progress * 100)}%`
          : `Warming up… ${Math.round(m.progress * 100)}%`;
    $('model-info').textContent =
      `${model.name} · ${model.triangleCount.toLocaleString('en-US')} triangles · fills ${(s.grid.blockage * 100).toFixed(1)}% of the tunnel`;
    $('sim-info').textContent =
      `${Math.round(s.stepsPerSecond)} steps/s · Reynolds ${s.flow.simulatedReynolds.toExponential(1)} simulated, ${s.flow.realReynolds.toExponential(1)} real`;

    if (!m.settled) {
      settledHandled = false;
      return;
    }
    if (settledHandled) return;
    settledHandled = true;
    const key = model.name;
    const context = { snapshot: s, modelName: model.name, imported, firstCd: firstCd.get(key) ?? null };
    if (!firstCd.has(key)) firstCd.set(key, m.cd);
    const won = evaluate(context, completed);
    for (const c of won) {
      completed.add(c.id);
      toast(`${c.emoji} Challenge complete: ${c.title}`, 'win');
    }
    if (won.length) saveCompleted(completed);
    renderChallenges();
  };

  // ----- Go -----
  refreshNested();
  renderChallenges();
  const first = PRESETS[0];
  useModel(first.build(), first);
  refreshLegend();

  if (import.meta.env.DEV) (window as unknown as { __app: unknown }).__app = { sim, useModel, PRESETS };

  let last = performance.now();
  let lastHud = 0;
  const loop = (now: number): void => {
    sim.frame(now, last);
    last = now;
    if (now - lastHud > 200) {
      lastHud = now;
      updateHud();
    }
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}

void start();
