"""
api/ui_html.py — self-contained HTML for the Sling tactical viewer.
Served by GET /ui in api/server.py (DEV_MODE / local use only).
"""

UI_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Sling — Tactical Intelligence</title>
  <style>
    :root {
      --bg:       #0d1117;
      --surface:  #161b22;
      --border:   #30363d;
      --home:     #4da8ff;
      --away:     #ff6b6b;
      --accent:   #58a6ff;
      --text:     #e6edf3;
      --muted:    #8b949e;
      --settled:  #3fb950;
      --pitch-bg: #1a472a;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: system-ui, -apple-system, sans-serif;
      font-size: 13px;
      height: 100vh;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    /* ── Header ── */
    header {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 10px 20px;
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      flex-shrink: 0;
    }
    .logo { font-size: 18px; font-weight: 700; letter-spacing: 2px; color: var(--accent); }
    .logo-sub { font-size: 11px; color: var(--muted); letter-spacing: 1px; }
    .badge {
      margin-left: auto;
      font-size: 10px;
      padding: 2px 8px;
      border-radius: 20px;
      border: 1px solid var(--border);
      color: var(--muted);
    }

    /* ── Main layout ── */
    .main {
      display: flex;
      flex: 1;
      overflow: hidden;
    }

    /* ── Pitch area ── */
    .pitch-area {
      flex: 1;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 16px;
      gap: 12px;
    }
    canvas#pitch {
      border-radius: 8px;
      border: 2px solid var(--border);
      box-shadow: 0 0 40px rgba(77, 168, 255, 0.07);
      max-width: 100%;
    }

    /* ── Legend ── */
    .legend {
      display: flex;
      gap: 20px;
      align-items: center;
    }
    .legend-item {
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 11px;
      color: var(--muted);
    }
    .dot { width: 10px; height: 10px; border-radius: 50%; }

    /* ── Controls bar ── */
    .controls {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 8px 20px;
      background: var(--surface);
      border-top: 1px solid var(--border);
      flex-shrink: 0;
    }
    .frame-display {
      font-size: 11px;
      color: var(--muted);
      min-width: 90px;
    }
    .btn {
      background: var(--surface);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 4px 12px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 13px;
      transition: background 0.15s;
    }
    .btn:hover { background: var(--border); }
    .btn.active { border-color: var(--accent); color: var(--accent); }
    .slider-wrap {
      display: flex;
      align-items: center;
      gap: 8px;
      flex: 1;
    }
    input[type=range] {
      flex: 1;
      accent-color: var(--accent);
    }
    .speed-group { display: flex; gap: 4px; }
    .speed-btn {
      background: none;
      border: 1px solid var(--border);
      color: var(--muted);
      padding: 2px 7px;
      border-radius: 4px;
      cursor: pointer;
      font-size: 11px;
    }
    .speed-btn.active { border-color: var(--accent); color: var(--accent); }

    /* ── Sidebar ── */
    .sidebar {
      width: 260px;
      border-left: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      overflow-y: auto;
      background: var(--surface);
    }
    .sb-section {
      border-bottom: 1px solid var(--border);
      padding: 14px 16px;
    }
    .sb-title {
      font-size: 10px;
      letter-spacing: 1px;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 10px;
    }
    .formation-row {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 6px;
    }
    .team-dot { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
    .formation-name { font-size: 18px; font-weight: 700; letter-spacing: 1px; }
    .conf-bar-wrap {
      background: var(--bg);
      border-radius: 3px;
      height: 3px;
      margin-top: 3px;
      overflow: hidden;
    }
    .conf-bar { height: 100%; border-radius: 3px; transition: width 0.3s; }
    .settled-badge {
      display: inline-block;
      font-size: 9px;
      padding: 1px 6px;
      border-radius: 10px;
      border: 1px solid;
      margin-top: 4px;
    }
    .settled-yes  { color: var(--settled); border-color: var(--settled); }
    .settled-no   { color: var(--muted);   border-color: var(--border); }

    .metric-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 4px 0;
    }
    .metric-label { color: var(--muted); font-size: 11px; }
    .metric-value { font-size: 12px; font-weight: 600; }
    .metric-home  { color: var(--home); }
    .metric-away  { color: var(--away); }

    /* counter cards */
    .counter-card {
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 8px 10px;
      margin-bottom: 6px;
    }
    .counter-title { font-size: 11px; font-weight: 600; margin-bottom: 4px; }
    .counter-mech  { font-size: 10px; color: var(--muted); margin-bottom: 5px; line-height: 1.4; }
    .counter-conf-row { display: flex; align-items: center; gap: 6px; }
    .counter-conf-bar { flex: 1; height: 2px; background: var(--border); border-radius: 1px; }
    .counter-conf-fill { height: 100%; background: var(--accent); border-radius: 1px; transition: width 0.3s; }
    .counter-conf-pct { font-size: 9px; color: var(--muted); }
    .no-counters { font-size: 11px; color: var(--muted); font-style: italic; }

    /* loading */
    #loading {
      position: fixed;
      inset: 0;
      background: var(--bg);
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 12px;
      z-index: 100;
    }
    .spinner {
      width: 36px; height: 36px;
      border: 3px solid var(--border);
      border-top-color: var(--accent);
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    #loading p { color: var(--muted); font-size: 12px; }
  </style>
</head>
<body>

<div id="loading">
  <div class="spinner"></div>
  <p>Loading frames...</p>
</div>

<header>
  <div>
    <div class="logo">SLING</div>
    <div class="logo-sub">TACTICAL INTELLIGENCE</div>
  </div>
  <span class="badge" id="hdr-badge">positions demo</span>
</header>

<div class="main">

  <!-- ── Pitch ── -->
  <div class="pitch-area">
    <canvas id="pitch" width="700" height="455"></canvas>
    <div class="legend">
      <div class="legend-item"><div class="dot" style="background:var(--home)"></div>HOME</div>
      <div class="legend-item"><div class="dot" style="background:var(--away)"></div>AWAY</div>
      <div class="legend-item"><div class="dot" style="background:var(--settled);box-shadow:0 0 4px var(--settled)"></div>Settled</div>
    </div>
  </div>

  <!-- ── Sidebar ── -->
  <div class="sidebar">

    <!-- Formations -->
    <div class="sb-section">
      <div class="sb-title">Formations</div>

      <div class="formation-row">
        <div class="team-dot" style="background:var(--home)"></div>
        <div>
          <div class="formation-name" id="home-form">—</div>
          <div class="conf-bar-wrap" style="background:var(--border)">
            <div class="conf-bar" id="home-conf-bar" style="background:var(--home);width:0%"></div>
          </div>
          <div class="settled-badge" id="home-settled">—</div>
        </div>
      </div>

      <div class="formation-row" style="margin-top:10px">
        <div class="team-dot" style="background:var(--away)"></div>
        <div>
          <div class="formation-name" id="away-form">—</div>
          <div class="conf-bar-wrap" style="background:var(--border)">
            <div class="conf-bar" id="away-conf-bar" style="background:var(--away);width:0%"></div>
          </div>
          <div class="settled-badge" id="away-settled">—</div>
        </div>
      </div>
    </div>

    <!-- Metrics -->
    <div class="sb-section">
      <div class="sb-title">Metrics</div>
      <div class="metric-row">
        <span class="metric-label">Pressing height</span>
        <span class="metric-value metric-home" id="home-press">—</span>
      </div>
      <div class="metric-row">
        <span class="metric-label">Pressing height</span>
        <span class="metric-value metric-away" id="away-press">—</span>
      </div>
      <div class="metric-row">
        <span class="metric-label">Defensive line</span>
        <span class="metric-value metric-home" id="home-def">—</span>
      </div>
      <div class="metric-row">
        <span class="metric-label">Defensive line</span>
        <span class="metric-value metric-away" id="away-def">—</span>
      </div>
    </div>

    <!-- Counter suggestions -->
    <div class="sb-section" style="flex:1">
      <div class="sb-title">Counter Suggestions</div>
      <div id="counters-list"><p class="no-counters">Waiting for settled formation...</p></div>
    </div>

  </div>
</div>

<!-- Controls -->
<div class="controls">
  <button class="btn" id="btn-prev" title="Step back">&#9664;</button>
  <button class="btn active" id="btn-play" title="Play/Pause">&#9646;&#9646;</button>
  <button class="btn" id="btn-next" title="Step forward">&#9654;</button>

  <div class="slider-wrap">
    <input type="range" id="scrubber" min="0" value="0">
  </div>
  <div class="frame-display">Frame <span id="frame-num">0</span> / <span id="frame-total">0</span></div>

  <div class="speed-group">
    <button class="speed-btn" data-speed="0.5">0.5x</button>
    <button class="speed-btn active" data-speed="1">1x</button>
    <button class="speed-btn" data-speed="2">2x</button>
    <button class="speed-btn" data-speed="4">4x</button>
  </div>
</div>

<script>
// ── Constants ──────────────────────────────────────────────────────────────────
const PITCH_W_M  = 105;   // pitch width in metres
const PITCH_H_M  = 68;    // pitch height in metres
const PAD        = 30;    // canvas padding px
const DOT_R      = 7;     // player dot radius
const BASE_FPS   = 10;    // animation frames per second at 1x speed

// ── Canvas setup ──────────────────────────────────────────────────────────────
const canvas = document.getElementById('pitch');
const ctx    = canvas.getContext('2d');
const CW     = canvas.width;
const CH     = canvas.height;
const scaleX = (CW - PAD*2) / PITCH_W_M;
const scaleY = (CH - PAD*2) / PITCH_H_M;

function pitchX(m) { return PAD + m * scaleX; }
function pitchY(m) { return PAD + m * scaleY; }

// ── State ─────────────────────────────────────────────────────────────────────
let frames    = [];
let frameIdx  = 0;
let playing   = false;
let speed     = 1;
let lastTime  = 0;
let rafId     = null;

// ── Load frames ───────────────────────────────────────────────────────────────
fetch('/api/demo-frames')
  .then(r => { if (!r.ok) throw new Error(r.status); return r.json(); })
  .then(data => {
    frames = data;
    document.getElementById('loading').style.display = 'none';
    document.getElementById('scrubber').max = frames.length - 1;
    document.getElementById('frame-total').textContent = frames.length;
    document.getElementById('hdr-badge').textContent = frames.length + ' frames';
    render(0);
    startPlay();
  })
  .catch(e => {
    document.getElementById('loading').innerHTML =
      '<p style="color:#ff6b6b">Failed to load frames: ' + e.message + '<br>Make sure the API is running on :8000</p>';
  });

// ── Draw pitch ────────────────────────────────────────────────────────────────
function drawPitch() {
  // Background
  ctx.fillStyle = '#1a472a';
  ctx.fillRect(0, 0, CW, CH);

  // Alternating stripes
  const stripeW = (CW - PAD*2) / 8;
  for (let i = 0; i < 8; i++) {
    ctx.fillStyle = i % 2 === 0 ? 'rgba(0,0,0,0.12)' : 'rgba(0,0,0,0)';
    ctx.fillRect(PAD + i * stripeW, PAD, stripeW, CH - PAD*2);
  }

  ctx.strokeStyle = 'rgba(255,255,255,0.6)';
  ctx.lineWidth   = 1.5;

  // Pitch outline
  ctx.strokeRect(PAD, PAD, CW - PAD*2, CH - PAD*2);

  // Halfway line
  const midX = pitchX(PITCH_W_M / 2);
  ctx.beginPath(); ctx.moveTo(midX, PAD); ctx.lineTo(midX, CH - PAD); ctx.stroke();

  // Centre circle (r ~ 9.15m)
  ctx.beginPath();
  ctx.arc(midX, pitchY(PITCH_H_M/2), 9.15 * scaleX, 0, Math.PI*2);
  ctx.stroke();
  ctx.beginPath(); ctx.arc(midX, pitchY(PITCH_H_M/2), 2, 0, Math.PI*2);
  ctx.fillStyle = 'rgba(255,255,255,0.6)'; ctx.fill();

  // Penalty boxes (approx 16.5m deep, 40.3m wide)
  const penD = 16.5 * scaleX;
  const penTop = pitchY((PITCH_H_M - 40.3) / 2);
  const penBot = pitchY((PITCH_H_M + 40.3) / 2);
  ctx.strokeRect(PAD, penTop, penD, penBot - penTop);                        // left
  ctx.strokeRect(CW - PAD - penD, penTop, penD, penBot - penTop);            // right

  // Goal areas (5.5m deep, 18.3m wide)
  const gaD   = 5.5 * scaleX;
  const gaTop = pitchY((PITCH_H_M - 18.3) / 2);
  const gaBot = pitchY((PITCH_H_M + 18.3) / 2);
  ctx.strokeRect(PAD, gaTop, gaD, gaBot - gaTop);
  ctx.strokeRect(CW - PAD - gaD, gaTop, gaD, gaBot - gaTop);
}

// ── Draw players ──────────────────────────────────────────────────────────────
function drawTeam(positions, lines, color, settled) {
  if (!positions || positions.length === 0) return;

  // Draw formation lines first
  ctx.strokeStyle = color.replace(')', ', 0.30)').replace('rgb(', 'rgba(').replace('#', null);
  // use a simpler approach
  ctx.lineWidth = 1.5;
  if (lines && lines.length > 0) {
    lines.forEach(lineGroup => {
      // connect players in this line group sequentially by y-coordinate
      const pts = lineGroup
        .filter(i => i < positions.length)
        .map(i => positions[i])
        .sort((a, b) => a[1] - b[1]);
      if (pts.length < 2) return;
      ctx.beginPath();
      ctx.moveTo(pitchX(pts[0][0]), pitchY(pts[0][1]));
      for (let i = 1; i < pts.length; i++) {
        ctx.lineTo(pitchX(pts[i][0]), pitchY(pts[i][1]));
      }
      ctx.strokeStyle = hexToRgba(color, 0.3);
      ctx.stroke();
    });
  }

  // Draw dots
  positions.forEach(([x, y], idx) => {
    const px = pitchX(x);
    const py = pitchY(y);

    // Glow if settled
    if (settled) {
      ctx.beginPath();
      ctx.arc(px, py, DOT_R + 4, 0, Math.PI*2);
      ctx.fillStyle = hexToRgba(color, 0.15);
      ctx.fill();
    }

    // Dot
    ctx.beginPath();
    ctx.arc(px, py, DOT_R, 0, Math.PI*2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = 'rgba(255,255,255,0.5)';
    ctx.lineWidth = 1;
    ctx.stroke();

    // Player number
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 7px system-ui';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(idx + 1, px, py);
  });
}

function hexToRgba(hex, a) {
  if (hex.startsWith('var(')) return `rgba(100,180,255,${a})`;
  const r = parseInt(hex.slice(1,3),16);
  const g = parseInt(hex.slice(3,5),16);
  const b = parseInt(hex.slice(5,7),16);
  return `rgba(${r},${g},${b},${a})`;
}

// ── Draw pressing/defensive lines ─────────────────────────────────────────────
function drawTacticalLines(f) {
  if (!f) return;
  const home = f.home;
  const away = f.away;

  if (home && home.pressing_height != null) {
    const x = pitchX(home.pressing_height);
    ctx.beginPath(); ctx.moveTo(x, PAD); ctx.lineTo(x, CH - PAD);
    ctx.strokeStyle = 'rgba(77,168,255,0.25)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.stroke();
    ctx.setLineDash([]);
  }
  if (away && away.pressing_height != null) {
    const x = pitchX(PITCH_W_M - away.pressing_height); // mirror for away
    ctx.beginPath(); ctx.moveTo(x, PAD); ctx.lineTo(x, CH - PAD);
    ctx.strokeStyle = 'rgba(255,107,107,0.25)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.stroke();
    ctx.setLineDash([]);
  }
}

// ── Interpolate positions ─────────────────────────────────────────────────────
function interpolate(posA, posB, t) {
  if (!posA || !posB || posA.length !== posB.length) return posA || posB || [];
  return posA.map((p, i) => [
    p[0] + (posB[i][0] - p[0]) * t,
    p[1] + (posB[i][1] - p[1]) * t,
  ]);
}

// ── Render a frame ────────────────────────────────────────────────────────────
function render(idx, t) {
  if (frames.length === 0) return;
  idx = Math.max(0, Math.min(frames.length - 1, idx));
  const f    = frames[idx];
  const fNext = frames[Math.min(idx + 1, frames.length - 1)];
  const lerp = t || 0;

  drawPitch();
  drawTacticalLines(f);

  const homePos = interpolate(f.home?.positions, fNext.home?.positions, lerp);
  const awayPos = interpolate(f.away?.positions, fNext.away?.positions, lerp);

  drawTeam(awayPos, f.away?.lines, '#ff6b6b', f.away?.is_settled);
  drawTeam(homePos, f.home?.lines, '#4da8ff', f.home?.is_settled);

  updateSidebar(f);

  // Controls
  document.getElementById('frame-num').textContent = idx + 1;
  document.getElementById('scrubber').value = idx;
}

// ── Sidebar ───────────────────────────────────────────────────────────────────
function fmt(v, unit) {
  if (v == null) return '—';
  return v.toFixed(1) + (unit || '');
}

function updateSidebar(f) {
  const home = f.home || {};
  const away = f.away || {};

  // Formations
  document.getElementById('home-form').textContent = home.formation || '—';
  document.getElementById('away-form').textContent = away.formation || '—';

  const hConf = (home.confidence || 0) * 100;
  const aConf = (away.confidence || 0) * 100;
  document.getElementById('home-conf-bar').style.width = hConf + '%';
  document.getElementById('away-conf-bar').style.width = aConf + '%';

  const hSett = document.getElementById('home-settled');
  const aSett = document.getElementById('away-settled');
  hSett.textContent = home.is_settled ? 'SETTLED' : 'forming';
  hSett.className   = 'settled-badge ' + (home.is_settled ? 'settled-yes' : 'settled-no');
  aSett.textContent = away.is_settled ? 'SETTLED' : 'forming';
  aSett.className   = 'settled-badge ' + (away.is_settled ? 'settled-yes' : 'settled-no');

  // Metrics
  document.getElementById('home-press').textContent = fmt(home.pressing_height, 'm');
  document.getElementById('away-press').textContent = fmt(away.pressing_height, 'm');
  document.getElementById('home-def').textContent   = fmt(home.defensive_line_x, 'm');
  document.getElementById('away-def').textContent   = fmt(away.defensive_line_x, 'm');

  // Counters
  const allCounters = [...(f.counters_home || []), ...(f.counters_away || [])];
  const list = document.getElementById('counters-list');
  if (allCounters.length === 0) {
    list.innerHTML = '<p class="no-counters">Waiting for settled formation...</p>';
  } else {
    list.innerHTML = allCounters.slice(0, 5).map(c => {
      const pct = Math.round((c.confidence || 0) * 100);
      return `
        <div class="counter-card">
          <div class="counter-title">${c.title || '—'}</div>
          <div class="counter-mech">${(c.mechanism || '').slice(0, 80)}</div>
          <div class="counter-conf-row">
            <div class="counter-conf-bar"><div class="counter-conf-fill" style="width:${pct}%"></div></div>
            <span class="counter-conf-pct">${pct}%</span>
          </div>
        </div>`;
    }).join('');
  }
}

// ── Animation loop ────────────────────────────────────────────────────────────
const frameInterval = () => 1000 / (BASE_FPS * speed);

function loop(ts) {
  if (!playing) return;
  rafId = requestAnimationFrame(loop);
  const elapsed = ts - lastTime;
  const interval = frameInterval();
  if (elapsed < interval) {
    // sub-frame interpolation
    render(frameIdx, elapsed / interval);
    return;
  }
  lastTime = ts;
  frameIdx = (frameIdx + 1) % frames.length;
  render(frameIdx, 0);
}

function startPlay() {
  if (playing) return;
  playing = true;
  lastTime = performance.now();
  document.getElementById('btn-play').innerHTML = '&#9646;&#9646;';
  document.getElementById('btn-play').classList.add('active');
  rafId = requestAnimationFrame(loop);
}

function stopPlay() {
  playing = false;
  if (rafId) cancelAnimationFrame(rafId);
  document.getElementById('btn-play').innerHTML = '&#9654;';
  document.getElementById('btn-play').classList.remove('active');
}

// ── Controls ──────────────────────────────────────────────────────────────────
document.getElementById('btn-play').addEventListener('click', () => {
  playing ? stopPlay() : startPlay();
});

document.getElementById('btn-prev').addEventListener('click', () => {
  stopPlay();
  frameIdx = Math.max(0, frameIdx - 1);
  render(frameIdx);
});

document.getElementById('btn-next').addEventListener('click', () => {
  stopPlay();
  frameIdx = Math.min(frames.length - 1, frameIdx + 1);
  render(frameIdx);
});

document.getElementById('scrubber').addEventListener('input', e => {
  stopPlay();
  frameIdx = parseInt(e.target.value);
  render(frameIdx);
});

document.querySelectorAll('.speed-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.speed-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    speed = parseFloat(btn.dataset.speed);
    if (playing) { lastTime = performance.now(); }
  });
});
</script>
</body>
</html>"""
