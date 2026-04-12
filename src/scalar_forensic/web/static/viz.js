// ── 3-D vector visualization (Canvas 2D, no WebGL) ──────────────────────────
// Self-contained: call initVectorViz(data) where data = { sscd: [[x,y,z],...],
//                                                         dino: [[x,y,z],...] }
// The canvas must have id="vec-canvas" and a sized parent element.
function initVectorViz(data) {
  const canvas = document.getElementById('vec-canvas');
  if (!canvas) return;
  const sscd = data.sscd || [];
  const dino = data.dino || [];
  if (!sscd.length && !dino.length) return;

  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  // ── Camera state (spherical coords, orbiting origin) ───────────────────
  let radius = 2.5;
  let theta  = 0.7;    // azimuth  (rotation around Y)
  let phi    = 1.15;   // polar    (0 = north pole, π = south pole)
  const PHI_MIN = 0.05, PHI_MAX = Math.PI - 0.05;
  let   zoomFactor = 1.0;
  const Z_MIN = 0.25,   Z_MAX = 4.0;

  // ── Perspective projection ──────────────────────────────────────────────
  // Rotate world so the camera lands at (0, 0, radius), then apply perspective.
  function project(px, py, pz) {
    const W = canvas.width, H = canvas.height;
    // 1. Rotate around Y by -theta
    const x1 =  px * Math.cos(theta) - pz * Math.sin(theta);
    const z1 =  px * Math.sin(theta) + pz * Math.cos(theta);
    // 2. Rotate around X by (π/2 - phi)
    const sp = Math.sin(phi), cp = Math.cos(phi);
    const y2 =  py * sp - z1 * cp;
    const z2 =  py * cp + z1 * sp;
    // 3. Perspective (camera at z=radius, looking toward -Z)
    const d = radius - z2;
    if (d < 0.01) return null;
    const unit = Math.min(W, H) * 0.52 * zoomFactor;
    const s = (radius * unit) / d;
    return { sx: x1 * s + W * 0.5, sy: -y2 * s + H * 0.5, depth: z2, s };
  }

  // ── Draw one axis segment (line between two 3D points) ─────────────────
  function seg(x0, y0, z0, x1, y1, z1) {
    const a = project(x0, y0, z0), b = project(x1, y1, z1);
    if (!a || !b) return;
    ctx.moveTo(a.sx, a.sy);
    ctx.lineTo(b.sx, b.sy);
  }

  // ── Draw axes + tick marks ─────────────────────────────────────────────
  function drawAxes() {
    const L = 1.25, T = 0.045;
    const TICKS = [-1.0, -0.5, 0.5, 1.0];
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(107,122,141,0.55)';
    ctx.lineWidth   = 1;
    seg(-L, 0, 0, L, 0, 0);
    for (const v of TICKS) seg(v, -T, 0, v, T, 0);
    seg(0, -L, 0, 0, L, 0);
    for (const v of TICKS) seg(-T, v, 0, T, v, 0);
    seg(0, 0, -L, 0, 0, L);
    for (const v of TICKS) seg(-T, 0, v, T, 0, v);
    ctx.stroke();
  }

  // ── Draw a point cloud, batched by depth bucket ───────────────────────────
  // Depth shading requires varying alpha per point, but drawing each point
  // individually is O(N) draw calls. Instead we bin points into N_BUCKETS
  // depth levels and batch each bin into one path → O(N_BUCKETS) draw calls.
  const N_BUCKETS = 8;
  function drawCloud(pts, cr, cg, cb) {
    const unit    = Math.min(canvas.width, canvas.height) * 0.52 * zoomFactor;
    const buckets = Array.from({ length: N_BUCKETS }, () => []);
    for (let i = 0; i < pts.length; i++) {
      const q = project(pts[i][0], pts[i][1], pts[i][2]);
      if (!q) continue;
      const t  = Math.max(0, Math.min(0.9999, (q.depth + 1.15) / 2.3));
      q._r     = Math.max(1.0, q.s / unit * 2.2);
      buckets[Math.floor(t * N_BUCKETS)].push(q);
    }
    for (let b = 0; b < N_BUCKETS; b++) {
      if (!buckets[b].length) continue;
      const alpha = (0.12 + ((b + 0.5) / N_BUCKETS) * 0.78).toFixed(3);
      ctx.beginPath();
      ctx.fillStyle = `rgba(${cr},${cg},${cb},${alpha})`;
      for (const q of buckets[b]) {
        ctx.moveTo(q.sx + q._r, q.sy);   // moveTo prevents connecting lines
        ctx.arc(q.sx, q.sy, q._r, 0, Math.PI * 2);
      }
      ctx.fill();
    }
  }

  // ── Radial gradient background — dark centre, subtle glow at edges ───────
  // Cached and rebuilt only on resize; creating a gradient every frame is
  // unnecessary work. Darker centre = true black on OLED (pixels off).
  let bgGradient = null;
  function rebuildBgGradient() {
    const W = canvas.width, H = canvas.height;
    const g = ctx.createRadialGradient(W / 2, H / 2, 0, W / 2, H / 2, Math.hypot(W / 2, H / 2));
    g.addColorStop(0,    '#000000');
    g.addColorStop(0.45, '#03060e');
    g.addColorStop(1,    '#0b1222');
    bgGradient = g;
  }
  function drawBackground() {
    ctx.fillStyle = bgGradient;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }

  // ── Sync canvas pixel buffer to its CSS-displayed size ─────────────────
  function resize() {
    const p = canvas.parentElement;
    canvas.width  = p.clientWidth  || 800;
    canvas.height = p.clientHeight || 400;
    rebuildBgGradient();
  }
  resize();
  const ro = new ResizeObserver(resize);
  ro.observe(canvas.parentElement);

  // ── Spin state ─────────────────────────────────────────────────────────
  // After user interaction, spin pauses then fades back in over BLEND_TIME.
  const SPIN_BASE      = 0.01 * Math.PI; // 20 % of original 0.05 * π
  const IDLE_DELAY     = 3.0;            // seconds idle before spin resumes
  const BLEND_TIME     = 2.0;            // seconds to fade spin back to full
  const AXIS_HOLD_MIN  = 6;             // min seconds on one random axis
  const AXIS_HOLD_MAX  = 14;            // max seconds on one random axis

  let spinVTheta  = SPIN_BASE;  // angular velocity on theta
  let spinVPhi    = 0;          // angular velocity on phi
  let nextAxisAt  = 0;          // clock-time to pick a new random axis
  let lastActT    = -999;       // last time user interacted (seconds from t0)
  let spinBlend   = 1;          // 0 = fully paused, 1 = full speed

  function pickAxis(t) {
    const r = Math.random();
    if (r < 0.35) {
      // Pure Y rotation (most recognisable, feels like a globe)
      spinVTheta = (Math.random() > 0.5 ? 1 : -1) * SPIN_BASE;
      spinVPhi   = 0;
    } else if (r < 0.65) {
      // Y + slow phi drift (tilted tumble)
      spinVTheta = (Math.random() > 0.5 ? 1 : -1) * SPIN_BASE;
      spinVPhi   = (Math.random() - 0.5) * SPIN_BASE * 0.5;
    } else {
      // Diagonal spin (feels like free tumble)
      spinVTheta = (Math.random() - 0.5) * SPIN_BASE * 1.6;
      spinVPhi   = (Math.random() - 0.5) * SPIN_BASE * 0.7;
    }
    nextAxisAt = t + AXIS_HOLD_MIN + Math.random() * (AXIS_HOLD_MAX - AXIS_HOLD_MIN);
  }
  pickAxis(0); // initialise first axis immediately

  // ── Spark (glowing dino↔sscd pair) system ─────────────────────────────
  const SPARK_FADE_IN   = 0.6;   // s to reach full brightness
  const SPARK_FADE_OUT  = 1.2;   // s to fade back to nothing
  const SPARK_HOLD_MIN  = 0.8;   // s at full brightness (min)
  const SPARK_HOLD_MAX  = 2.5;   // s at full brightness (max)
  const smooth = x => x * x * (3 - 2 * x); // smoothstep for nicer easing

  // Pre-baked glow sprite — rendered once into an offscreen canvas.
  // drawImage + globalAlpha is far cheaper than ctx.shadowBlur per frame.
  const GLOW_SZ = 64;
  function makeGlowSprite(c0, c1, c2, c3) {
    const oc = document.createElement('canvas');
    oc.width = oc.height = GLOW_SZ;
    const og = oc.getContext('2d');
    const h  = GLOW_SZ / 2;
    const g  = og.createRadialGradient(h, h, 0, h, h, h);
    g.addColorStop(0,    c0); g.addColorStop(0.2,  c1);
    g.addColorStop(0.55, c2); g.addColorStop(1,    c3);
    og.fillStyle = g; og.fillRect(0, 0, GLOW_SZ, GLOW_SZ);
    return oc;
  }
  const glowSprite    = makeGlowSprite('rgba(220,255,240,1.0)', 'rgba(80,255,120,0.85)',
                                        'rgba(60,230,100,0.3)',  'rgba(60,230,100,0)');
  const redGlowSprite = makeGlowSprite('rgba(255,220,200,1.0)', 'rgba(255,70,50,0.85)',
                                        'rgba(200,30,20,0.3)',   'rgba(180,20,10,0)');

  let sparks = [];   // { di, si, born, duration } — populated by traverser hits

  function drawSparks(t) {
    sparks = sparks.filter(sp => t - sp.born < sp.duration);
    if (!sparks.length) return;

    const unit = Math.min(canvas.width, canvas.height) * 0.52 * zoomFactor;
    ctx.save();
    ctx.lineWidth = 1;

    for (const sp of sparks) {
      const age = t - sp.born;
      const a   = smooth(Math.min(1, age / SPARK_FADE_IN))
                * smooth(Math.min(1, (sp.duration - age) / SPARK_FADE_OUT));
      if (a <= 0) continue;

      const pD = project(dino[sp.di][0], dino[sp.di][1], dino[sp.di][2]);
      const pS = project(sscd[sp.si][0], sscd[sp.si][1], sscd[sp.si][2]);
      if (!pD || !pS) continue;

      // Connecting line — no shadowBlur, just a clean alpha stroke
      ctx.beginPath();
      ctx.strokeStyle = `rgba(60,230,100,${(a * 0.5).toFixed(3)})`;
      ctx.moveTo(pD.sx, pD.sy);
      ctx.lineTo(pS.sx, pS.sy);
      ctx.stroke();

      const rD     = Math.max(2, pD.s / unit * 2.2) * 2.2;
      const rS     = Math.max(2, pS.s / unit * 2.2) * 2.2;
      const pulseT = smooth(Math.min(1, age / sp.duration));
      const pulseX = pD.sx + (pS.sx - pD.sx) * pulseT;
      const pulseY = pD.sy + (pS.sy - pD.sy) * pulseT;

      // Glow halos via pre-baked sprite — replaces ctx.shadowBlur entirely
      ctx.globalAlpha = a;
      ctx.drawImage(glowSprite, pulseX - 18,     pulseY - 18,     36,      36);
      ctx.drawImage(glowSprite, pD.sx - rD * 4,  pD.sy - rD * 4,  rD * 8,  rD * 8);
      ctx.drawImage(glowSprite, pS.sx - rS * 4,  pS.sy - rS * 4,  rS * 8,  rS * 8);
      ctx.globalAlpha = 1;

      // Crisp center dots drawn on top of the halos
      ctx.beginPath();
      ctx.fillStyle = `rgba(215,255,235,${(a * 0.98).toFixed(3)})`;
      ctx.arc(pulseX, pulseY, 3, 0, Math.PI * 2);
      ctx.fill();

      ctx.beginPath();
      ctx.fillStyle = `rgba(80,255,120,${(a * 0.95).toFixed(3)})`;
      ctx.moveTo(pD.sx + rD, pD.sy);
      ctx.arc(pD.sx, pD.sy, rD, 0, Math.PI * 2);
      ctx.moveTo(pS.sx + rS, pS.sy);
      ctx.arc(pS.sx, pS.sy, rS, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.restore();
  }

  // ── Traverser (query flythrough) ──────────────────────────────────────────
  // Represents a single image vector flying through the embedding space.
  // Fires 1-5 red beams to nearby cloud points mid-flight, which each
  // inject a green spark so the existing connector animation takes over.
  const TRAV_FADE_IN  = 0.5;
  const TRAV_FADE_OUT = 1.0;
  const TRAV_DUR_MIN  = 4.0;    // s — minimum traversal time
  const TRAV_DUR_MAX  = 8.0;    // s — maximum traversal time
  const TRAV_GAP_MIN  = 5.0;    // s — minimum gap between traversers
  const TRAV_GAP_MAX  = 30.0;   // s — maximum gap between traversers
  const BEAM_DUR      = 0.55;   // s — how long each red connection line lingers

  let traversers     = [];
  let nextTraverserT = 3 + Math.random() * 10; // first one appears 3-13 s after load

  function spawnTraverser(t) {
    if (!dino.length && !sscd.length) return;
    // Random unit direction
    const th = Math.random() * Math.PI * 2;
    const ph = Math.acos(2 * Math.random() - 1);
    const dx = Math.sin(ph) * Math.cos(th);
    const dy = Math.sin(ph) * Math.sin(th);
    const dz = Math.cos(ph);
    // Random offset so the path doesn't always pass through the origin
    const offR = Math.random() * 0.45;
    const oth  = Math.random() * Math.PI * 2;
    const oph  = Math.acos(2 * Math.random() - 1);
    const ox   = offR * Math.sin(oph) * Math.cos(oth);
    const oy   = offR * Math.sin(oph) * Math.sin(oth);
    const oz   = offR * Math.cos(oph);
    const reach    = 1.6;
    const duration = TRAV_DUR_MIN + Math.random() * (TRAV_DUR_MAX - TRAV_DUR_MIN);
    traversers.push({
      sx: ox - dx * reach, sy: oy - dy * reach, sz: oz - dz * reach,
      ex: ox + dx * reach, ey: oy + dy * reach, ez: oz + dz * reach,
      born: t, duration,
      triggerAt: 0.25 + Math.random() * 0.5, // fire connections at 25-75 % through
      triggered: false,
      beams: [],
    });
    nextTraverserT = t + TRAV_GAP_MIN + Math.random() * (TRAV_GAP_MAX - TRAV_GAP_MIN);
  }

  function drawTraversers(t) {
    traversers = traversers.filter(tr => t - tr.born < tr.duration);
    if (!traversers.length) return;
    ctx.save();
    ctx.lineWidth = 1;
    for (const tr of traversers) {
      const age      = t - tr.born;
      const progress = age / tr.duration;
      const a        = smooth(Math.min(1, age / TRAV_FADE_IN))
                     * smooth(Math.min(1, (tr.duration - age) / TRAV_FADE_OUT));
      if (a <= 0) continue;
      // Current 3-D position along the path
      const cx = tr.sx + (tr.ex - tr.sx) * progress;
      const cy = tr.sy + (tr.ey - tr.sy) * progress;
      const cz = tr.sz + (tr.ez - tr.sz) * progress;
      const p  = project(cx, cy, cz);
      if (!p) continue;
      // Fire connections once — injects beams and green sparks
      if (!tr.triggered && progress >= tr.triggerAt) {
        tr.triggered = true;
        const hasBoth = dino.length > 0 && sscd.length > 0;
        const count   = 1 + Math.floor(Math.random() * 5);
        for (let i = 0; i < count; i++) {
          const useDino = hasBoth ? Math.random() < 0.5 : dino.length > 0;
          const di = Math.floor(Math.random() * Math.max(1, dino.length));
          const si = Math.floor(Math.random() * Math.max(1, sscd.length));
          tr.beams.push({ pt: useDino ? dino[di] : sscd[si], born: t, duration: BEAM_DUR });
          // Green spark connector requires both collections to have valid points
          if (hasBoth) {
            const hold = SPARK_HOLD_MIN + Math.random() * (SPARK_HOLD_MAX - SPARK_HOLD_MIN);
            sparks.push({ di, si, born: t, duration: SPARK_FADE_IN + hold + SPARK_FADE_OUT });
          }
        }
      }
      // Red beams from traverser to hit cloud points
      for (const beam of tr.beams) {
        const ba = smooth(Math.max(0, 1 - (t - beam.born) / beam.duration));
        if (ba <= 0) continue;
        const bp = project(beam.pt[0], beam.pt[1], beam.pt[2]);
        if (!bp) continue;
        ctx.beginPath();
        ctx.strokeStyle = `rgba(255,90,60,${(ba * a * 0.75).toFixed(3)})`;
        ctx.moveTo(p.sx, p.sy);
        ctx.lineTo(bp.sx, bp.sy);
        ctx.stroke();
      }
      // Red glow halo + crisp centre dot
      ctx.globalAlpha = a;
      ctx.drawImage(redGlowSprite, p.sx - 22, p.sy - 22, 44, 44);
      ctx.globalAlpha = 1;
      ctx.beginPath();
      ctx.fillStyle = `rgba(255,90,70,${(a * 0.95).toFixed(3)})`;
      ctx.arc(p.sx, p.sy, 3.5, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }

  // ── Mouse / wheel interaction ──────────────────────────────────────────
  let dragging = false, lastX = 0, lastY = 0;
  const touch  = t => { lastActT = t; };

  const onDown  = e => {
    dragging = true; lastX = e.clientX; lastY = e.clientY; e.preventDefault();
  };
  const onUp    = () => { dragging = false; };
  const onMove  = e => {
    if (!dragging) return;
    const t = (performance.now() - t0) / 1000;
    touch(t);
    theta -= (e.clientX - lastX) * 0.006;
    phi    = Math.max(PHI_MIN, Math.min(PHI_MAX, phi + (e.clientY - lastY) * 0.006));
    lastX = e.clientX; lastY = e.clientY;
  };
  const onWheel = e => {
    e.preventDefault();
    const t = (performance.now() - t0) / 1000;
    touch(t);
    zoomFactor = Math.max(Z_MIN, Math.min(Z_MAX, zoomFactor * Math.exp(-e.deltaY * 0.001)));
  };
  canvas.addEventListener('mousedown', onDown);
  window.addEventListener('mouseup',   onUp);
  window.addEventListener('mousemove', onMove);
  canvas.addEventListener('wheel',     onWheel, { passive: false });

  // ── Animation loop ─────────────────────────────────────────────────────
  let rafId;
  const t0 = performance.now();
  let prevT = 0;
  function animate() {
    rafId = requestAnimationFrame(animate);
    const t  = (performance.now() - t0) / 1000;
    const dt = Math.min(t - prevT, 0.1); prevT = t; // clamp dt to avoid jump on tab-switch

    // Compute how much spin to apply this frame
    const idle = t - lastActT;
    if (idle < IDLE_DELAY) {
      spinBlend = 0;
    } else {
      spinBlend = Math.min(1, (idle - IDLE_DELAY) / BLEND_TIME);
    }

    // Randomly change rotation axis while actively spinning
    if (spinBlend > 0 && t > nextAxisAt) pickAxis(t);

    if (spinBlend > 0) {
      theta += spinVTheta * dt * spinBlend;
      phi   += spinVPhi   * dt * spinBlend;
      phi    = Math.max(PHI_MIN, Math.min(PHI_MAX, phi));
      // Bounce phi velocity at the poles so it never gets stuck
      if (phi <= PHI_MIN || phi >= PHI_MAX) spinVPhi *= -1;
    }

    drawBackground();
    drawAxes();
    if (sscd.length) drawCloud(sscd, 232, 147,  32);
    if (dino.length) drawCloud(dino,  74, 158, 224);

    if (t >= nextTraverserT) spawnTraverser(t);
    drawSparks(t);
    drawTraversers(t);
  }
  animate();

  // ── Cleanup ────────────────────────────────────────────────────────────
  canvas._stopViz = () => {
    cancelAnimationFrame(rafId);
    canvas.removeEventListener('mousedown', onDown);
    window.removeEventListener('mouseup',   onUp);
    window.removeEventListener('mousemove', onMove);
    canvas.removeEventListener('wheel',     onWheel);
    ro.disconnect();
  };
}
