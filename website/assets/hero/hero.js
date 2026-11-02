/* Aerolume home page: the live flow behind the hero.

   The background is a small but real 2D incompressible flow past a heated, slowly pitching
   airfoil, drawn the way a flow lab records one: tracer particles photographed as short streaks
   (particle streak velocimetry: each tracer exposes a line along its own velocity), lit by the
   schlieren signal of the temperature they sit in. The drawing was reworked in website round 3,
   when the first version's evenly spaced glowing dots read as generic decoration:
   - sampling is irregular, as seeded tracers are: smoke-wire rake positions are jittered and each
     wire carries its own density, and scattered tracers respawn on a jittered grid (blue-noise-
     like, never clumped and never a lattice);
   - a tracer's mark follows the flow: a streak along its velocity, as long as the distance it
     travels in one exposure (fast flow draws long faint streaks, slow flow short dense ones,
     exactly as in a streak photograph), instead of a square dot that shimmers in place;
   - size and brightness follow the measured signal (wider and darker where the temperature
     gradient is strong), scattered tracers show only where there is a signal at all, and the
     rake's own brightness is kept low so the schlieren-lit wake carries the picture;
   - no glow on each dot: one faint, wide halation over the whole field, plus a fixed sensor-like
     grain, and a plain linear greyscale between the page's own background and ink tokens.
   Website round 5 (critique of round 4: in the free stream, about 80 % of the hero, the smoke
   wires' streaks overlapped into static ruled lines, so the dot/LiDAR character was gone there):
   - only the rake tracers released in a short per-wire window are lit (puffs()), about one per
     38 CSS px of travel, so a wire reads as separate dashes that visibly travel downstream (frame
     to frame change per lit pixel in the free stream about 2.8x round 4's, from live frames 200 ms
     apart); a quarter of the scattered tracers keep a faint floor so the free stream is sampled;
   - the timelines' off phase drops to 0.15 of the on phase, so they march visibly;
   - light theme: narrower streaks with a tighter edge (no soft pencil smudges);
   - grain at 3/255 in an interleaved-gradient (blue-noise-like) pattern, one-sided on the dark
     theme's pure black so it is not clipped away.

   Physics (deliberately small, but real):
   - incompressible 2D Navier-Stokes without explicit viscosity (the numerical diffusion of the
     advection plays that role), MacCormack advection with a min/max limiter so vortices
     survive the coarse grid, light vorticity confinement, Jacobi pressure projection; inflow on
     the left, p = 0 outflow on the right, free-slip top and bottom;
   - a NACA 0012 at 17 deg +/- 4 deg (24 s period), heated to T = 1, with weak Boussinesq
     buoyancy; the inflow angle drifts by +/- 1.5 deg (31 s) so the wake never locks into one
     periodic pattern and never turns to noise; a far-field sponge keeps the picture "flow past
     a body" for as long as the page stays open;
   - tracers: RK2 in the same velocity field; rake tracers are released continuously from the
     left edge (smoke-wire streaklines) with a slow brightness pulse by release time, so
     timelines march downstream.
   Rendering: streaks are drawn additively into a half-float exposure buffer that decays fast,
   plus a quarter-resolution wide blur for the halation; the composite maps the exposure between
   --field-bg and --field-ink, so the field follows the theme switch, and damps it behind the
   copy and the navigation so the text stays the focus.

   Behaviour on a public page:
   - prefers-reduced-motion shows one still frame; the pause button stops and starts the motion
     (remembered per browser); the loop stops while the tab is hidden or the hero is off-screen;
   - start-up waits for the page to load, and the shaders compile in parallel where the browser
     offers KHR_parallel_shader_compile, so loading the page never waits on the hero; the canvas
     is absolutely placed over the background colour and fades in, so nothing moves when it
     starts; the warm-up (a formed wake before the first frame) is spread over frames;
   - cost is bounded: the sim grid holds ~20,000 cells whatever the screen, the canvas at most
     2.6 million pixels (device-pixel ratio capped at 2), frames are paced to at most 60 a
     second whatever the display's refresh rate (minFrameMs; 30 on the CPU path), and a device
     whose frames take longer than 21 ms drops to two thirds of the resolution once;
   - without WebGL2 float targets, or on a software rasteriser, the same scene runs on a coarse
     CPU grid drawn with Canvas2D at a capped 30 fps; its tracers are deposited along the path
     they travelled in the frame (the same streak look), and its calm zones get a hard ink bound
     like the WebGL composite has (see draw()); a lost GPU context is rebuilt when the browser
     restores it.
   Measured (website round 3, desktop RTX 4060 Ti / i7-13700F, headless Chrome; the GPU may have
   been shared with other jobs): text over the field never below 7.1:1 (WebGL) or 8.5:1 (Canvas2D)
   at any pixel, both themes, at 1440, 1280 and 390 px; 1.2 ms per WebGL frame median (1.6 p95),
   2.5 ms per Canvas2D frame; no long task during load (two loads); on a 144 Hz display the
   frames alternate 2 and 3 display ticks (median 14.0 ms, p95 21.0 ms) and the main thread spends
   37 ms a second, down from 64.
   Website round 4 (same machine, GPU possibly shared; pages served over local http): text over the
   field at scroll offsets 0-480 px, with the field lagging the copy, never below 6.84:1 (light,
   390 px) and 8.46:1 (dark); ?bench=240 0.5 ms per WebGL frame median (1.2 p95; the headless
   timer is coarse), 3.5 ms per Canvas2D frame; 33-35 ms a second of main-thread time with the
   hero on screen, 0.2 off-screen, 0.5 in a hidden tab; no load-time long task in 12 loads (6 with
   motion, 6 reduced-motion, where the 540-step warm-up had produced 77-104 ms tasks before the
   fence pacing).
   Website round 5 (same machine; the GPU was shared with other jobs at 25-45 % utilisation
   throughout, so absolute timings are high): text over the field at scroll offsets 0-480 px never
   below 7.32:1 (light) and 8.92:1 (dark) at 1440 and 390 px; ?bench=240 0.8 ms per WebGL frame
   median (2.3 p95), 3.2 ms per Canvas2D frame; 48-52 ms a second of main-thread time with the hero
   on screen for this build and for round 4's alike (interleaved, two runs each: the extra time
   against round 4's figure is the shared GPU, not this code), 0.3 off-screen, 0.1 in a hidden tab;
   load-time long tasks (read 4.5 s after navigation): 0 in 20 reduced-motion loads (the critique
   had seen one of 112 ms in 10). Read 9 s after navigation with the GPU 60-98 % busy, they came
   in round 4's build and this one alike, 50-300 ms, just after the load event: the hero's
   start-up, which until then could run before the first paint. Since it waits for the first
   paint (see the end of this file) the first paint is never behind it (FCP 100-136 ms against
   40-168 ms with the hero script removed, 12 loads each); a start-up task of 54-104 ms still
   occurred in 2 of 12 loads, after the paint. Moving the whole engine into a worker
   (OffscreenCanvas) would remove it; not done yet. No integrated GPU has
   been measured yet: this machine has none (i7-13700F), so the 60 fps-on-an-iGPU target is
   unverified; run ?bench=240 on one and record it here.

   Query hooks for capture and measurement only (a visitor never sees them):
     t=<s>     simulate that many seconds, then hold one frame     still    the reduced-motion frame
     theme=    force light | dark                                  notext   hide the hero text
     bench=<n> time n synchronous frames (written to data-hero-perf and the title)
     cpu       force the Canvas2D path                              allowsoft  allow software GL
     k_<name>  override one of the TUNE values below
*/
(function () {
  "use strict";
  const hero = document.querySelector("[data-hero]");
  let canvas = hero ? hero.querySelector("canvas[data-hero-field]") : null;
  if (!hero || !canvas || !window.matchMedia || !window.URLSearchParams) return;
  const copy = hero.querySelector("[data-hero-copy]");
  // The site bar sits over the hero but outside it in the document (it is the page's banner),
  // so these two are looked up in the whole page.
  const quiet = document.querySelector("[data-hero-quiet]");
  const pauseBtn = document.querySelector("[data-hero-pause]");
  const root = document.documentElement;

  const Q = new URLSearchParams(location.search);
  if (Q.get("theme") === "dark" || Q.get("theme") === "light") root.setAttribute("data-theme", Q.get("theme"));
  if (Q.has("notext")) root.classList.add("hero-notext");
  const mqReduce = matchMedia("(prefers-reduced-motion: reduce)");
  const startT = Math.max(0, parseFloat(Q.get("t") || "0") || 0);
  const benchFrames = parseInt(Q.get("bench") || "0", 10) || 0;
  const forceCpu = Q.has("cpu");
  const capture = startT > 0 || benchFrames > 0;
  const PAUSE_KEY = "aerolume-hero-paused";
  let userPaused = false;
  try { userPaused = window.localStorage.getItem(PAUSE_KEY) === "1"; } catch (e) { /* storage blocked */ }
  const frozen = () => mqReduce.matches || Q.has("still") || capture;
  const still = () => frozen() || userPaused;

  const DEG = Math.PI / 180;
  /* Look and feel, all relative to the scene's own scales (U/c, U^2/c), so the picture does not
     change with the grid. conf: vorticity confinement; at 1.0 (0.35 x 6 U/c) the field went
     unstable after ~400 s, 0.4 measured stable past 600 s. maxDark/maxLight: the field's
     strongest ink, as a fraction of the way from background to ink; calm: how much of it is
     taken away behind the copy; bloomDark/bloomLight: halation strength (one wide, faint blur,
     not a glow per dot); gain: exposure per streak; expo: exposure time of a streak in seconds;
     rakeDark/rakeLight: brightness of the smoke-wire streaklines where there is no schlieren
     signal (kept low, so the wake carries the picture); grain: fixed sensor grain, in 1/255;
     decay: what is left of the previous exposure after one 60 Hz frame; norm: how far a streak's
     brightness falls with its length (1 would be film's full 1/length, 0 none); wMin/wMax: the
     half-width of a streak in device px, from no signal to a strong one (wMaxLight: the cap in
     the light theme, where dark ink on paper reads wider than light on black, and a wide soft
     mark looked like a pencil smudge); edgeDark/edgeLight: the anti-aliasing ramp of a streak's
     edge in device px (tighter in light for the same reason); puff: the spacing of the lit
     tracers along a smoke wire in CSS px of free-stream travel, about 1.5-2x a streak's length,
     so a wire reads as separate tracers and not a ruled line, and puffGain how much brighter a
     lit tracer is drawn (see puffs() below); pulseLo: a wire's
     brightness between its timelines, as a fraction of the timeline's (low, so the timelines
     visibly march downstream); dust/dustShare: the faint floor of the scattered tracers outside
     the wake, and the share of them that carry it, so the free stream looks sampled. Chosen from
     side-by-side 1:1 crops against the first version (website round 3) and against round 4's
     ruled far field (website round 5). */
  const TUNE = { conf: 0.4, maxDark: 0.62, maxLight: 0.56, calm: 0.62, bloomDark: 0.22, bloomLight: 0.08,
    bodyDark: 0.07, bodyLight: 0.06, gain: 0.6, expo: 0.18, rakeDark: 0.32, rakeLight: 0.3, grain: 3, decay: 0.5,
    norm: 0.2, wMin: 0.45, wMax: 0.9, wMaxLight: 0.65, edgeDark: 0.5, edgeLight: 0.35, puff: 38, puffGain: 2.4, pulseLo: 0.15,
    dust: 0.12, dustShare: 0.25 };
  for (const k of Object.keys(TUNE)) if (Q.has("k_" + k)) TUNE[k] = parseFloat(Q.get("k_" + k));
  let quality = 1;   // render-resolution factor; lowered once if the device cannot keep up
  const RAKE = 0.4;  // share of the tracers that belong to the smoke-wire rake

  // Shared scene description (both back ends read it).
  const scene = {
    alpha0: 17 * DEG, alphaAmp: 4 * DEG, alphaPeriod: 24,
    inflowAmp: 1.5 * DEG, inflowPeriod: 31,
    thick: 0.12,
  };

  function css(name) { return getComputedStyle(root).getPropertyValue(name).trim(); }
  function rgb(hex) {
    const m = /^#?([0-9a-f]{6})$/i.exec(hex);
    if (!m) return null;
    const n = parseInt(m[1], 16);
    return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
  }
  function themeColors() {
    return { paper: rgb(css("--field-bg")) || rgb(css("--bg")) || [0.96, 0.96, 0.96],
      ink: rgb(css("--field-ink")) || rgb(css("--fg")) || [0.08, 0.08, 0.08] };
  }
  const isDark = (c) => c.paper[0] + c.paper[1] + c.paper[2] < 1.5;

  /* Smoke-wire positions. A real rake is never perfectly even and its wires do not all carry the
     same amount of smoke: each wire gets a fixed offset (up to 0.3 of the spacing) and a density
     between 0.45 and 1. The same hash runs in GLSL (lineHash) so respawns land on the same wires. */
  const lineHash = (k) => { const s = Math.sin(k * 12.9898 + 4.1) * 43758.5453; return s - Math.floor(s); };
  const lineY = (k, n) => (k + 0.5 + 0.6 * (lineHash(k * 1.7 + 0.3) - 0.5)) / n;

  /* Layout-dependent sizing. The grid holds ~20k square cells whatever the aspect, so the cost
     is fixed. The page puts the copy at the bottom left, so on landscape screens the airfoil
     sits right of the headline (the wake runs away from it) and on portrait screens above it. */
  function layout() {
    const r = hero.getBoundingClientRect();
    const w = Math.max(1, r.width), h = Math.max(1, r.height);
    const aspect = w / h;
    const nx = Math.max(48, Math.round(Math.sqrt(20000 * aspect)));
    const ny = Math.max(48, Math.round(nx / aspect));
    const portrait = aspect < 0.9;
    const minN = Math.min(nx, ny);
    const chord = (portrait ? 0.36 : 0.34) * minN;
    const body = portrait ? [0.30 * nx, 0.72 * ny] : [0.585 * nx, 0.54 * ny];
    const U = chord / 3.0;           // cells per second: one chord every 3 s, calm
    const cssArea = w * h;
    // Fewer tracers than the first version (about one per 70 CSS px^2 of hero): each one now
    // draws a streak, and a sparse field reads as seeded tracers rather than a texture.
    const particles = Math.min(20480, Math.max(4096, Math.round(cssArea / 70 / 256) * 256));
    const lines = Math.round(Math.min(32, Math.max(14, h / 34)));
    const dpr = quality * Math.min(window.devicePixelRatio || 1, 2, Math.sqrt(2.6e6 / cssArea));
    return { w, h, nx, ny, chord, body, U, particles, lines, dpr, portrait };
  }

  /* Calm zones in canvas pixels (y up, as GL sees it). Measured with Ranges so a zone hugs the
     glyphs, not the padded block boxes. An element that is missing gives a zone off-canvas.
     Measured against the canvas's own box, transform included: as the page scrolls, site.css
     lets the field lag behind the copy (depth), so the copy moves across the canvas and its calm
     zone has to move with it (re-measured whenever the scroll position changes, see track()). */
  function textRect(el, pw, ph) {
    if (!el) return [-1e5, -1e5, -1e5 + 1, -1e5 + 1];
    const hr = canvas.getBoundingClientRect();
    const rg = document.createRange(); rg.selectNodeContents(el);
    const b = rg.getBoundingClientRect();
    const sx = pw / hr.width, sy = ph / hr.height;
    return [(b.left - hr.left) * sx, ph - (b.bottom - hr.top) * sy, (b.right - hr.left) * sx, ph - (b.top - hr.top) * sy];
  }
  const calmRect = (pw, ph) => textRect(copy, pw, ph);
  const calmRect2 = (pw, ph) => textRect(quiet, pw, ph);

  /* Which rake tracers are lit. A smoke wire releases tracers continuously, a few CSS px apart
     along the wire; drawn as streaks 10-26 px long they overlapped into solid ruled lines (website
     round 4 critique: about 80 % of the hero read as flow-lines wallpaper, with no visible motion).
     A tracer is lit only if it was released inside a short window of a per-wire release cycle:
     the cycle is TUNE.puff CSS px of free-stream travel, and the window is sized so it holds about
     one tracer on average (Poisson, so some cycles are empty and a few hold two: irregular, like
     real seeding, never a lattice). Each wire gets its own phase, so the lit tracers never line
     up across wires. A lit tracer is drawn brighter (TUNE.puffGain, never more than the light
     the unlit ones gave up), since one separate streak no longer adds onto its overlapping
     neighbours. Returns [cycle in s, lit share of the cycle, brightness gain]. */
  function puffs(lay, rakeCount) {
    const pxPerS = lay.U * lay.w / lay.nx;                     // free stream, CSS px per second
    const period = TUNE.puff / Math.max(pxPerS, 1e-3);
    const perWire = rakeCount / lay.lines / (lay.nx / lay.U);  // tracers released per second per wire
    const duty = Math.min(1, 1.15 / Math.max(perWire * period, 1e-3));
    return [period, duty, Math.min(TUNE.puffGain, 1 / duty)];
  }

  function alphaAt(t) { return scene.alpha0 + scene.alphaAmp * Math.sin(2 * Math.PI * t / scene.alphaPeriod); }
  function inflowAt(t) { return scene.inflowAmp * Math.sin(2 * Math.PI * t / scene.inflowPeriod + 0.7); }

  /* ===================================================================== WebGL2 back end
     startGL() only creates the context and starts compiling; compiled() says when the driver has
     finished (always true without KHR_parallel_shader_compile), and finish() checks the result
     and reads the uniforms. Asking for a link status before then would block the main thread
     until every program is built (measured: most of a 140-480 ms load-time task). */
  function startGL() {
    const gl = canvas.getContext("webgl2", {
      alpha: false, antialias: false, depth: false, stencil: false,
      premultipliedAlpha: false, preserveDrawingBuffer: false, powerPreference: "default",
    });
    if (!gl) return null;
    if (!gl.getExtension("EXT_color_buffer_float")) return null;
    // A software rasteriser (GPU blocklisted, remote desktop, VM) runs this at ~50 ms a frame on
    // a full-HD canvas, where the Canvas2D path costs ~2 ms; hand over to it.
    const dbg = gl.getExtension("WEBGL_debug_renderer_info");
    const rend = dbg ? String(gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL)) : "";
    if (/swiftshader|llvmpipe|softpipe|software|basic render/i.test(rend) && !Q.has("allowsoft")) {
      const lose = gl.getExtension("WEBGL_lose_context");
      if (lose) lose.loseContext();
      return null;
    }
    const parallel = gl.getExtension("KHR_parallel_shader_compile");
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);

    const HEAD = "#version 300 es\nprecision highp float;\nprecision highp int;\nprecision highp sampler2D;\n";
    const FULL_VS = HEAD + "out vec2 vUv;void main(){vec2 p=vec2(float((gl_VertexID<<1)&2),float(gl_VertexID&2));vUv=p;gl_Position=vec4(p*2.0-1.0,0.0,1.0);}";
    // Airfoil: signed distance (in cells) to a NACA 00xx at angle uAoa about its quarter chord.
    const BODY = `
uniform vec4 uBody; // cx, cy, chord (cells), aoa (rad)
uniform float uThick;
float yt(float x){x=clamp(x,0.0,1.0);return 5.0*uThick*(0.2969*sqrt(x)-0.1260*x-0.3516*x*x+0.2843*x*x*x-0.1036*x*x*x*x);}
float bodySd(vec2 c){
  vec2 d=c-uBody.xy; float ca=cos(uBody.w), sa=sin(uBody.w);
  vec2 q=vec2(ca*d.x-sa*d.y, sa*d.x+ca*d.y)/uBody.z+vec2(0.25,0.0);
  float th=max(yt(q.x),0.7/uBody.z);
  if(q.x<0.0) return (length(q)-0.0)*uBody.z;
  if(q.x>1.0) return (length(q-vec2(1.0,0.0))-0.7/uBody.z)*uBody.z;
  return (abs(q.y)-th)*uBody.z;
}`;
    const LINEHASH = "float lineHash(float k){return fract(sin(k*12.9898+4.1)*43758.5453);}";
    const all = [];
    function compile(type, src) {
      const s = gl.createShader(type);
      gl.shaderSource(s, src); gl.compileShader(s);
      return s;
    }
    function program(fs, vs) {
      const p = gl.createProgram();
      const v = compile(gl.VERTEX_SHADER, vs || FULL_VS), f = compile(gl.FRAGMENT_SHADER, HEAD + fs);
      gl.attachShader(p, v); gl.attachShader(p, f);
      gl.linkProgram(p);
      const prog = { p, u: null, v, f };
      all.push(prog);
      return prog;
    }
    function finishProgram(prog) {
      const p = prog.p;
      if (!gl.getProgramParameter(p, gl.LINK_STATUS) && !gl.isContextLost()) {
        throw new Error((gl.getShaderInfoLog(prog.v) || "") + (gl.getShaderInfoLog(prog.f) || "") + gl.getProgramInfoLog(p));
      }
      const u = {};
      const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS) || 0;
      for (let i = 0; i < n; i++) { const a = gl.getActiveUniform(p, i); u[a.name] = gl.getUniformLocation(p, a.name); }
      prog.u = u;
    }

    const P = {
      // Semi-Lagrangian predictor: q at the foot of the characteristic.
      advect: program(`in vec2 vUv;out vec4 o;uniform sampler2D uQ,uVel;uniform vec2 uGrid;uniform float uDt;
void main(){vec2 u=texture(uVel,vUv).xy;o=texture(uQ,vUv-uDt*u/uGrid);}`),
      // MacCormack corrector + limiter + forces + boundary conditions (kind 0 velocity, 1 temperature).
      correct: program(`in vec2 vUv;out vec4 o;uniform sampler2D uQ,uHat,uVel,uCurl,uTemp;uniform vec2 uGrid;
uniform float uDt,uKind,uConf,uBuoy,uDecay,uU,uInflow,uT;${BODY}
void main(){
  vec2 c=vUv*uGrid; vec2 u=texture(uVel,vUv).xy;
  vec2 back=vUv-uDt*u/uGrid;
  vec4 hat=texture(uHat,vUv);
  vec4 fwd=texture(uHat,vUv+uDt*u/uGrid);
  vec4 r=hat+0.5*(texture(uQ,vUv)-fwd);
  ivec2 b=ivec2(floor(back*uGrid-0.5)); ivec2 m=ivec2(uGrid)-1;
  vec4 a0=texelFetch(uQ,clamp(b,ivec2(0),m),0),a1=texelFetch(uQ,clamp(b+ivec2(1,0),ivec2(0),m),0);
  vec4 a2=texelFetch(uQ,clamp(b+ivec2(0,1),ivec2(0),m),0),a3=texelFetch(uQ,clamp(b+ivec2(1,1),ivec2(0),m),0);
  r=clamp(r,min(min(a0,a1),min(a2,a3)),max(max(a0,a1),max(a2,a3)));
  float sd=bodySd(c);
  if(uKind<0.5){
    ivec2 ij=ivec2(c);
    float wl=texelFetch(uCurl,clamp(ij-ivec2(1,0),ivec2(0),m),0).r, wr=texelFetch(uCurl,clamp(ij+ivec2(1,0),ivec2(0),m),0).r;
    float wb=texelFetch(uCurl,clamp(ij-ivec2(0,1),ivec2(0),m),0).r, wt=texelFetch(uCurl,clamp(ij+ivec2(0,1),ivec2(0),m),0).r;
    float w=texelFetch(uCurl,ij,0).r;
    vec2 g=vec2(abs(wr)-abs(wl),abs(wt)-abs(wb)); g/=length(g)+1e-4;
    vec2 f=uConf*w*vec2(g.y,-g.x)*smoothstep(0.08,0.25,abs(w)*uBody.z/uU);
    f.y+=uBuoy*texture(uTemp,vUv).r;
    // Far-field sponge: the box has walls and a finite pressure solve, so without this the
    // heated wake slowly spins up box-sized gyres (measured: chaos by t = 30 s). Relaxing
    // toward the freestream near the inflow/top/bottom edges, and very weakly everywhere,
    // keeps the picture "flow past a body" indefinitely.
    float edge=min(min(c.x,c.y),uGrid.y-c.y);
    float k=0.02+1.6*(1.0-smoothstep(0.0,10.0,edge));
    f+=k*(uU*vec2(cos(uInflow),sin(uInflow))-r.xy);
    r.xy+=uDt*f;
    if(c.y<1.0||c.y>uGrid.y-1.0) r.y=0.0;
    if(c.x<1.5){ float wob=0.004*uU*sin(uT*1.3+c.y*0.21); r.xy=uU*vec2(cos(uInflow),sin(uInflow))+vec2(0.0,wob);}
    float sp=length(r.xy); if(sp>2.5*uU) r.xy*=2.5*uU/sp;   // hard bound: a slow instability can never blow up the page
    if(sd<0.0) r.xy=vec2(0.0);
  } else {
    r*=uDecay;
    if(c.x<1.5) r=vec4(0.0);
    if(sd<0.0) r=vec4(1.0);
  }
  o=r;}`),
      curl: program(`in vec2 vUv;out vec4 o;uniform sampler2D uVel;uniform vec2 uGrid;
void main(){ivec2 ij=ivec2(vUv*uGrid);ivec2 m=ivec2(uGrid)-1;
float l=texelFetch(uVel,clamp(ij-ivec2(1,0),ivec2(0),m),0).y,r=texelFetch(uVel,clamp(ij+ivec2(1,0),ivec2(0),m),0).y;
float b=texelFetch(uVel,clamp(ij-ivec2(0,1),ivec2(0),m),0).x,t=texelFetch(uVel,clamp(ij+ivec2(0,1),ivec2(0),m),0).x;
o=vec4(0.5*(r-l-t+b));}`),
      div: program(`in vec2 vUv;out vec4 o;uniform sampler2D uVel;uniform vec2 uGrid;
void main(){ivec2 ij=ivec2(vUv*uGrid);ivec2 m=ivec2(uGrid)-1;
float l=texelFetch(uVel,clamp(ij-ivec2(1,0),ivec2(0),m),0).x,r=texelFetch(uVel,clamp(ij+ivec2(1,0),ivec2(0),m),0).x;
float b=texelFetch(uVel,clamp(ij-ivec2(0,1),ivec2(0),m),0).y,t=texelFetch(uVel,clamp(ij+ivec2(0,1),ivec2(0),m),0).y;
if(ij.y==0)b=-texelFetch(uVel,ij,0).y; if(ij.y==m.y)t=-texelFetch(uVel,ij,0).y;
o=vec4(0.5*(r-l+t-b));}`),
      jacobi: program(`in vec2 vUv;out vec4 o;uniform sampler2D uP,uDiv;uniform vec2 uGrid;
void main(){ivec2 ij=ivec2(vUv*uGrid);ivec2 m=ivec2(uGrid)-1;
if(ij.x>=m.x){o=vec4(0.0);return;}
float l=texelFetch(uP,clamp(ij-ivec2(1,0),ivec2(0),m),0).r,r=texelFetch(uP,clamp(ij+ivec2(1,0),ivec2(0),m),0).r;
float b=texelFetch(uP,clamp(ij-ivec2(0,1),ivec2(0),m),0).r,t=texelFetch(uP,clamp(ij+ivec2(0,1),ivec2(0),m),0).r;
o=vec4(0.25*(l+r+b+t-texelFetch(uDiv,ij,0).r));}`),
      project: program(`in vec2 vUv;out vec4 o;uniform sampler2D uP,uVel;uniform vec2 uGrid;uniform float uU,uInflow;${BODY}
void main(){vec2 c=vUv*uGrid;ivec2 ij=ivec2(c);ivec2 m=ivec2(uGrid)-1;
float l=texelFetch(uP,clamp(ij-ivec2(1,0),ivec2(0),m),0).r,r=texelFetch(uP,clamp(ij+ivec2(1,0),ivec2(0),m),0).r;
float b=texelFetch(uP,clamp(ij-ivec2(0,1),ivec2(0),m),0).r,t=texelFetch(uP,clamp(ij+ivec2(0,1),ivec2(0),m),0).r;
vec2 u=texelFetch(uVel,ij,0).xy-0.5*vec2(r-l,t-b);
if(ij.y==0||ij.y==m.y)u.y=0.0;
if(c.x<1.5)u=uU*vec2(cos(uInflow),sin(uInflow));
if(bodySd(c)<0.0)u=vec2(0.0);
o=vec4(u,0.0,1.0);}`),
      // Schlieren: R = knife-edge (vertical gradient, signed), G = gradient magnitude.
      schlieren: program(`in vec2 vUv;out vec4 o;uniform sampler2D uTemp;uniform vec2 uGrid;
void main(){ivec2 ij=ivec2(vUv*uGrid);ivec2 m=ivec2(uGrid)-1;
float l=texelFetch(uTemp,clamp(ij-ivec2(1,0),ivec2(0),m),0).r,r=texelFetch(uTemp,clamp(ij+ivec2(1,0),ivec2(0),m),0).r;
float b=texelFetch(uTemp,clamp(ij-ivec2(0,1),ivec2(0),m),0).r,t=texelFetch(uTemp,clamp(ij+ivec2(0,1),ivec2(0),m),0).r;
vec2 g=0.5*vec2(r-l,t-b);o=vec4(g.y,length(g),0.0,1.0);}`),
      // Tracer update: RK2 in the current velocity. A rake tracer respawns on one of the jittered
      // wires (its wire number is kept in w, below uRake); a scattered tracer respawns at a random
      // point of its own cell of a jittered grid (one cell per tracer id), so fresh tracers never
      // clump the way independent uniform points do.
      particles: program(`out vec4 o;uniform sampler2D uP,uVel;uniform vec2 uGrid,uCells;uniform float uT,uDt,uLines,uRake,uU;uniform int uPW;${BODY}
${LINEHASH}
uint pcg(uint v){uint s=v*747796405u+2891336453u;uint w=((s>>((s>>28u)+4u))^s)*277803737u;return (w>>22u)^w;}
float h1(uint v){return float(pcg(v))*(1.0/4294967295.0);}
void main(){ivec2 ij=ivec2(gl_FragCoord.xy);vec4 p=texelFetch(uP,ij,0);uint id=uint(ij.y*uPW+ij.x);
  vec2 v1=texture(uVel,p.xy).xy/uGrid; vec2 v2=texture(uVel,p.xy+0.5*uDt*v1).xy/uGrid; p.xy+=uDt*v2;
  bool rake=p.w<uRake; float age=uT-p.z;
  float life=rake?90.0:mix(5.0,10.0,fract(p.w*7.31));
  bool dead=p.x>1.01||p.x<-0.05||p.y<-0.03||p.y>1.03||age>life||bodySd(p.xy*uGrid)<-0.2;
  if(dead){uint s=pcg(id*1664525u+floatBitsToUint(uT));
    if(rake){float line=floor(h1(s)*uLines);
      p.y=(line+0.5+0.6*(lineHash(line*1.7+0.3)-0.5))/uLines+(h1(s+1u)-0.5)*0.0016;
      p.x=h1(s+2u)*uDt*uU/uGrid.x; p.w=uRake*(line+0.5)/uLines;}
    else{uint n=uint(uCells.x*uCells.y); uint cell=id%n;
      p.xy=(vec2(float(cell%uint(uCells.x)),float(cell/uint(uCells.x)))+vec2(h1(s+3u),h1(s+4u)))/uCells;}
    p.z=uT-(rake?0.0:h1(s+5u)*0.3);}
  o=p;}`),
      /* Streaks into the exposure buffer: six vertices (two triangles) per tracer, laid along its
         velocity with the head at the tracer. Half-length = half the distance travelled in one
         exposure; half-width follows the schlieren signal (wMin-wMax device px). A fast tracer
         spreads its light over a longer streak (dimmer per pixel, as on film), partly: the full
         1/length law would make the freestream vanish. */
      draw: program(`in float vB;in vec2 vL;in vec2 vH;out vec4 o;
uniform float uEdge;
void main(){float ax=max(abs(vL.x)-vH.x,0.0);float d=length(vec2(ax,vL.y));
  float a=1.0-smoothstep(vH.y-uEdge,vH.y+uEdge,d);
  float head=vH.x>0.01?clamp(0.5+0.5*vL.x/vH.x,0.0,1.0):1.0;
  o=vec4(vB*a*mix(0.6,1.0,head),0.0,0.0,1.0);}`,
      HEAD + `uniform sampler2D uP,uSch,uVel;uniform int uPW;uniform float uT,uRake,uGain,uLines,uRakeK,uDpr,uExpo,uNorm,uWMin,uWMax,uPulseLo,uDust,uDustShare;uniform vec2 uGrid,uPx;uniform vec3 uPuff;
uniform vec4 uBody;
out float vB;out vec2 vL;out vec2 vH;
${LINEHASH}
uint pcg(uint v){uint s=v*747796405u+2891336453u;uint w=((s>>((s>>28u)+4u))^s)*277803737u;return (w>>22u)^w;}
float h1(uint v){return float(pcg(v))*(1.0/4294967295.0);}
const vec2 C[6]=vec2[6](vec2(-1.0,-1.0),vec2(1.0,-1.0),vec2(1.0,1.0),vec2(-1.0,-1.0),vec2(1.0,1.0),vec2(-1.0,1.0));
void main(){int pid=gl_VertexID/6;int cn=gl_VertexID-pid*6;
  ivec2 ij=ivec2(pid%uPW,pid/uPW);vec4 p=texelFetch(uP,ij,0);
  float age=uT-p.z; vec2 s=texture(uSch,p.xy).xy;
  float sch=clamp(abs(s.y)*7.0+s.x*5.0,0.0,1.6);
  float b;
  if(p.w<uRake){
    float wire=floor(p.w/uRake*uLines+0.01);
    float dens=0.45+0.55*lineHash(wire*3.1+1.7);
    float ph=fract(p.z/1.9);
    float pulse=uPulseLo+(1.0-uPulseLo)*smoothstep(0.0,0.08,ph)*(1.0-smoothstep(0.52,0.62,ph));
    // Lit only if released in this wire's puff window (see puffs() in the script); an unlit
    // tracer still shows where the schlieren signal is clearly present, so the wake keeps its
    // full sampling and only the free stream thins out.
    float lit=fract(p.z/uPuff.x+lineHash(wire*7.7+0.1))<uPuff.y?1.0:0.0;
    // A smoke wire is lit where the body is: brightness falls with the distance from the body
    // and some of the far wires carry no smoke at all, so away from the wake the field never
    // reads as evenly ruled lines. The distance is measured in 0.34 hero heights (one chord on a
    // landscape screen), so a tall phone hero keeps its middle wires.
    float dyc=abs(p.y*uGrid.y-uBody.y)/(0.34*uGrid.y);
    float fall=mix(0.3,1.0,exp(-0.9*dyc*dyc))*((dyc>0.9&&lineHash(wire*5.3+0.7)<0.4)?0.0:1.0);
    b=(uRakeK*uPuff.z*pulse*dens*fall*lit+0.85*sch*mix(smoothstep(0.15,0.6,sch),1.0,lit))*smoothstep(0.0,0.25,age);
  } else {
    float life=mix(5.0,10.0,fract(p.w*7.31));
    // A share of the scattered tracers keeps a faint floor, so the free stream reads as sampled
    // points drifting with the flow rather than empty paper between the wires.
    float floorB=h1(uint(pid)*2246822519u+3u)<uDustShare?uDust:0.0;
    b=max(1.15*sch*smoothstep(0.04,0.5,sch),floorB)*smoothstep(0.0,1.0,age)*(1.0-smoothstep(life-1.5,life,age));
  }
  vec2 vp=texture(uVel,p.xy).xy*uPx/uGrid; float sp=length(vp);
  vec2 d=sp>1e-3?vp/sp:vec2(1.0,0.0); vec2 n=vec2(-d.y,d.x);
  // Each tracer has its own fixed exposure (0.55-1.45), size (0.8-1.2) and scatter (0.7-1.3),
  // as the particles of a real streak photograph differ in size and in how long they stay in
  // the light sheet: tracers moving at the same speed then draw dashes of different lengths and
  // weights, not uniform confetti.
  uint id=uint(pid);
  float hw=uDpr*mix(uWMin,uWMax,smoothstep(0.1,1.2,sch))*(0.8+0.4*h1(id*2654435761u+7u));
  float hl=0.5*sp*uExpo*(0.55+0.9*h1(id*40503u+1013u));
  b*=0.7+0.6*h1(id*9176u+31u);
  vec2 q=C[cn]; vec2 loc=vec2(q.x*(hl+hw+1.0),q.y*(hw+1.0));
  vec2 pos=p.xy*uPx-d*hl+d*loc.x+n*loc.y;
  vL=loc; vH=vec2(hl,hw);
  vB=b*uGain*pow(hw/(hw+hl),uNorm);
  gl_Position=b<0.004?vec4(2.0,2.0,2.0,1.0):vec4(pos/uPx*2.0-1.0,0.0,1.0);}`),
      // Clear-by-multiply is done with blending; this pass only copies (bloom downsample).
      down: program(`in vec2 vUv;out vec4 o;uniform sampler2D uSrc;uniform vec2 uTexel;
void main(){vec2 d=uTexel;o=vec4(0.25*(texture(uSrc,vUv+vec2(-d.x,-d.y)).r+texture(uSrc,vUv+vec2(d.x,-d.y)).r+texture(uSrc,vUv+vec2(-d.x,d.y)).r+texture(uSrc,vUv+vec2(d.x,d.y)).r));}`),
      blur: program(`in vec2 vUv;out vec4 o;uniform sampler2D uSrc;uniform vec2 uDir;
void main(){float s=texture(uSrc,vUv).r*0.2270;
s+=(texture(uSrc,vUv+uDir*1.3846).r+texture(uSrc,vUv-uDir*1.3846).r)*0.3162;
s+=(texture(uSrc,vUv+uDir*3.2308).r+texture(uSrc,vUv-uDir*3.2308).r)*0.0703;o=vec4(s);}`),
      composite: program(`in vec2 vUv;out vec4 o;uniform sampler2D uTrail,uBloom;uniform vec3 uPaper,uInk;uniform vec2 uGrid,uPx;
uniform vec4 uCalm,uCalm2;uniform float uMaxC,uCalmK,uBloomK,uBodyK,uGrain,uGrainLo;${BODY}
float hash(vec2 p){return fract(sin(dot(p,vec2(12.9898,78.233)))*43758.5453);}
float boxSd(vec2 p,vec4 r){vec2 c=0.5*(r.xy+r.zw),h=0.5*(r.zw-r.xy);vec2 q=abs(p-c)-h;return length(max(q,0.0))+min(max(q.x,q.y),0.0);}
void main(){
  // Fixed grain, like a sensor's: it speckles the exposure (streaks pass through it) and adds a
  // faint +/- texture everywhere. Fixed in place, so it never shimmers. The texture is
  // interleaved gradient noise (Jimenez 2014), a blue-noise-like screen pattern with no clumps,
  // so 3/255 reads as a fine tooth rather than blotches. On the dark theme's pure-black field the
  // darker half of a +/- grain would be clipped away (round-4 critique: 0.49 of a level measured),
  // so there it is one-sided, 0 to +3/255, on purpose (uGrainLo = 0); lifting --field-bg off black
  // instead would leave a visible step where the hero meets the page.
  float g=hash(floor(gl_FragCoord.xy));
  vec2 fc=floor(gl_FragCoord.xy);
  float g2=fract(52.9829189*fract(dot(fc,vec2(0.06711056,0.00583715))));
  float tr=texture(uTrail,vUv).r*(0.8+0.4*g), bl=texture(uBloom,vUv).r;
  float I=1.0-exp(-(tr+uBloomK*bl));
  vec2 px=vUv*uPx;
  // Calm zones: the flow is damped (not removed) behind the copy and behind the navigation.
  float z1=1.0-smoothstep(0.0,0.09*uPx.y,boxSd(px,uCalm));
  float z2=1.0-smoothstep(0.0,0.03*uPx.y,boxSd(px,uCalm2));
  float calm=1.0-uCalmK*max(z1,z2);
  float cellPx=uPx.x/uGrid.x;
  float sd=bodySd(vUv*uGrid);
  float body=1.0-smoothstep(-0.6/cellPx,0.6/cellPx,sd);
  float k=clamp(I*uMaxC*calm+body*uBodyK,0.0,1.0);
  vec3 col=mix(uPaper,uInk,k)+(g2-uGrainLo)*uGrain;
  o=vec4(col,1.0);}`),
    };

    function tex(w, h, ifmt, fmt, type, filter, data) {
      const t = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texImage2D(gl.TEXTURE_2D, 0, ifmt, w, h, 0, fmt, type, data || null);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      const fb = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
      if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE && !gl.isContextLost()) throw new Error("fbo incomplete");
      return { t, fb, w, h };
    }
    function pair(w, h, ifmt, fmt, type, filter, data) {
      return { a: tex(w, h, ifmt, fmt, type, filter, data), b: tex(w, h, ifmt, fmt, type, filter, data),
        swap() { const x = this.a; this.a = this.b; this.b = x; } };
    }
    function free(o) { if (!o) return; gl.deleteTexture(o.t); gl.deleteFramebuffer(o.fb); }

    let L, S = null, R = null, T = 0, colors = themeColors();
    const LIN = gl.LINEAR, NEAR = gl.NEAREST;

    function buildSim() {
      if (S) { for (const k of ["vel", "tmp", "pr", "parts"]) { free(S[k].a); free(S[k].b); } for (const k of ["hat", "hatT", "div", "curl", "sch"]) free(S[k]); }
      const { nx, ny } = L;
      const v0 = new Float32Array(nx * ny * 2);
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
        // Freestream plus one weak vortex behind the body: breaks the symmetry so shedding
        // starts within a few seconds instead of waiting for round-off.
        const dx = i - (L.body[0] + 0.8 * L.chord), dy = j - (L.body[1] - 0.15 * L.chord);
        const r2 = dx * dx + dy * dy, s = 0.6 * L.U * Math.exp(-r2 / (0.08 * L.chord * L.chord));
        v0[2 * (j * nx + i)] = L.U - s * dy / L.chord * 3;
        v0[2 * (j * nx + i) + 1] = s * dx / L.chord * 3;
      }
      const PW = 256, PH = L.particles / PW;
      const p0 = new Float32Array(PW * PH * 4);
      let seed = 12345;
      const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
      for (let k = 0; k < PW * PH; k++) {
        const kind = rnd();
        if (kind < RAKE) {
          const line = Math.floor(rnd() * L.lines);
          p0[4 * k] = rnd(); p0[4 * k + 1] = lineY(line, L.lines); p0[4 * k + 2] = -p0[4 * k] * nx / L.U;
          p0[4 * k + 3] = RAKE * (line + 0.5) / L.lines;
        } else {
          p0[4 * k] = rnd(); p0[4 * k + 1] = rnd(); p0[4 * k + 2] = -rnd() * 6;
          p0[4 * k + 3] = kind;
        }
      }
      // One jittered-grid cell per scattered tracer, shaped like the hero.
      const nDust = Math.max(1, Math.round(PW * PH * (1 - RAKE)));
      const cx = Math.max(1, Math.round(Math.sqrt(nDust * L.w / L.h)));
      S = {
        vel: pair(nx, ny, gl.RG16F, gl.RG, gl.FLOAT, LIN, v0),
        tmp: pair(nx, ny, gl.R16F, gl.RED, gl.FLOAT, LIN),
        pr: pair(nx, ny, gl.R16F, gl.RED, gl.FLOAT, NEAR),
        hat: tex(nx, ny, gl.RG16F, gl.RG, gl.FLOAT, LIN),
        hatT: tex(nx, ny, gl.R16F, gl.RED, gl.FLOAT, LIN),
        div: tex(nx, ny, gl.R16F, gl.RED, gl.FLOAT, NEAR),
        curl: tex(nx, ny, gl.R16F, gl.RED, gl.FLOAT, NEAR),
        sch: tex(nx, ny, gl.RG16F, gl.RG, gl.FLOAT, LIN),
        parts: pair(PW, PH, gl.RGBA32F, gl.RGBA, gl.FLOAT, NEAR, p0),
        PW, PH, cells: [cx, Math.max(1, Math.round(nDust / cx))],
      };
      T = 0;
    }
    function buildRender() {
      if (R) { free(R.trail); free(R.b1); free(R.b2); }
      const pw = Math.max(1, Math.round(L.w * L.dpr)), ph = Math.max(1, Math.round(L.h * L.dpr));
      canvas.width = pw; canvas.height = ph;
      const bw = Math.max(1, pw >> 2), bh = Math.max(1, ph >> 2);
      R = { pw, ph, bw, bh,
        trail: tex(pw, ph, gl.R16F, gl.RED, gl.FLOAT, LIN),
        b1: tex(bw, bh, gl.R16F, gl.RED, gl.FLOAT, LIN),
        b2: tex(bw, bh, gl.R16F, gl.RED, gl.FLOAT, LIN) };
      gl.bindFramebuffer(gl.FRAMEBUFFER, R.trail.fb); gl.clearColor(0, 0, 0, 1); gl.clear(gl.COLOR_BUFFER_BIT);
      R.calm = calmRect(pw, ph); R.calm2 = calmRect2(pw, ph);
    }

    function bind(prog, target, uniforms, textures) {
      gl.useProgram(prog.p);
      if (target) { gl.bindFramebuffer(gl.FRAMEBUFFER, target.fb); gl.viewport(0, 0, target.w, target.h); }
      else { gl.bindFramebuffer(gl.FRAMEBUFFER, null); gl.viewport(0, 0, R.pw, R.ph); }
      let unit = 0;
      for (const name in textures) {
        const loc = prog.u[name];
        if (loc == null) continue;
        gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, textures[name].t);
        gl.uniform1i(loc, unit++);
      }
      for (const name in uniforms) {
        const loc = prog.u[name]; if (loc == null) continue;
        const v = uniforms[name];
        if (typeof v === "number") gl.uniform1f(loc, v);
        else if (v.int !== undefined) gl.uniform1i(loc, v.int);
        else if (v.length === 2) gl.uniform2fv(loc, v);
        else if (v.length === 3) gl.uniform3fv(loc, v);
        else if (v.length === 4) gl.uniform4fv(loc, v);
      }
    }
    function full(prog, target, u, t) { bind(prog, target, u, t); gl.drawArrays(gl.TRIANGLES, 0, 3); }

    function common() {
      return { uGrid: [L.nx, L.ny], uBody: [L.body[0], L.body[1], L.chord, alphaAt(T)], uThick: scene.thick };
    }

    function step(dt) {
      const c = common();
      const inflow = inflowAt(T);
      full(P.curl, S.curl, c, { uVel: S.vel.a });
      // velocity: predictor, corrector (+ confinement, buoyancy, BCs)
      full(P.advect, S.hat, { ...c, uDt: dt }, { uQ: S.vel.a, uVel: S.vel.a });
      full(P.correct, S.vel.b, { ...c, uDt: dt, uKind: 0, uConf: TUNE.conf * 0.35 * L.U / L.chord * 6, uBuoy: 0.015 * L.U * L.U / L.chord,
        uDecay: 1, uU: L.U, uInflow: inflow, uT: T },
        { uQ: S.vel.a, uHat: S.hat, uVel: S.vel.a, uCurl: S.curl, uTemp: S.tmp.a });
      // temperature (advected by the old velocity, same as the velocity itself)
      full(P.advect, S.hatT, { ...c, uDt: dt }, { uQ: S.tmp.a, uVel: S.vel.a });
      full(P.correct, S.tmp.b, { ...c, uDt: dt, uKind: 1, uConf: 0, uBuoy: 0, uDecay: Math.exp(-0.17 * dt), uU: L.U, uInflow: inflow, uT: T },
        { uQ: S.tmp.a, uHat: S.hatT, uVel: S.vel.a, uCurl: S.curl, uTemp: S.tmp.a });
      S.vel.swap(); S.tmp.swap();
      full(P.div, S.div, c, { uVel: S.vel.a });
      for (let i = 0; i < 28; i++) { full(P.jacobi, S.pr.b, c, { uP: S.pr.a, uDiv: S.div }); S.pr.swap(); }
      full(P.project, S.vel.b, { ...c, uU: L.U, uInflow: inflow }, { uP: S.pr.a, uVel: S.vel.a });
      S.vel.swap();
      full(P.schlieren, S.sch, c, { uTemp: S.tmp.a });
      T += dt;
      full(P.particles, S.parts.b, { ...c, uT: T, uDt: dt, uLines: L.lines, uRake: RAKE, uU: L.U, uCells: S.cells, uPW: { int: S.PW } },
        { uP: S.parts.a, uVel: S.vel.a });
      S.parts.swap();
    }

    function deposit(dt) {
      // Exposure persistence: dst *= decay (blend only, no extra buffer), then add the streaks.
      gl.bindFramebuffer(gl.FRAMEBUFFER, R.trail.fb); gl.viewport(0, 0, R.pw, R.ph);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ZERO, gl.CONSTANT_ALPHA);
      gl.blendColor(0, 0, 0, Math.pow(TUNE.decay, dt * 60));
      full(P.down, R.trail, { uTexel: [0, 0] }, { uSrc: R.b2 });   // any shader: colour is ignored by ZERO
      gl.blendFunc(gl.ONE, gl.ONE);
      const dark = isDark(colors);
      bind(P.draw, R.trail, { ...common(), uT: T, uRake: RAKE, uGain: TUNE.gain * dt * 60, uLines: L.lines, uPW: { int: S.PW },
        uRakeK: dark ? TUNE.rakeDark : TUNE.rakeLight, uDpr: Math.max(1, L.dpr), uExpo: TUNE.expo, uNorm: TUNE.norm,
        uWMin: TUNE.wMin, uWMax: dark ? TUNE.wMax : Math.min(TUNE.wMax, TUNE.wMaxLight), uEdge: dark ? TUNE.edgeDark : TUNE.edgeLight,
        uPuff: puffs(L, RAKE * S.PW * S.PH), uPulseLo: TUNE.pulseLo, uDust: TUNE.dust, uDustShare: TUNE.dustShare, uPx: [R.pw, R.ph] },
        { uP: S.parts.a, uSch: S.sch, uVel: S.vel.a });
      gl.drawArrays(gl.TRIANGLES, 0, 6 * S.PW * S.PH);
      gl.disable(gl.BLEND);
    }
    function present() {
      full(P.down, R.b1, { uTexel: [1 / R.pw, 1 / R.ph] }, { uSrc: R.trail });
      // A wide blur (about 25 device px): a halation over the field, not a glow around each mark.
      full(P.blur, R.b2, { uDir: [3.2 / R.bw, 0] }, { uSrc: R.b1 });
      full(P.blur, R.b1, { uDir: [0, 3.2 / R.bh] }, { uSrc: R.b2 });
      const dark = isDark(colors);
      full(P.composite, null, { ...common(), uPaper: colors.paper, uInk: colors.ink, uPx: [R.pw, R.ph], uCalm: R.calm, uCalm2: R.calm2,
        uMaxC: dark ? TUNE.maxDark : TUNE.maxLight, uCalmK: TUNE.calm, uBloomK: dark ? TUNE.bloomDark : TUNE.bloomLight,
        uBodyK: dark ? TUNE.bodyDark : TUNE.bodyLight, uGrain: TUNE.grain / 255, uGrainLo: dark ? 0 : 0.5 },
        { uTrail: R.trail, uBloom: R.b1 });
    }
    function frame(dt) { step(dt); deposit(dt); }

    // Advance n fixed steps; the last `drawn` of them also deposit tracers into the trail.
    function warmSteps(n, drawn) { const dt = 1 / 60; for (let i = 0; i < n; i++) { step(dt); if (i >= n - drawn) deposit(dt); } }

    function init() { L = layout(); buildSim(); buildRender(); }
    return { name: "webgl2", gl, init, warmSteps, frame, present,
      compiled() { return !parallel || gl.isContextLost() || all.every((p) => gl.getProgramParameter(p.p, parallel.COMPLETION_STATUS_KHR)); },
      finish() { for (const p of all) finishProgram(p); },
      resized() {
        const old = L; L = layout();
        const reSim = Math.abs(Math.log((old.nx / old.ny) / (L.nx / L.ny))) > 0.22 || L.particles !== old.particles;
        if (reSim) buildSim();
        else { L.nx = old.nx; L.ny = old.ny; L.body = old.body; L.chord = old.chord; L.U = old.U; L.lines = old.lines; L.particles = old.particles; }
        buildRender(); return reSim;
      },
      recalm() { if (R) { R.calm = calmRect(R.pw, R.ph); R.calm2 = calmRect2(R.pw, R.ph); } },
      theme() { colors = themeColors(); },
      sync() { const px = new Uint8Array(4); gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px); },
      info() { const e = gl.getExtension("WEBGL_debug_renderer_info"); return e ? gl.getParameter(e.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER); },
      layout: () => L, time: () => T, pixels: () => (R ? R.pw * R.ph : 0),
      /* Warm-up flow control: mark() drops a fence after a chunk of steps, busy() says whether the
         GPU is still working through it. The warm-up waits for the fence before queueing more, so
         the GPU queue never fills to the point where the driver blocks a WebGL call on the main
         thread (measured: 77-104 ms load-time tasks for the 540-step reduced-motion warm-up). */
      fence: null,
      busy() {
        if (!this.fence || gl.isContextLost()) return false;
        if (gl.getSyncParameter(this.fence, gl.SYNC_STATUS) !== gl.SIGNALED) return true;
        gl.deleteSync(this.fence); this.fence = null; return false;
      },
      mark() { if (this.fence) gl.deleteSync(this.fence); this.fence = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0); gl.flush(); },
      // At most 60 frames a second, even on a 144-240 Hz display: a background decoration has no
      // use for more, and the GPU work scales with the frame rate (measured 2.4x at 144 Hz).
      minFrameMs: 1000 / 60 };
  }

  /* =================================================================== Canvas2D fallback
     Same scene on a ~4k-cell grid in plain JS (semi-Lagrangian + 16 Jacobi sweeps) with 2,400
     tracers; enough to read as the same picture where WebGL2 float targets are absent. */
  function startCPU() {
    const ctx = canvas.getContext("2d", { alpha: false });
    if (!ctx) return null;
    let L, nx, ny, u, v, u2, v2, tt, t2, p, dv, solid, px, py, ox, oy, pb, pk, pd, T = 0, colors = themeColors(), dpr = 1, calm, calm2, puff = [1, 1, 1];
    const N = 2400;
    // Ink per streak, chosen by eye from 1440 x 900 frames after 12 s in both themes, side by
    // side with the WebGL frames it stands in for.
    const CPU_INK = Q.has("k_cpuink") ? parseFloat(Q.get("k_cpuink")) : 1;
    let seed = 777;
    const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);   // repeatable stills
    function idx(i, j) { return j * nx + i; }
    function sample(f, x, y) {
      x = Math.min(nx - 1.001, Math.max(0, x - 0.5)); y = Math.min(ny - 1.001, Math.max(0, y - 0.5));
      const i = x | 0, j = y | 0, fx = x - i, fy = y - j, k = j * nx + i;
      return (f[k] * (1 - fx) + f[k + 1] * fx) * (1 - fy) + (f[k + nx] * (1 - fx) + f[k + nx + 1] * fx) * fy;
    }
    function inBody(x, y, a) {
      const dx = x - L.body[0], dy = y - L.body[1], ca = Math.cos(a), sa = Math.sin(a);
      const qx = (ca * dx - sa * dy) / L.chord + 0.25, qy = (sa * dx + ca * dy) / L.chord;
      if (qx < 0 || qx > 1) return false;
      const yt = 5 * scene.thick * (0.2969 * Math.sqrt(qx) - 0.126 * qx - 0.3516 * qx * qx + 0.2843 * qx ** 3 - 0.1036 * qx ** 4);
      return Math.abs(qy) < Math.max(yt, 0.7 / L.chord);
    }
    // Calm zones in canvas pixels, y down (the copy, and the navigation bar over the hero).
    function measureCalm() {
      const H = canvas.height, flip = (c) => [c[0], H - c[3], c[2], H - c[1]];
      calm = flip(calmRect(canvas.width, H)); calm2 = flip(calmRect2(canvas.width, H));
    }
    // A rake tracer takes a wire and that wire's smoke density; a scattered one a random point.
    function spawn(k, rake) {
      if (rake) { const line = Math.floor(rnd() * L.lines); py[k] = lineY(line, L.lines); pd[k] = 0.45 + 0.55 * lineHash(line * 3.1 + 1.7); }
      else { py[k] = rnd(); pd[k] = 1; }
    }
    function init() {
      const big = layout();
      const scale = Math.sqrt(4200 / (big.nx * big.ny));
      nx = Math.max(32, Math.round(big.nx * scale)); ny = Math.max(32, Math.round(big.ny * scale));
      L = { ...big, nx, ny, chord: big.chord * scale, body: [big.body[0] * scale, big.body[1] * scale], U: big.U * scale, particles: N };
      const n = nx * ny;
      u = new Float32Array(n).fill(L.U); v = new Float32Array(n); u2 = new Float32Array(n); v2 = new Float32Array(n);
      tt = new Float32Array(n); t2 = new Float32Array(n); p = new Float32Array(n); dv = new Float32Array(n); solid = new Uint8Array(n);
      px = new Float32Array(N); py = new Float32Array(N); pb = new Float32Array(N); pk = new Float32Array(N); pd = new Float32Array(N);
      ox = new Float32Array(N); oy = new Float32Array(N);
      for (let k = 0; k < N; k++) { pk[k] = rnd(); px[k] = rnd(); spawn(k, pk[k] < 0.7); pb[k] = -rnd() * 20; ox[k] = px[k]; oy[k] = py[k]; }
      dpr = Math.min(window.devicePixelRatio || 1, 1.5);
      canvas.width = Math.round(L.w * dpr); canvas.height = Math.round(L.h * dpr);
      const pap = colors.paper.map((c) => Math.round(c * 255)).join(",");
      ctx.fillStyle = "rgb(" + pap + ")"; ctx.fillRect(0, 0, canvas.width, canvas.height);
      measureCalm();
      puff = puffs(L, 0.7 * N);   // the same lit-tracer windows as the WebGL rake
      T = 0;
    }
    function step(dt) {
      const a = alphaAt(T), inflow = inflowAt(T);
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) solid[idx(i, j)] = inBody(i + 0.5, j + 0.5, a) ? 1 : 0;
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
        const k = idx(i, j), x = i + 0.5, y = j + 0.5;
        const bx = x - dt * u[k], by = y - dt * v[k];
        u2[k] = sample(u, bx, by); v2[k] = sample(v, bx, by) + dt * 0.015 * L.U * L.U / L.chord * tt[k] + dt * 0.02 * (L.U * Math.sin(inflow) - v[k]);
        t2[k] = sample(tt, bx, by) * Math.exp(-0.17 * dt);
        if (i < 1) { u2[k] = L.U * Math.cos(inflow); v2[k] = L.U * Math.sin(inflow) + 0.004 * L.U * Math.sin(T * 1.3 + j * 0.21); t2[k] = 0; }
        if (j === 0 || j === ny - 1) v2[k] = 0;
        if (solid[k]) { u2[k] = 0; v2[k] = 0; t2[k] = 1; }
      }
      [u, u2] = [u2, u]; [v, v2] = [v2, v]; [tt, t2] = [t2, tt];
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
        const l = u[idx(Math.max(i - 1, 0), j)], r = u[idx(Math.min(i + 1, nx - 1), j)];
        const b = v[idx(i, Math.max(j - 1, 0))], t = v[idx(i, Math.min(j + 1, ny - 1))];
        dv[idx(i, j)] = 0.5 * (r - l + t - b);
      }
      for (let it = 0; it < 16; it++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx - 1; i++) {
        const k = idx(i, j);
        p[k] = 0.25 * (p[idx(Math.max(i - 1, 0), j)] + p[idx(i + 1, j)] + p[idx(i, Math.max(j - 1, 0))] + p[idx(i, Math.min(j + 1, ny - 1))] - dv[k]);
      }
      for (let j = 0; j < ny; j++) for (let i = 1; i < nx; i++) {
        const k = idx(i, j);
        u[k] -= 0.5 * (p[idx(Math.min(i + 1, nx - 1), j)] - p[idx(i - 1, j)]);
        v[k] -= 0.5 * (p[idx(i, Math.min(j + 1, ny - 1))] - p[idx(i, Math.max(j - 1, 0))]);
        if (solid[k]) { u[k] = 0; v[k] = 0; }
      }
      T += dt;
      for (let k = 0; k < N; k++) {
        let x = px[k] * nx, y = py[k] * ny;
        ox[k] = px[k]; oy[k] = py[k];
        const ux = sample(u, x, y), vy = sample(v, x, y);
        x += dt * sample(u, x + 0.5 * dt * ux, y + 0.5 * dt * vy); y += dt * sample(v, x + 0.5 * dt * ux, y + 0.5 * dt * vy);
        px[k] = x / nx; py[k] = y / ny;
        const rake = pk[k] < 0.7, age = T - pb[k];
        if (px[k] > 1.01 || py[k] < 0 || py[k] > 1 || age > (rake ? 90 : 7) || inBody(x, y, alphaAt(T))) {
          px[k] = rake ? rnd() * 0.004 : rnd();
          spawn(k, rake);
          pb[k] = T; ox[k] = px[k]; oy[k] = py[k];     // a respawn draws no streak across the page
        }
      }
    }
    function draw(dt) {
      const W = canvas.width, H = canvas.height, pap = colors.paper, ink = colors.ink;
      const dark = isDark(colors);
      ctx.globalAlpha = 1 - Math.pow(0.75, dt * 60);
      ctx.fillStyle = "rgb(" + pap.map((c) => Math.round(c * 255)).join(",") + ")";
      ctx.fillRect(0, 0, W, H);
      const inkS = "rgb(" + ink.map((c) => Math.round(c * 255)).join(",") + ")";
      ctx.fillStyle = inkS;
      const rakeK = dark ? TUNE.rakeDark : TUNE.rakeLight;
      // Each tracer is drawn as a streak along its velocity, as long as the distance it travels
      // in one exposure (TUNE.expo, the same as the WebGL path), so the look does not depend on
      // the frame rate. Velocity comes from the step just taken.
      const back = TUNE.expo / Math.max(lastStep, 1e-3);
      ctx.strokeStyle = inkS; ctx.lineCap = "round";
      for (let k = 0; k < N; k++) {
        const x = px[k] * nx, y = py[k] * ny;
        const gy = 0.5 * (sample(tt, x, y + 1) - sample(tt, x, y - 1)), gx = 0.5 * (sample(tt, x + 1, y) - sample(tt, x - 1, y));
        const sch = Math.max(0, Math.min(1.6, Math.hypot(gx, gy) * 7 + gy * 5));
        const rake = pk[k] < 0.7, ph = ((pb[k] / 1.9) % 1 + 1) % 1, age = T - pb[k];
        // the same wire fall-off and per-tracer exposure as the WebGL draw shader
        const dyc = Math.abs(py[k] * ny - L.body[1]) / (0.34 * ny);
        const fall = (0.3 + 0.7 * Math.exp(-0.9 * dyc * dyc)) * (dyc > 0.9 && lineHash(pd[k] * 97.1) < 0.4 ? 0 : 1);
        // ... and the same puff windows, timeline contrast and faint floor on scattered tracers
        const lit = ((pb[k] / puff[0] + lineHash(pd[k] * 13.7)) % 1 + 1) % 1 < puff[1] ? 1 : 0;
        const schOn = Math.min(1, Math.max(0, (sch - 0.15) / 0.45));
        let b = rake ? rakeK * puff[2] * pd[k] * fall * (ph < 0.57 ? 1 : TUNE.pulseLo) * lit + 0.85 * sch * (lit ? 1 : schOn)
          : Math.max(1.1 * sch * Math.min(1, Math.max(0, (sch - 0.04) / 0.46)), pk[k] < 0.7 + 0.3 * TUNE.dustShare ? TUNE.dust : 0) * Math.min(1, age);
        if (b < 0.01) continue;
        const X = px[k] * W, Y = H - py[k] * H;
        if (X > calm[0] - 60 && X < calm[2] + 60 && Y > calm[1] - 60 && Y < calm[3] + 60) b *= 0.4;
        ctx.globalAlpha = Math.min(0.9, b * 0.5 * CPU_INK);
        ctx.lineWidth = dpr * (0.9 + 0.8 * Math.min(1, sch));     // the mark widens with the signal
        ctx.beginPath();
        const ef = back * (0.55 + 0.9 * lineHash(k * 0.618 + 0.1));
        ctx.moveTo(X - (px[k] - ox[k]) * W * ef, Y + (py[k] - oy[k]) * H * ef);
        ctx.lineTo(X, Y);
        ctx.stroke();
      }
      // The WebGL composite bounds the ink behind the text (at most maxC x (1 - calm), about 0.21
      // of the way to the ink). Alpha-composited dots have no such bound: deposits that repeat
      // along a streak line add up toward full ink. So the calm zones are washed toward the paper
      // after the dots, in three feathered steps: whatever was drawn, at most 0.7 x 0.55 x 0.4 =
      // 15 % of the ink survives behind the text (measured from pixels: see website/tests).
      ctx.fillStyle = "rgb(" + pap.map((c) => Math.round(c * 255)).join(",") + ")";
      for (const r of [calm, calm2]) {
        for (const [pad, al] of [[60, 0.3], [30, 0.45], [6, 0.6]]) {
          const q = pad * dpr;
          ctx.globalAlpha = al;
          ctx.fillRect(r[0] - q, r[1] - q, r[2] - r[0] + 2 * q, r[3] - r[1] + 2 * q);
        }
      }
      ctx.globalAlpha = 1;
    }
    let lastDt = 1 / 60, lastStep = 1 / 30;
    return { name: "canvas2d", init, compiled: () => true, finish() {},
      warmSteps(n, drawn) { lastStep = 1 / 30; for (let i = 0; i < n; i++) { step(1 / 30); if (i >= n - drawn) draw(1 / 30); } },
      frame(dt) { lastDt = dt; lastStep = dt; step(dt); }, present() { draw(lastDt); }, resized() { init(); return true; },
      recalm: measureCalm, theme() { colors = themeColors(); }, sync() {}, info: () => "Canvas2D (CPU)",
      layout: () => L, time: () => T, pixels: () => canvas.width * canvas.height, stepScale: 0.5,
      // 30 frames a second is plenty for this calm a picture, and halves the main-thread time
      // (measured 43 % of a core at a 143 Hz tick uncapped, on a desktop i7).
      minFrameMs: 1000 / 30 };
  }

  /* ============================================================================ driver */
  /* Pick a back end and hand it to `done` once its shaders are built. With `sync` (capture
     hooks only) everything happens before this returns. */
  function engine(done, sync) {
    let e = null;
    try { if (!forceCpu) e = startGL(); } catch (err) { console.warn("aerolume hero: WebGL2 path failed, using the CPU path", err); e = null; }
    const cpu = () => {
      // A canvas that once gave a WebGL context can never give a 2D one: use a fresh element.
      if (canvas.getContext && !forceCpu) {
        const fresh = canvas.cloneNode(false);
        canvas.replaceWith(fresh);
        canvas = fresh;
      }
      return startCPU();
    };
    if (!e) { done(cpu()); return; }
    // Poll for the parallel compile, but not forever: after about three seconds (120 polls)
    // finish() simply waits for the driver. The hero is decorative, so starting it late is
    // better than a load-time long task; a capture tool's virtual time burns through the polls
    // in a few real milliseconds, so captures are not held up for long either.
    let tries = 0;
    const poll = () => {
      if (!sync && tries++ < 120 && !e.compiled()) { setTimeout(poll, 24); return; }
      try { e.finish(); } catch (err) { console.warn("aerolume hero: WebGL2 shaders failed, using the CPU path", err); e = cpu(); }
      done(e);
    };
    poll();
  }

  let E = null;
  let visible = !document.hidden, onScreen = true, running = false, last = 0, due = 0, raf = 0, ready = false;
  const frames = [];
  function show() { canvas.classList.add("is-live"); }

  /* The calm zones follow the copy across the canvas while the page scrolls (site.css lets the
     field lag behind the copy). The running loop re-measures them in the frame when the scroll
     position has changed; a still or paused field is re-composited on scroll instead, which costs
     one composite pass (WebGL) or one redraw of the same tracers (Canvas2D). */
  let lastY = NaN, scrollQ = false;
  function track() {
    const y = window.scrollY;
    if (y === lastY) return false;
    lastY = y; E.recalm(); return true;
  }
  window.addEventListener("scroll", () => {
    if (scrollQ || !E || !ready || running || !onScreen) return;
    scrollQ = true;
    requestAnimationFrame(() => { scrollQ = false; if (E && ready && !running && track()) E.present(); });
  }, { passive: true });

  /* Frame pacing: a frame is drawn when its slot (every minFrameMs) has come, give or take half
     a short display tick, so a 144 Hz display alternates 2 and 3 ticks (about 58 fps) instead of
     falling to every third tick (48 fps). dt is the real time since the last drawn frame. */
  function loop(now) {
    raf = 0;
    if (!visible || !onScreen || still()) { running = false; return; }
    const minMs = E.minFrameMs || 0;
    if (last && now < due - 3) { raf = requestAnimationFrame(loop); return; }
    const dt = last ? Math.min((now - last) / 1000, 1 / 30) : 1 / 60;
    if (last) frames.push(now - last);
    due = Math.max((due || now) + minMs, now - minMs);
    last = now;
    E.frame(dt); track(); E.present();
    if (frames.length === 240) {
      const med = report(frames.splice(0), "raf");
      // One-step adaptive quality: a slow GPU gets fewer pixels (the sim grid is unchanged). The
      // CPU path is capped at 30 fps on purpose and has no pixel budget to trade, so it is exempt.
      if (E.name === "webgl2" && med > 21 && quality === 1) { quality = 0.66; E.resized(); }
    }
    raf = requestAnimationFrame(loop);
  }
  function run() { if (running || !ready || still() || !visible || !onScreen) return; running = true; last = 0; due = 0; raf = requestAnimationFrame(loop); }
  function report(list, kind) {
    const s = list.slice().sort((a, b) => a - b);
    const med = s[s.length >> 1], p95 = s[Math.floor(s.length * 0.95)];
    const msg = "flow " + kind + " " + E.name + " median " + med.toFixed(2) + " ms, p95 " + p95.toFixed(2) + " ms (" + E.info() + ")";
    hero.setAttribute("data-hero-perf", msg);
    return med;
  }

  /* Pause control (WCAG 2.2.2): visible whenever the motion could run. The accessible name stays
     "Pause the background animation" and aria-pressed carries the state; changing both would
     make screen readers announce "Play ..., pressed". */
  function syncPause() {
    if (!pauseBtn) return;
    pauseBtn.hidden = frozen();
    pauseBtn.setAttribute("aria-pressed", userPaused ? "true" : "false");
    pauseBtn.setAttribute("title", userPaused ? "Play the background animation" : "Pause the background animation");
  }
  if (pauseBtn) {
    pauseBtn.addEventListener("click", () => {
      userPaused = !userPaused;
      try { window.localStorage.setItem(PAUSE_KEY, userPaused ? "1" : "0"); } catch (e) { /* storage blocked */ }
      syncPause();
      if (!userPaused) run();
    });
  }
  syncPause();

  // Off-screen and hidden-tab pauses: the loop simply stops scheduling itself.
  document.addEventListener("visibilitychange", () => { visible = !document.hidden; if (visible) run(); });
  if ("IntersectionObserver" in window) {
    new IntersectionObserver((es) => { onScreen = es[es.length - 1].isIntersecting; if (onScreen) run(); }, { threshold: 0 }).observe(hero);
  }
  let rT = 0;
  window.addEventListener("resize", () => {
    if (!E || !ready) return;
    clearTimeout(rT);
    rT = setTimeout(() => {
      const re = E.resized();
      const n = Math.round((re ? 2.0 : 1.2) * 60 * (E.stepScale || 1));
      E.warmSteps(n, Math.min(n, 90)); E.present();
    }, 150);
  });
  const reTheme = () => { if (!E) return; E.theme(); if (!running && ready) E.present(); };
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", reTheme);
  new MutationObserver(reTheme).observe(root, { attributes: true, attributeFilter: ["data-theme"] });
  mqReduce.addEventListener("change", () => { syncPause(); if (!still()) run(); });
  // The calm zones follow the text: re-measure once the web fonts have replaced the fallback.
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => { if (E) { E.recalm(); if (!running && ready) E.present(); } });

  function watchContext() {
    // A lost GPU context (driver reset, too many contexts) is rebuilt when the browser gives it back.
    canvas.addEventListener("webglcontextlost", (ev) => { ev.preventDefault(); running = false; ready = false; if (raf) cancelAnimationFrame(raf); raf = 0; });
    canvas.addEventListener("webglcontextrestored", () => {
      engine((e) => {
        if (!e) return;
        E = e; E.init(); hero.setAttribute("data-hero-engine", E.name);
        E.warmSteps(Math.round(2.5 * 60 * (E.stepScale || 1)), 90); E.present(); ready = true; run();
      });
    });
  }

  function boxes() {
    const box = (el) => { if (!el) return null; const rg = document.createRange(); rg.selectNodeContents(el); const b = rg.getBoundingClientRect();
      return [Math.round(b.left), Math.round(b.top), Math.round(b.right), Math.round(b.bottom)]; };
    const q = (s) => hero.querySelector(s);
    hero.setAttribute("data-hero-boxes", JSON.stringify({ h1: box(q("h1")), lede: box(q("[data-hero-copy] p")),
      primary: box(q("[data-hero-copy] .btn-primary")), secondary: box(q("[data-hero-copy] .btn-secondary")), nav: box(quiet) }));
  }

  function begin(e) {
    E = e;
    if (!E) return;
    E.init();
    hero.setAttribute("data-hero-engine", E.name);
    watchContext();

    if (benchFrames > 0) {
      // Synchronous benchmark: each frame is simulated, drawn and then forced to finish with a
      // 1-pixel readback, so the times include the GPU work, not just command submission.
      E.warmSteps(Math.round(2 * 60 * (E.stepScale || 1)), 60);
      const list = [], cpu = [];
      for (let i = 0; i < benchFrames; i++) {
        const t0 = performance.now(); E.frame(1 / 60); E.present(); const t1 = performance.now(); E.sync();
        list.push(performance.now() - t0); cpu.push(t1 - t0);
      }
      const sorted = (a) => a.slice().sort((x, y) => x - y);
      const at = (a, q) => a[Math.min(a.length - 1, Math.floor(q * a.length))].toFixed(2);
      const sl = sorted(list), sc = sorted(cpu), Lb = E.layout();
      const msg = "bench " + E.name + " frames=" + benchFrames + " full_med=" + at(sl, 0.5) + " full_p95=" + at(sl, 0.95) +
        " js_med=" + at(sc, 0.5) + " js_p95=" + at(sc, 0.95) + " grid=" + Lb.nx + "x" + Lb.ny + " tracers=" + Lb.particles +
        " canvas=" + canvas.width + "x" + canvas.height + " dpr=" + (window.devicePixelRatio || 1) + " gl=" + E.info();
      hero.setAttribute("data-hero-perf", msg);
      document.title = msg;
      show();
      return;
    }

    if (startT > 0) {
      // A ?t= capture: simulate ahead in one go (capture tools read the frame right after load).
      E.warmSteps(Math.round(startT * 60 * (E.stepScale || 1)), 90); E.present();
      canvas.style.transition = "none"; show(); ready = true;
      boxes();
      if (document.fonts && document.fonts.ready) document.fonts.ready.then(boxes);
      return;
    }

    // Start with a formed wake (2.5 s of flow; the reduced-motion still frame shows 9 s), spread
    // over frames so loading never stalls the page, even on a slow GPU.
    let warmLeft = Math.round((frozen() ? 9 : 2.5) * 60 * (E.stepScale || 1));
    const warmTotal = warmLeft;
    // Each chunk is a separate task of at most 12 steps (WebGL; 8 on the CPU path) or about 6 ms
    // of script, whichever comes first, so no chunk becomes a long task even on a slow or busy
    // GPU (a 30-step chunk was the likely source of a 176 ms load-time task, website round 4).
    // Only the last 90 steps expose tracers, as before.
    let waits = 0;
    (function warmChunk() {
      if (E.busy && E.busy() && waits++ < 60) { setTimeout(warmChunk, 4); return; }
      waits = 0;
      const t0 = performance.now(), cap = E.name === "webgl2" ? 12 : 8;
      for (let n = 0; warmLeft > 0 && n < cap && (n === 0 || performance.now() - t0 < 6); n++) {
        E.warmSteps(1, warmLeft <= 90 ? 1 : 0);
        warmLeft--;
      }
      if (E.mark) E.mark();
      if (warmLeft > 0) { setTimeout(warmChunk, 0); return; }   // tasks, so other work interleaves
      E.present(); show(); ready = true;
      hero.setAttribute("data-hero-warm", String(warmTotal));
      run();
    })();
  }

  if (capture) engine(begin, true);
  else {
    // After the page has loaded and painted: the page's own text and layout come first, the
    // background after. "Loaded" alone is not "painted": served from disk the load event came at
    // 40-90 ms and the first paint at 100-260 ms, so starting at load put the hero's start-up task
    // (53-166 ms with the GPU busy with other work) in front of the first paint (website round 5;
    // 4 load-time long tasks in 12 loads, against 1 in 12 with the hero script removed). So it
    // waits for the first contentful paint as well, and then for a task, not an animation frame,
    // so it still runs in a background tab (which never paints: the 1 s fallback) and under
    // headless capture tools' virtual time.
    const later = () => {
      let started = false;
      const go = () => { if (!started) { started = true; setTimeout(() => engine(begin, false), 0); } };
      if (performance.getEntriesByName && performance.getEntriesByName("first-contentful-paint").length) { go(); return; }
      try {
        const po = new PerformanceObserver(() => { po.disconnect(); go(); });
        po.observe({ type: "paint", buffered: true });
      } catch (e) { /* no paint timing: the fallback starts it */ }
      setTimeout(go, 1000);
    };
    if (document.readyState === "complete") later(); else window.addEventListener("load", later, { once: true });
  }
})();
