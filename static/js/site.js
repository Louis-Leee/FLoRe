// FLoRe project page: clips that load and play only while they are watched, the nav that marks
// the section you are in, and the hero, whose teaser plays as the page is scrolled before it gives
// way to the page.  The structure follows the Workhorse page's script
// (hybridrobotics.github.io/workhorse), trimmed to what is used here; the hero follows the
// DoorMan page's (doorman-humanoid.github.io).

const reduced = matchMedia("(prefers-reduced-motion: reduce)");
const DWELL = 500; // ms a figure stays on screen before it starts, so a page jump starts nothing

// --- things rise into place the first time they are seen ---
// Set up first: until it runs, the inline script in <head> has them hidden, and it shows them
// again by itself if this never gets here.
const rise = new IntersectionObserver(
  (entries) => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      e.target.classList.add("in");
      rise.unobserve(e.target);
    }
  },
  { rootMargin: "0px 0px -12% 0px", threshold: 0.08 },
);
document.querySelectorAll(".reveal").forEach((el, i) => {
  el.style.transitionDelay = `${Math.min(i % 4, 3) * 70}ms`; // a short stagger inside a group
  rise.observe(el);
});
window.revealReady = true;

// Runs `fn` once `el` is about a screen away: posters are fetched then, ahead of the figure,
// and not as the page opens.
const nearBy = new Map();
const near = new IntersectionObserver(
  (es) => {
    for (const e of es) {
      if (!e.isIntersecting) continue;
      near.unobserve(e.target);
      nearBy.get(e.target)();
      nearBy.delete(e.target);
    }
  },
  { rootMargin: "1000px 0px" },
);
const whenNear = (el, fn) => {
  nearBy.set(el, fn);
  near.observe(el);
};

// --- clips ---
// A clip plays once a quarter of it has stayed on screen for DWELL, pauses as it leaves, and lets
// go of its download once it is out of sight, so the clips in view have the line to themselves.
// A clip the visitor paused stays paused, and under reduced motion nothing starts on its own.

document.querySelectorAll("video[data-poster]").forEach((v) =>
  whenNear(v, () => {
    v.poster = v.dataset.poster;
    delete v.dataset.poster;
    if (v.controls && !v.hasAttribute("data-autoplay")) v.preload = "metadata"; // shows its length
  }),
);

// The hero's clips keep their addresses in data-src and fetch nothing before it is set.
const attach = (v) => {
  if (!v.dataset.src) return;
  v.src = v.dataset.src;
  delete v.dataset.src;
};

// picture in picture, full screen, or the iPhone's own player: watched, though not on the page
const watchedElsewhere = (v) =>
  document.pictureInPictureElement === v ||
  document.fullscreenElement === v ||
  v.webkitDisplayingFullscreen === true;

// Stops a video's download and frees its decoder.  Its time is kept aside and set again as it
// next plays, whichever control starts it: a time set on an emptied video makes the browser
// fetch it straight back.
const resumeAt = new WeakMap();
const release = (v) => {
  if (v.readyState === 0 && v.networkState !== v.NETWORK_LOADING) return; // holds nothing
  if (watchedElsewhere(v)) return;
  resumeAt.set(v, v.currentTime);
  v.pause();
  v.load();
};

const quiet = new WeakSet(); // pauses the page made, which are not the visitor's

function player(v, { auto = true } = {}) {
  let want = auto && !reduced.matches,
    shown = false,
    start = 0,
    drop = 0;
  const run = () => v.play().catch(() => {}); // a refusal leaves the poster and the play button up
  const halt = () => {
    if (v.paused || watchedElsewhere(v)) return;
    quiet.add(v);
    v.pause();
  };
  // whatever starts it -- the page, a click, the browser's own controls -- picks up where the
  // released clip left off
  v.addEventListener("play", () => {
    want = true;
    const t = resumeAt.get(v);
    resumeAt.delete(v);
    if (t) v.currentTime = t;
  });
  v.addEventListener("pause", () => {
    if (quiet.delete(v)) return;
    want = false; // the visitor's pause: a start still pending must not undo it
    clearTimeout(start);
  });
  const leave = () => {
    halt();
    drop = setTimeout(() => release(v), 1000);
  };
  new IntersectionObserver(
    (es) => {
      const e = es[es.length - 1]; // records can queue up: the last one is where it is now
      clearTimeout(start);
      clearTimeout(drop);
      shown = e.intersectionRatio >= 0.25;
      if (shown) {
        start = setTimeout(() => {
          if (want) run();
        }, DWELL);
        return;
      }
      if (e.isIntersecting) halt();
      else leave();
    },
    { threshold: [0, 0.25] },
  ).observe(v);
  // back from full screen or picture in picture to a place on the page that is out of view
  for (const ev of ["leavepictureinpicture", "webkitendfullscreen", "fullscreenchange"]) {
    // checked a moment later, once the observer has seen where the clip is now
    (ev === "fullscreenchange" ? document : v).addEventListener(ev, () =>
      setTimeout(() => {
        if (!shown && !watchedElsewhere(v)) leave();
      }, 200),
    );
  }
  return {
    toggle() {
      if (v.paused) run();
      else v.pause();
    },
  };
}

// A clip that illustrates a figure loops with no control row and is worked on its picture: a
// click plays or pauses it, and so do Space and Enter once it has the focus.  The browser's own
// controls stay in the markup for a page without the script.
function byHand(v, p) {
  v.controls = false;
  v.tabIndex = 0;
  v.setAttribute("aria-label", `${v.getAttribute("aria-label") || "Clip"}: Space plays or pauses it`);
  v.addEventListener("click", () => p.toggle());
  v.addEventListener("keydown", (e) => {
    if (e.target !== v || (e.key !== " " && e.key !== "Enter")) return;
    e.preventDefault();
    p.toggle();
  });
}

// The trial clips are small in their grid while the footage is 1080p, and the controller's screen
// recording has small labels: a button on the corner of each takes it to full screen, where the
// browser's controls come back for scrubbing.
const FULL =
  '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M2 6V2h4M10 2h4v4M14 10v4h-4M6 14H2v-4"/></svg>';
function fullScreenButton(v) {
  const b = document.createElement("button");
  b.type = "button";
  b.className = "full";
  b.innerHTML = FULL;
  b.title = "Full screen";
  b.setAttribute("aria-label", "Play this clip in full screen");
  b.addEventListener("click", () => {
    if (v.requestFullscreen)
      v.requestFullscreen()
        .then(() => v.play().catch(() => {}))
        .catch(() => {});
    else if (v.webkitEnterFullscreen) {
      // iPhone: its own player, which refuses a clip that has no metadata yet
      v.play().catch(() => {});
      try {
        v.webkitEnterFullscreen();
      } catch {
        v.addEventListener("loadedmetadata", () => v.webkitEnterFullscreen(), { once: true });
      }
    }
  });
  v.after(b);
}
document.addEventListener("fullscreenchange", () => {
  document.querySelectorAll(".tile video").forEach((v) => {
    v.controls = document.fullscreenElement === v;
  });
});

// On the page a trial clip has a seek bar along its foot instead of a control row.  It shows where
// the clip is, and a click or a drag takes it elsewhere, as do the arrow keys once the bar has the
// focus.  A playing clip holds still while it is dragged and goes on from where it is let go; one
// without its data plays from the point picked.
const lengths = new WeakMap(); // each clip's length, for when it has let go of its data
const clock = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const KEYS = {
  // a key's move, from the time the clip is at and its length
  ArrowLeft: (t) => t - 2,
  ArrowDown: (t) => t - 2,
  ArrowRight: (t) => t + 2,
  ArrowUp: (t) => t + 2,
  PageDown: (t) => t - 10,
  PageUp: (t) => t + 10,
  Home: () => 0,
  End: (t, len) => len,
};
const SLOP = 6; // px a finger moves sideways before it drags, rather than scrolls the page

function seekBar(v, name, p) {
  const bar = document.createElement("div");
  bar.className = "seek";
  bar.tabIndex = 0;
  bar.setAttribute("role", "slider");
  bar.setAttribute("aria-label", `Position in ${name}`);
  bar.setAttribute("aria-valuemin", "0");
  bar.setAttribute("aria-valuenow", "0");
  bar.setAttribute("aria-valuetext", "0:00");
  bar.innerHTML = '<span class="fill"></span><span class="knob"></span>';
  v.after(bar);

  const length = () => (v.duration > 0 && v.duration < Infinity ? v.duration : lengths.get(v) || 0);
  const now = () => resumeAt.get(v) ?? v.currentTime; // a released clip goes on from where it let go
  const show = (f) => bar.style.setProperty("--p", Math.min(1, Math.max(0, f)).toFixed(4));
  let said = -1; // the second the slider last told assistive technology
  const draw = (at = now()) => {
    const len = length();
    if (!len) return;
    show(at / len);
    if (Math.floor(at) === said) return;
    said = Math.floor(at);
    bar.setAttribute("aria-valuemax", String(Math.floor(len)));
    bar.setAttribute("aria-valuenow", String(Math.min(said, Math.floor(len))));
    bar.setAttribute("aria-valuetext", `${clock(at)} of ${clock(len)}`);
  };
  let frame = 0;
  const follow = () => {
    draw();
    frame = v.paused ? 0 : requestAnimationFrame(follow);
  };
  v.addEventListener("play", () => {
    if (!frame) frame = requestAnimationFrame(follow);
  });
  v.addEventListener("durationchange", () => {
    if (v.duration > 0 && v.duration < Infinity) lengths.set(v, v.duration);
  });
  for (const ev of ["durationchange", "seeking", "seeked", "pause", "emptied"])
    v.addEventListener(ev, () => draw());

  // A seek is a rule from the clip's length to a time.  A clip with its data goes there at once; one
  // without keeps the point and goes there as it plays, which is what fetches it.  It tells whether
  // the clip has to be played for that.
  let pending = null;
  v.addEventListener("loadedmetadata", () => {
    if (!pending) return;
    resumeAt.delete(v);
    v.currentTime = Math.max(0, Math.min(pending(v.duration), v.duration - 0.05));
    pending = null;
  });
  const seek = (to) => {
    const len = length();
    if (!len) {
      pending = to;
      return true;
    }
    const t = Math.max(0, Math.min(to(len), len - 0.05));
    draw(t);
    if (v.readyState || !v.paused) {
      resumeAt.delete(v); // the point picked, not the one it let go at
      v.currentTime = t;
      return false;
    }
    resumeAt.set(v, t); // where the play handler puts it
    return true;
  };

  // While a drag seeks, it owns the playback: the clip holds still, whatever starts it meanwhile,
  // and plays once it is let go if it was playing or has yet to load.  One pointer at a time.
  const fraction = (e) => {
    const r = bar.getBoundingClientRect();
    return Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
  };
  let drag = null;
  const hold = () => {
    if (v.paused) return;
    quiet.add(v);
    v.pause();
    drag.resume = true;
  };
  v.addEventListener("play", () => {
    if (drag?.seeking) hold();
  });
  const move = (e) => {
    const f = fraction(e);
    show(f);
    if (seek((len) => f * len)) drag.load = true;
  };
  const start = (e) => {
    drag.seeking = true;
    drag.from = now();
    hold();
    move(e);
  };
  bar.addEventListener("pointerdown", (e) => {
    if (drag || e.button !== 0) return;
    drag = { id: e.pointerId, x: e.clientX, y: e.clientY, seeking: false, resume: false, load: false };
    bar.setPointerCapture(e.pointerId);
    bar.classList.add("drag");
    if (e.pointerType !== "touch") start(e); // a finger first shows it is not scrolling the page
  });
  bar.addEventListener("pointermove", (e) => {
    if (drag?.id !== e.pointerId) return;
    if (drag.seeking) move(e);
    else if (Math.abs(e.clientX - drag.x) > Math.max(SLOP, Math.abs(e.clientY - drag.y))) start(e);
  });
  const end = (e, done) => {
    if (drag?.id !== e.pointerId) return;
    const d = drag;
    if (done) {
      if (d.seeking) move(e);
      else start(e); // a tap
    } else if (d.seeking) {
      // the gesture went to the page, or the bar lost the pointer: the clip goes back
      pending = null;
      d.load = false;
      if (length()) seek(() => d.from);
    }
    drag = null;
    bar.classList.remove("drag");
    if (d.load || d.resume) v.play().catch(() => {});
  };
  bar.addEventListener("pointerup", (e) => end(e, true));
  bar.addEventListener("pointercancel", (e) => end(e, false));
  bar.addEventListener("lostpointercapture", (e) => end(e, false));

  bar.addEventListener("keydown", (e) => {
    if (e.key === " " || e.key === "Enter") {
      e.preventDefault();
      p.toggle();
      return;
    }
    const key = Object.hasOwn(KEYS, e.key) && KEYS[e.key];
    if (!key) return;
    e.preventDefault();
    const from = now();
    if (seek((len) => key(from, len))) v.play().catch(() => {});
  });
}

document.querySelectorAll("video[data-autoplay]").forEach((v) => {
  const name = v.getAttribute("aria-label") || "Clip";
  const p = player(v);
  byHand(v, p);
  if (!v.closest(".tile")) return;
  fullScreenButton(v);
  seekBar(v, name, p); // between the clip and its button, in the order of the keyboard
});

// The narrated video is started by hand and keeps the browser's controls, for its sound and full
// screen; it still pauses when scrolled away.
document
  .querySelectorAll("video[controls]:not([data-autoplay])")
  .forEach((v) => player(v, { auto: false }));

// --- the interactive controller ---
// The bundle is about 85 MB, so it is loaded only when asked for.  On a narrow screen the start
// link opens it in a tab of its own, where its panels have room.  Inside the page a veil covers
// it until it is clicked and comes back once the pointer is on the page again, so scrolling the
// page past the demo never zooms its camera.
const demo = document.getElementById("demo");
const wide = matchMedia("(min-width: 761px)");

function startDemo() {
  const preview = demo.querySelector(".demo-preview");
  preview.pause();
  preview.querySelectorAll("source").forEach((s) => s.remove());
  preview.load(); // drops what the recording had downloaded
  preview.remove();
  demo.querySelector(".demo-launch").remove();

  const frame = document.createElement("iframe");
  frame.src = demo.dataset.src;
  frame.title = "Interactive demo of the FLoRe controller";
  frame.allow = "fullscreen";

  const veil = document.createElement("button");
  veil.type = "button";
  veil.className = "demo-veil";
  veil.innerHTML = "<span>Click to interact</span>";
  veil.setAttribute("aria-label", "Interact with the demo");
  veil.hidden = true; // the visitor has just clicked: the demo has the mouse
  veil.addEventListener("click", () => {
    veil.hidden = true;
    frame.focus();
  });
  // Over the demo the pointer's moves go to its own document; any move this page sees outside
  // the demo means the visitor is back on the page.  (A pointerleave fires as soon as the
  // pointer enters the frame, so it cannot tell the two apart.)
  document.addEventListener(
    "pointermove",
    (e) => {
      if (!demo.contains(e.target) && document.fullscreenElement !== demo) veil.hidden = false;
    },
    { passive: true },
  );

  const full = document.createElement("button");
  full.type = "button";
  full.className = "full";
  full.innerHTML = FULL;
  full.title = "Full screen";
  full.setAttribute("aria-label", "Show the demo in full screen");
  full.addEventListener("click", () => {
    if (document.fullscreenElement === demo) document.exitFullscreen();
    else if (demo.requestFullscreen) demo.requestFullscreen().catch(() => {});
    else window.open(frame.src.replace("?embed", ""), "_blank", "noopener"); // iPhone: no element full screen
  });

  demo.append(frame, veil, full);
  frame.focus();
}

demo?.querySelector(".demo-launch").addEventListener("click", (e) => {
  if (!wide.matches) return; // the link opens its own tab
  e.preventDefault();
  startDemo();
});

// The link to the demo's own tab hops letter by letter (style.css): each letter gets its place in
// the wave, the arrow comes last, and a word stays in one piece while the line can still break
// between words.  The link is named by its words, not by its many letters.
const hop = document.querySelector(".demo-open .hop");
if (hop) {
  const text = hop.textContent;
  hop.closest("a").setAttribute("aria-label", text);
  let beat = 0;
  hop.replaceChildren(
    ...text.split(" ").flatMap((word, k) => {
      const w = document.createElement("span");
      w.className = "w";
      for (const c of word) {
        const s = document.createElement("span");
        s.className = "ch";
        s.style.setProperty("--i", beat++);
        s.textContent = c;
        w.append(s);
      }
      beat++; // the space between words takes a beat too
      return k ? [" ", w] : [w];
    }),
  );
  hop.nextElementSibling?.style.setProperty("--i", beat);
}

// --- the integral at work: the press that the generator's diagram is drawn for, run again as its
// gains are dragged on the pad beside it ---
// An idealised press of the controller's law along x: the reference brings a hand that lags its
// target to a surface that stops it and pushes back in proportion to how deep the target lies, at a
// stiffness the robot does not know; the command ramps up, holds and ramps down, and the reference
// leaves again.  It illustrates the law and is not a recording.  scripts/draw_controller.py draws
// the generator to scale for this press once it has settled at the paper's gains, so the two share
// their constants: change them together.  A run takes well under a millisecond, so the press is run
// again whenever the pad moves.
const PRESS = {
  k: 200, // N/m, the paper's K, at the pad's centre
  ki: 0.5, // 1/s, the paper's K_I, at the pad's centre
  kSpan: [100, 400], // the pad's K, on a log scale
  kiSpan: [0.125, 2], // and its K_I
  iMax: 40, // N, the integral's bound
  onset: 2, // N: the contact phase switches on above this force,
  release: 1.5, // N: and off below this one,
  tauC: 0.05, // s: and a low-pass smooths it
  kContact: 150, // N/m: how hard the stopped hand presses per metre its target lies beyond the surface
  tauHand: 0.12, // s, how far the hand lags its target
  arrive: 1.2, // s: the reference reaches the surface,
  leave: 7.9, // s: and leaves it,
  speed: 0.1, // m/s: on its way there and away
  fPeak: 15, // N, the command held
  ramp: [1.2, 1.7, 7.7, 8.2], // s: it ramps up, holds, and ramps down
  tEnd: 9.6, // s
  dt: 0.005, // s, the step the integral runs at
};

function press(k, ki) {
  const P = PRESS;
  const [a, b, c, d] = P.ramp;
  const fCmd = (t) => {
    if (t <= a || t >= d) return 0;
    if (t < b) return (P.fPeak * (t - a)) / (b - a);
    return t <= c ? P.fPeak : (P.fPeak * (d - t)) / (d - c);
  };
  const gap = (t) => P.speed * (Math.max(0, P.arrive - t) + Math.max(0, t - P.leave)); // p* short of the surface
  const n = Math.round(P.tEnd / P.dt) + 1;
  const run = { n, cmd: new Float64Array(n), meas: new Float64Array(n), integ: new Float64Array(n) };
  const beta = 1 - Math.exp(-P.dt / P.tauC);
  let x = -gap(0), // the hand, from the surface: where it would be, unstopped, as it lags its target
    integ = 0,
    phase = 0,
    on = false;
  for (let i = 0; i < n; i++) {
    const t = i * P.dt;
    const fc = fCmd(t);
    x += (((fc + integ) / k - gap(t) - x) * P.dt) / P.tauHand;
    const f = P.kContact * Math.max(0, x); // -f_meas: the reading points out of the surface
    if (!on && f > P.onset) {
      on = true;
      run.onset ??= t;
    } else if (on && f < P.release) {
      on = false;
      run.release = t;
    }
    phase += beta * (+on - phase);
    integ = phase * Math.min(Math.max(integ + ki * (fc - f) * P.dt, -P.iMax), P.iMax); // fc - f: f_cmd + f_meas
    run.cmd[i] = fc;
    run.meas[i] = f;
    run.integ[i] = integ;
  }
  return run;
}

// A gain from its place across the pad, 0 to 1 over a log scale, and back.
const gainAt = ([lo, hi], q) => lo * (hi / lo) ** q;
const placeOf = ([lo, hi], g) => Math.log(g / lo) / Math.log(hi / lo);
const f1 = (x) => x.toFixed(1);

function gainsChart(root) {
  const P = PRESS;
  // the force axis spans all that the pad can reach, so that it stays put while the dot moves
  let lo = 0,
    hi = 0;
  for (const q of [0, 0.5, 1])
    for (const r of [0, 0.5, 1]) {
      const run = press(gainAt(P.kSpan, q), gainAt(P.kiSpan, r));
      for (const ys of [run.cmd, run.meas, run.integ])
        for (const y of ys) {
          lo = Math.min(lo, y);
          hi = Math.max(hi, y);
        }
    }
  const yLo = Math.floor(lo) - 1,
    yHi = Math.ceil(hi) + 1;
  const home = [placeOf(P.kSpan, P.k), placeOf(P.kiSpan, P.ki)];
  let [u, v] = home; // the dot's place: K across, K_I up

  const svg = (cls) => {
    const s = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    s.setAttribute("class", cls);
    return s;
  };
  const plotBox = document.createElement("div");
  plotBox.className = "gains-plot";
  const plot = svg("gchart");
  plot.setAttribute("role", "img");
  plot.setAttribute(
    "aria-label",
    "The press over time: the commanded force, the measured force and the integral",
  );
  plotBox.append(plot);

  // the pad, in units of its own: a square of 160 with the paper's gains at its centre
  const padBox = document.createElement("div");
  padBox.className = "gains-pad";
  const pad = svg("ctl gpad");
  pad.setAttribute("viewBox", "0 0 210 210");
  pad.setAttribute("aria-hidden", "true");
  const S = { x0: 34, y1: 182, side: 160 };
  const grid = [0, 0.25, 0.5, 0.75, 1].map((q) => {
    const x = S.x0 + q * S.side,
      y = S.y1 - q * S.side;
    return (
      `<line class="grid" x1="${x}" y1="${S.y1 - S.side}" x2="${x}" y2="${S.y1}"/>` +
      `<line class="grid" x1="${S.x0}" y1="${y}" x2="${S.x0 + S.side}" y2="${y}"/>`
    );
  });
  pad.innerHTML =
    '<defs><marker id="gpad-head" markerUnits="userSpaceOnUse" markerWidth="10" markerHeight="10" ' +
    'refX="9" refY="5" orient="auto"><path d="M0,0 L10,5 L0,10 z"/></marker></defs>' +
    grid.join("") +
    `<line class="axis ax-k" x1="${S.x0}" y1="${S.y1}" x2="204" y2="${S.y1}" marker-end="url(#gpad-head)"/>` +
    `<line class="axis ax-ki" x1="${S.x0}" y1="${S.y1}" x2="${S.x0}" y2="6" marker-end="url(#gpad-head)"/>` +
    '<text class="ax-k" x="204" y="204" text-anchor="end"><tspan class="v">K</tspan></text>' +
    '<text class="ax-ki" x="4" y="18"><tspan class="v">K</tspan><tspan class="vs" dy="4.5">I</tspan></text>' +
    `<circle class="home" cx="${S.x0 + home[0] * S.side}" cy="${S.y1 - home[1] * S.side}" r="14"/>` +
    '<g class="knob"><circle class="pulse" r="10"/><circle class="dot" r="10"/>' +
    '<text class="hint" x="17" y="29">drag</text></g>';
  const knob = pad.querySelector(".knob");

  // the keyboard's way in: a slider for each gain, out of sight, whose focus the pad shows
  const sliders = [
    ["K, the nominal stiffness", P.kSpan, "newtons per metre"],
    ["K I, the integral's gain", P.kiSpan, "per second"],
  ].map(([name, span, unit]) => {
    const input = document.createElement("input");
    Object.assign(input, { type: "range", className: "sr", min: 0, max: 1, step: 1 / 32 });
    input.setAttribute("aria-label", name);
    return { input, span, unit };
  });
  padBox.append(pad, ...sliders.map((s) => s.input));
  root.replaceChildren(plotBox, padBox);

  let chart = null; // the plot's scales and the elements that change with the gains
  const layout = () => {
    const w = plotBox.clientWidth,
      h = plotBox.clientHeight;
    if (!w || !h || (chart && w === chart.w && h === chart.h)) return false;
    const small = w < 480;
    const m = small ? { l: 44, r: 8, t: 24, b: 40 } : { l: 48, r: 10, t: 26, b: 42 };
    const X = (t) => m.l + ((w - m.l - m.r) * t) / P.tEnd;
    const Y = (f) => h - m.b - ((h - m.t - m.b) * (f - yLo)) / (yHi - yLo);
    const marks = [];
    for (let f = Math.ceil(yLo / 5) * 5; f <= yHi; f += 5) {
      const y = f1(Y(f));
      marks.push(
        `<line class="${f ? "grid" : "zero"}" x1="${m.l}" y1="${y}" x2="${w - m.r}" y2="${y}"/>` +
          `<text x="${m.l - 7}" y="${f1(Y(f) + 4.5)}" text-anchor="end">${f < 0 ? `−${-f}` : f}</text>`,
      );
    }
    for (let t = 0; t <= P.tEnd; t += 2)
      marks.push(`<text x="${f1(X(t))}" y="${h - m.b + 18}" text-anchor="middle">${t}</text>`);
    const phase = (name) => `<text class="phase" y="${m.t - 9}" text-anchor="middle">${name}</text>`;
    plot.setAttribute("viewBox", `0 0 ${w} ${h}`);
    plot.innerHTML =
      `<defs><clipPath id="gchart-area"><rect x="${m.l}" y="${m.t}" width="${w - m.l - m.r}" ` +
      `height="${h - m.t - m.b}"/></clipPath></defs>` +
      `<rect class="band" y="${m.t}" height="${h - m.t - m.b}"/>${marks.join("")}` +
      `<line class="axis" x1="${m.l}" y1="${h - m.b}" x2="${w - m.r}" y2="${h - m.b}"/>` +
      `<text class="title" x="${f1((m.l + w - m.r) / 2)}" y="${h - 4}" text-anchor="middle">time (s)</text>` +
      `<text class="title" text-anchor="middle" ` +
      `transform="translate(${small ? 10 : 13} ${f1((m.t + h - m.b) / 2)}) rotate(-90)">force (N)</text>` +
      phase("approach") +
      phase("in contact") +
      phase("retreat") +
      '<g clip-path="url(#gchart-area)"><path class="area"/><path class="integ"/><path class="meas"/>' +
      '<path class="cmd"/></g>';
    const el = (c) => plot.querySelector(`.${c}`);
    chart = { w, h, m, X, Y, band: el("band"), phases: [...plot.querySelectorAll(".phase")] };
    // the reference's approach and retreat keep their times; the contact's name follows the band
    const fit = (t, x0, x1) => {
      t.setAttribute("x", f1((x0 + x1) / 2));
      t.style.visibility = x1 - x0 > t.textContent.length * 7.2 ? "" : "hidden";
    };
    fit(chart.phases[0], m.l, X(P.arrive));
    fit(chart.phases[2], X(P.leave), w - m.r);
    chart.fit = fit;
    for (const c of ["area", "integ", "meas", "cmd"]) chart[c] = el(c);
    return true;
  };

  const STEP = 3; // samples per point drawn: about a pixel apart at the column's width
  const draw = () => {
    if (!chart) return;
    const { X, Y } = chart;
    const run = press(gainAt(P.kSpan, u), gainAt(P.kiSpan, v));
    const path = (ys) => {
      let d = "";
      for (let i = 0; i < run.n; i += STEP) d += `${i ? "L" : "M"}${f1(X(i * P.dt))},${f1(Y(ys[i]))}`;
      return d;
    };
    const integ = path(run.integ);
    chart.cmd.setAttribute("d", path(run.cmd));
    chart.meas.setAttribute("d", path(run.meas));
    chart.integ.setAttribute("d", integ);
    chart.area.setAttribute("d", `${integ}L${f1(X(P.tEnd))},${f1(Y(0))}L${f1(X(0))},${f1(Y(0))}Z`);
    const on = X(run.onset ?? P.tEnd),
      off = X(run.release ?? P.tEnd);
    chart.band.setAttribute("x", f1(on));
    chart.band.setAttribute("width", f1(Math.max(0, off - on)));
    chart.fit(chart.phases[1], on, off);
    // what a screen reader is told: the press at these gains, as the command ends
    const end = Math.round(P.ramp[2] / P.dt);
    plot.setAttribute(
      "aria-label",
      `The press over time at these gains: as the command ends, the hand presses ${run.meas[end].toFixed(1)} N ` +
        `for the ${P.fPeak} N commanded, and the integral stands at ${run.integ[end].toFixed(1)} N`,
    );
    knob.setAttribute("transform", `translate(${f1(S.x0 + u * S.side)} ${f1(S.y1 - v * S.side)})`);
    [u, v].forEach((q, i) => {
      const { input, span, unit } = sliders[i];
      const g = gainAt(span, q);
      input.value = q;
      input.setAttribute("aria-valuetext", `${g < 10 ? g.toFixed(2) : Math.round(g)} ${unit}`);
    });
  };
  let frame = 0;
  const redraw = () => {
    frame ||= requestAnimationFrame(() => {
      frame = 0;
      draw();
    });
  };

  const clamp01 = (q) => Math.min(1, Math.max(0, q));
  const follow = (e) => {
    const r = pad.getBoundingClientRect();
    u = clamp01((((e.clientX - r.left) / r.width) * 210 - S.x0) / S.side);
    v = clamp01((S.y1 - ((e.clientY - r.top) / r.height) * 210) / S.side);
    redraw();
  };
  let owner = null; // the one pointer that drags the dot; a second finger waits until it lets go
  pad.addEventListener("pointerdown", (e) => {
    if (e.button || owner !== null) return;
    e.preventDefault(); // no text selected on the way
    owner = e.pointerId;
    pad.setPointerCapture(owner);
    padBox.classList.add("used", "drag");
    follow(e);
  });
  pad.addEventListener("pointermove", (e) => {
    if (e.pointerId === owner) follow(e);
  });
  pad.addEventListener("lostpointercapture", (e) => {
    if (e.pointerId !== owner) return;
    owner = null;
    padBox.classList.remove("drag");
  });
  sliders.forEach(({ input }, i) => {
    input.addEventListener("input", () => {
      if (i) v = +input.value;
      else u = +input.value;
      padBox.classList.add("used");
      redraw();
    });
    // the pad lights the axis of the gain whose slider has the focus
    input.addEventListener("focus", () => (padBox.dataset.focus = i ? "ki" : "k"));
    input.addEventListener("blur", () => delete padBox.dataset.focus);
  });

  new ResizeObserver(() => {
    if (layout()) draw();
  }).observe(plotBox);
}

const gainsBox = document.getElementById("gains");
if (gainsBox) gainsChart(gainsBox);

// --- nav: the current section marked ---
const nav = document.getElementById("nav");
const links = [...document.querySelectorAll("#nav .links a")];
const sections = links.map((a) => document.querySelector(a.getAttribute("href"))).filter(Boolean);

const spy = new IntersectionObserver(
  (entries) => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      links.forEach((a) => a.classList.toggle("on", a.getAttribute("href") === "#" + e.target.id));
    }
  },
  { rootMargin: "-45% 0px -50% 0px" },
);

sections.forEach((s) => spy.observe(s));

// --- the hero ---
// It is sticky, so it never leaves the viewport and an observer on it never fires: where the page
// is comes from the scroll position instead.  With motion allowed, the empty run after the hero is
// the teaser's: as the run scrolls by, the teaser follows it frame by frame, under each task's
// name.  At the run's end the side view fades in and plays by itself, with the title over it, and
// scrolling on, the page slides up over the hero while its text fades.  Under reduced motion there
// is no run (style.css): the title is there from the start, and nothing plays.
const hero = document.querySelector(".hero");
const run = document.querySelector(".hero-run");
const teaser = hero.querySelector(".hero-teaser");
const heroClip = hero.querySelector(".hero-bg");
// The teaser's frame rate and the span of each of its shots in seconds, "from,to" apart by spaces,
// which scripts/build_media.py writes on it as it builds it.
const [rateNum, rateDen] = teaser.dataset.rate.split("/").map(Number);
const FPS = rateNum / rateDen;
const SPANS = teaser.dataset.spans.split(" ").map((span) => span.split(",").map(Number));
const LENGTH = SPANS[SPANS.length - 1][1];
// Screens of scrolling down the run for the fade to the title and for the title at rest; the
// teaser has the rest of the run, whose length in screens style.css gives as --run.
const FADE = 0.6;
const REST = 0.6;
const CUE = 1.2; // s of teaser over which a task's name comes in or goes
const LAG = 0.12; // s, the time constant with which the teaser's picture follows the scroll
const clamp = (x) => Math.min(1, Math.max(0, x));
const smooth = (x) => x * x * (3 - 2 * x);
const mainTop = () => document.querySelector("main").offsetTop;
// where the teaser has played, where the title is in and where the page covers the hero, in px
// scrolled; with no run they all come at once
const marks = () => {
  const len = run.offsetHeight;
  const screen = len / (parseFloat(getComputedStyle(run).getPropertyValue("--run")) || 1);
  return {
    len,
    played: len - (FADE + REST) * screen,
    titled: len - REST * screen,
    main: mainTop(),
  };
};

// A task's name comes in just after its shot has and goes just before the next shot's fade; the
// first is there from the start, and the last stays until the title takes over.  The names follow
// the frame on screen, not the one asked for.
const cues = [...hero.querySelectorAll(".hero-tasks p")].slice(0, SPANS.length).map((p, i) => ({
  p,
  from: SPANS[i][0],
  to: SPANS[i][1],
}));
const cue = (t) => {
  for (const c of cues) {
    const come = c.from > 0 ? clamp((t - c.from - 0.3) / CUE) : 1;
    const go = c.to < LENGTH ? clamp((c.to - 0.9 - t) / CUE) : 1;
    c.p.style.setProperty("--o", smooth(Math.min(come, go)).toFixed(3));
  }
};
cue(0); // the poster is the first frame

// The teaser is sought one seek at a time, each to the latest frame asked for, and its picture
// eases after the scroll, so that a wheel's notches play as motion.  A seek ends in "seeked"; a
// load that fails or starts over leaves none running.
let want = 0, // s of teaser that the scroll asks for
  eased = null, // s of teaser to seek to, easing after it
  asked = -1, // the frame last sought
  seeking = false,
  easing = 0,
  then = 0;
const seek = () => {
  if (seeking || eased === null || teaser.readyState < 1 || reduced.matches) return; // loadedmetadata comes back here
  const f = Math.max(0, Math.min(Math.round(eased * FPS), Math.round(teaser.duration * FPS) - 1));
  if (f === asked) return;
  asked = f;
  seeking = true;
  teaser.currentTime = (f + 0.5) / FPS;
};
teaser.addEventListener("seeked", () => {
  seeking = false;
  cue(teaser.currentTime);
  seek();
});
teaser.addEventListener("loadedmetadata", () => {
  asked = -1;
  seek();
});
teaser.addEventListener("emptied", () => {
  seeking = false;
  asked = -1;
});
const ease = (now) => {
  const dt = then ? Math.min(0.1, (now - then) / 1000) : 0;
  then = now;
  // a jump, or the first frame, is sought at once
  if (eased === null || Math.abs(want - eased) > 8) eased = want;
  else eased += (want - eased) * (1 - Math.exp(-dt / LAG));
  if (Math.abs(want - eased) < 0.2 / FPS) eased = want;
  seek();
  easing = eased === want ? 0 : requestAnimationFrame(ease);
  if (!easing) then = 0;
};

// A teaser that cannot be had leaves the title over the side view, as under reduced motion, and a
// visitor partway down the run back at the top.
teaser.addEventListener("error", () => {
  const inRun = scrollY < run.offsetHeight;
  hero.classList.add("still");
  if (inRun) scrollTo({ top: 0, behavior: "instant" });
  repaint();
});

// The teaser is fetched once the page comes to rest with the hero at least half in view: at the
// top once the page has loaded (or after 1.5 s), or on the way back up.  A page opened at a link
// into it, or scrolled past the hero on its way somewhere, fetches nothing for it.  The side view
// follows once the teaser can play through, or when it is seen.
const inView = (m) => m.len > 0 && scrollY < m.len + innerHeight * 0.5;
const fetchTeaser = () => {
  if (!teaser.dataset.src || reduced.matches || !inView(marks())) return;
  teaser.preload = "auto";
  attach(teaser);
  repaint();
  // iOS fetches no frame of a video that has not played: if none has come after a while, a play
  // that is paused at once fetches it
  setTimeout(() => {
    if (teaser.readyState >= 2 || reduced.matches) return;
    teaser.play().then(
      () => {
        teaser.pause();
        asked = -1;
        seek();
      },
      () => {},
    );
  }, 2000);
};
// a page opened at a link into it, which the browser is about to scroll to
const linked = (() => {
  try {
    const at = document.getElementById(decodeURIComponent(location.hash.slice(1)));
    return !!at && document.querySelector("main").contains(at);
  } catch {
    return false; // an address that does not decode names no element
  }
})();
let settle = 0; // pending while the page scrolls
const arrive = () => {
  if (!linked && !settle) fetchTeaser(); // a scroll under way fetches once it comes to rest
};
addEventListener("load", arrive, { once: true });
setTimeout(arrive, 1500);
teaser.addEventListener(
  "canplaythrough",
  () => {
    if (reduced.matches) return;
    heroClip.preload = "auto";
    attach(heroClip);
  },
  { once: true },
);

// The side view starts a moment after it comes into sight, however the scrolling goes on, and
// pauses as it leaves.
let heroSeen = false,
  heroStart = 0;
const syncHero = (seen) => {
  seen = seen && !reduced.matches;
  if (seen === heroSeen) return;
  heroSeen = seen;
  clearTimeout(heroStart);
  if (!seen) {
    heroClip.pause();
    return;
  }
  heroStart = setTimeout(() => {
    if (!heroSeen || reduced.matches) return;
    attach(heroClip);
    heroClip.play().catch(() => {});
  }, 150);
};

const paint = () => {
  const m = marks();
  const y = scrollY;
  const title = m.len ? smooth(clamp((y - m.played) / (m.titled - m.played))) : 1;
  const lift = clamp((y - m.len) / Math.max(1, innerHeight * 0.85));
  hero.style.setProperty("--title", title.toFixed(3));
  hero.style.setProperty("--lift", lift.toFixed(3));
  hero.style.setProperty("--fade", (1 - Math.min(1, lift * 1.35)).toFixed(3));
  hero.classList.toggle("titled", title > 0.5);
  nav.classList.toggle("stuck", y >= m.main - nav.offsetHeight);
  if (!m.len) {
    syncHero(hero.classList.contains("still") && lift < 0.99); // the teaser failed: the side view plays
    return;
  }
  want = clamp(y / m.played) * LENGTH;
  if (!easing) easing = requestAnimationFrame(ease);
  syncHero(!teaser.dataset.src && title > 0 && lift < 0.99); // once the hero is in use
};
let painting = 0;
const repaint = () => {
  if (!painting)
    painting = requestAnimationFrame(() => {
      painting = 0;
      paint();
    });
};
addEventListener("scroll", repaint, { passive: true });
addEventListener("resize", repaint);
reduced.addEventListener("change", () => {
  syncHero(false); // whatever was about to start does not
  repaint();
});
paint();

// The title's links are there to be reached by the keyboard before they are shown: one that takes
// the focus brings the title in.
hero.querySelector(".hero-body").addEventListener("focusin", () => {
  const m = marks();
  if (m.len && scrollY < m.titled) scrollTo({ top: m.titled, behavior: "instant" });
});

// Two stretches have no resting place: the fade from the teaser to the title, and the page's
// cover over the hero.  When the scrolling stops inside one, it goes on to the end it was last
// heading for, so a single wheel notch or arrow key is enough either way.  A link into the page
// is left where it landed, and under reduced motion the page is left where the visitor put it.
let lastY = scrollY,
  heading = 0,
  jumped = linked;
document.addEventListener("click", (e) => {
  if (e.target.closest?.('a[href^="#"]')) jumped = true;
});
// so do Back and Forward
addEventListener("hashchange", () => (jumped = true));
addEventListener("popstate", () => (jumped = true));
const settled = () => {
  fetchTeaser();
  const m = marks();
  const y = scrollY;
  const spans = m.len
    ? [
        [m.played, m.titled],
        [m.len, m.main],
      ]
    : [[0, m.main]];
  const span = spans.find(([a, b]) => y > a + 4 && y < b - 4);
  const link = jumped;
  jumped = false;
  if (!span || link || reduced.matches) return;
  scrollTo({ top: heading > 0 ? span[1] : span[0], behavior: "smooth" });
};
addEventListener(
  "scroll",
  () => {
    const y = scrollY;
    if (y !== lastY) heading = Math.sign(y - lastY);
    lastY = y;
    clearTimeout(settle);
    settle = setTimeout(() => {
      settle = 0;
      settled();
    }, 140);
  },
  { passive: true },
);

// --- a diagram or an equation too wide for the screen scrolls sideways in its own box; while it
// does, the keyboard can reach it too.  An equation's width is known once its fonts are in ---
const wides = document.querySelectorAll(".wide, .law .eq");
const reachable = () =>
  wides.forEach((w) => {
    if (w.scrollWidth > w.clientWidth + 1) w.tabIndex = 0;
    else w.removeAttribute("tabindex");
  });
addEventListener("resize", reachable);
reachable();
document.fonts?.ready.then(reachable);
