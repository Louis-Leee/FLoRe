// FLoRe project page: clips that load and play only while they are watched, the nav that marks
// the section you are in, and the teaser that gives way to the page.  The structure follows the
// Workhorse page's script (hybridrobotics.github.io/workhorse), trimmed to what is used here.

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

// The hero keeps its address in data-src and fetches nothing before it is set.
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

// --- the hero reel ---
// Its still is drawn with the page, and the reel is fetched once the rest has arrived -- or after
// 2.5 s on a slow line, where the load event can be minutes away.  It plays only while the teaser
// is on screen, so a link into the middle of the page fetches nothing for it.
const heroClip = document.querySelector(".hero-bg");
let heroReady = false,
  heroStart = 0;
const heroShown = () => scrollY < innerHeight * 0.85 * 0.99; // where the page covers it
const syncHero = () => {
  clearTimeout(heroStart);
  if (!heroReady || reduced.matches || !heroShown()) {
    heroClip.pause();
    return;
  }
  if (heroClip.paused)
    heroStart = setTimeout(() => {
      attach(heroClip);
      heroClip.play().catch(() => {});
    }, DWELL);
};
const heroArrives = () => {
  heroReady = true;
  syncHero();
};
addEventListener("load", heroArrives, { once: true });
setTimeout(heroArrives, 2500);

// --- nav: the current section marked, and a border once the page has covered the hero ---
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

// The hero is sticky, so it never leaves the viewport and an observer on it never fires: the
// page covering it is read from the scroll position instead.
const mainTop = () => document.querySelector("main").offsetTop;

// --- the teaser gives way to the page: its text fades and it drifts as the first screen scrolls ---
const hero = document.querySelector(".hero");
const onScroll = () => {
  const p = Math.min(1, scrollY / Math.max(1, innerHeight * 0.85));
  hero.style.setProperty("--lift", p.toFixed(3));
  hero.style.setProperty("--fade", (1 - Math.min(1, p * 1.35)).toFixed(3));
  nav.classList.toggle("stuck", scrollY >= mainTop() - nav.offsetHeight);
  syncHero();
};
addEventListener("scroll", () => requestAnimationFrame(onScroll), { passive: true });
addEventListener("resize", () => requestAnimationFrame(onScroll));
onScroll();

// The teaser has no resting place halfway: when the scrolling stops inside the handover, it goes
// on to the end it was heading for, so a single wheel notch or arrow key is enough either way.
// Under reduced motion the page is left where the visitor put it.
let settle,
  rest = scrollY; // where the page last came to rest outside the handover
addEventListener(
  "scroll",
  () => {
    clearTimeout(settle);
    settle = setTimeout(() => {
      const end = mainTop();
      if (scrollY <= 4 || scrollY >= end - 4) {
        rest = scrollY;
        return;
      }
      if (reduced.matches) return;
      scrollTo({ top: scrollY > rest ? end : 0, behavior: "smooth" });
    }, 140);
  },
  { passive: true },
);

// --- a diagram too wide for the screen scrolls sideways in its own box; while it does, the
// keyboard can reach it too ---
const wides = document.querySelectorAll(".wide");
const reachable = () =>
  wides.forEach((w) => {
    if (w.scrollWidth > w.clientWidth + 1) w.tabIndex = 0;
    else w.removeAttribute("tabindex");
  });
addEventListener("resize", reachable);
reachable();
