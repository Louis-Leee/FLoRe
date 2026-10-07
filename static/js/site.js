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

document.querySelectorAll("video[data-autoplay]").forEach((v) => {
  const p = player(v);
  byHand(v, p);
  if (v.closest(".tile")) fullScreenButton(v);
});

// The narrated video is started by hand and keeps the browser's controls, for its sound and full
// screen; it still pauses when scrolled away.
document
  .querySelectorAll("video[controls]:not([data-autoplay])")
  .forEach((v) => player(v, { auto: false }));

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
