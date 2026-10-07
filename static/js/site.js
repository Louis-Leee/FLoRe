// FLoRe project page: clips that load and play only while they are watched, the nav that marks
// the section you are in, and the teaser that gives way to the page.  The structure follows the
// Workhorse page's script (hybridrobotics.github.io/workhorse), trimmed to what is used here.

const reduced = matchMedia("(prefers-reduced-motion: reduce)");
const DWELL = 500; // ms a figure stays on screen before it starts, so a page jump starts nothing

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

const watchedElsewhere = (v) =>
  document.pictureInPictureElement === v || document.fullscreenElement === v;

// Stops a video's download and frees its decoder.  Its time is kept aside and set again only as
// it plays: a time set on an emptied video makes the browser fetch it straight back.
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
    start = 0,
    drop = 0;
  const run = () => {
    const t = resumeAt.get(v);
    resumeAt.delete(v);
    if (t) v.currentTime = t;
    v.play().catch(() => {}); // a refusal leaves the poster and the play button up
  };
  const halt = () => {
    if (v.paused || watchedElsewhere(v)) return;
    quiet.add(v);
    v.pause();
  };
  v.addEventListener("play", () => {
    want = true;
  });
  v.addEventListener("pause", () => {
    if (!quiet.delete(v)) want = false;
  });
  new IntersectionObserver(
    ([e]) => {
      clearTimeout(start);
      if (e.isIntersecting) clearTimeout(drop);
      if (e.intersectionRatio >= 0.25) {
        if (want) start = setTimeout(run, DWELL);
        return;
      }
      halt();
      if (!e.isIntersecting) drop = setTimeout(() => release(v), 1000);
    },
    { threshold: [0, 0.25] },
  ).observe(v);
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
  v.setAttribute(
    "aria-label",
    `${v.getAttribute("aria-label") || "Clip"}: Space plays or pauses it`,
  );
  v.addEventListener("click", () => p.toggle());
  v.addEventListener("keydown", (e) => {
    if (e.target !== v || (e.key !== " " && e.key !== "Enter")) return;
    e.preventDefault();
    p.toggle();
  });
}

// The trial clips are small in their grid, and the footage is 1080p: a button on each corner
// takes it to full screen, where the browser's controls come back for scrubbing.
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
      v.play().catch(() => {});
      v.webkitEnterFullscreen();
    } // iPhone
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

// --- nav: a border once you leave the hero, and the current section marked ---
const nav = document.getElementById("nav");
const links = [...document.querySelectorAll("#nav .links a")];
const sections = links
  .map((a) => document.querySelector(a.getAttribute("href")))
  .filter(Boolean);

const spy = new IntersectionObserver(
  (entries) => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      links.forEach((a) =>
        a.classList.toggle("on", a.getAttribute("href") === "#" + e.target.id),
      );
    }
  },
  { rootMargin: "-45% 0px -50% 0px" },
);

sections.forEach((s) => spy.observe(s));

const heroEl = document.getElementById("top"); // not `top`: that name is already window.top
new IntersectionObserver(
  ([e]) => nav.classList.toggle("stuck", !e.isIntersecting),
  { threshold: 0.02 },
).observe(heroEl);

// --- things rise into place the first time they are seen ---
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

// --- the teaser gives way to the page: its text fades and it drifts as the first screen scrolls ---
const hero = document.querySelector(".hero");
const onScroll = () => {
  const p = Math.min(1, scrollY / Math.max(1, innerHeight * 0.85));
  hero.style.setProperty("--lift", p.toFixed(3));
  hero.style.setProperty("--fade", (1 - Math.min(1, p * 1.35)).toFixed(3));
  syncHero();
};
addEventListener("scroll", () => requestAnimationFrame(onScroll), {
  passive: true,
});
onScroll();

// The teaser has no resting place halfway: when the scrolling stops inside the handover, it
// settles on whichever end is nearer.
const mainTop = () => document.querySelector("main").offsetTop;
let settle;
addEventListener(
  "scroll",
  () => {
    clearTimeout(settle);
    settle = setTimeout(() => {
      const end = mainTop();
      if (scrollY <= 4 || scrollY >= end - 4) return;
      scrollTo({
        top: scrollY < end * 0.5 ? 0 : end,
        behavior: reduced.matches ? "auto" : "smooth",
      });
    }, 140);
  },
  { passive: true },
);
