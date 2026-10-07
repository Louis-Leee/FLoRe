// FLoRe project page additions to the mjswan bundle (copied into demo/ by build_demo.py).
// Each debug panel folds away under its title.  Inside the page (?embed), and on a phone, the
// panels start smaller with the hybrid controls folded, so the robot is not hidden behind them;
// a phone folds the readout too.
(() => {
  const phone = matchMedia("(max-width: 760px)").matches;
  const embed = new URLSearchParams(location.search).has("embed") || phone;
  // panel id -> starts folded in the compact layout
  const PANELS = {
    "mjswan-slider-panel": false,
    "wbt-hybrid-controls": true,
    "wbt-realtime-metrics": phone,
  };

  const style = document.createElement("style");
  style.textContent = `
    .flore-fold > :first-child { cursor: pointer; user-select: none; }
    .flore-fold > :first-child::before { content: "\\25BE\\00a0"; opacity: .75; }
    .flore-fold.folded > :first-child::before { content: "\\25B8\\00a0"; }
    .flore-fold.folded > :not(:first-child) { display: none !important; }
    .flore-embed #mjswan-slider-panel { transform: scale(.82); transform-origin: top left; }
    .flore-embed #wbt-hybrid-controls { transform: scale(.82); transform-origin: bottom left; }
    .flore-embed #wbt-realtime-metrics { transform: scale(.82); transform-origin: bottom right; }
  `;
  document.head.append(style);
  if (embed) document.documentElement.classList.add("flore-embed");

  // The panels are made once the scene has loaded: watch for them, then stop watching.
  const ready = new Set();
  const watch = new MutationObserver(() => {
    for (const [id, folded] of Object.entries(PANELS)) {
      const panel = document.getElementById(id);
      if (ready.has(id) || !panel || !panel.firstElementChild) continue;
      panel.classList.add("flore-fold");
      panel.classList.toggle("folded", embed && folded);
      const title = panel.firstElementChild;
      title.title = "Click to fold or unfold";
      title.addEventListener("click", () => panel.classList.toggle("folded"));
      ready.add(id);
    }
    if (ready.size === Object.keys(PANELS).length) watch.disconnect();
  });
  watch.observe(document.documentElement, { childList: true, subtree: true });
})();
