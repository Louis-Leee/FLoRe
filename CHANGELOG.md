## [1.3.1] - 2026-10-07
### Features
- The demo's *F meas inertial term* is off from the start, as its switch shows. Before, only the
  switch started off: the term stayed on until the switch was clicked twice, so the F^meas arrow
  jittered as soon as the demo opened (up to 11 N with the robot standing).
### Design Rationale
- The release's tracking command publishes its switches when it starts, but the runtime wipes
  those values as it loads, and a newly built panel does not publish them again. Until a switch is
  touched, the runtime goes by its own fallbacks, which the release keeps equal to the switches'
  defaults. `build_demo.py` changed only the switch, so the fallback (on) and the switch (off)
  disagreed. It now turns the fallback off in the built client too, and stops unless it finds
  exactly one.
### Notes & Caveats
- In `demo/`, only one constant in `assets/index-BRqdi99b.js` changes. The file keeps its name, so
  a browser that cached it in the last ten minutes keeps the old copy until that cache expires.

## [1.3.0] - 2026-10-07
### Features
- The Controller section explains the controller before its demo, with two diagrams: the unified
  reference generator, which moves a hand's target from the motion's reference into the surface
  (p^tgt = p* + Δx, Δx = (f^cmd + I)/K), and the soft contact phase with its integrator, as one
  press over time (the command and the measured force, the contact phase c switching at 2.0 and
  1.5 N, and the shift split into f^cmd/K and I/K). `scripts/draw_controller.py` writes both into
  `index.html`.
- The running text of the Pipeline, Controller, Real-world Experiments and Data Collection
  sections fills the column and is justified, as the abstract is. Below 700 px it stays
  ragged-right.
### Design Rationale
- The diagrams are inline SVG, as on the Workhorse page: they use the page's font and colours,
  follow the dark theme and stay sharp. On a phone each keeps a width of 640 px and scrolls sideways
  in its own box, so its labels stay readable; while it scrolls, the box takes keyboard focus.
- Variables are set in a serif italic and Δ in the upright serif, as in the paper: Inter's italic
  capital I is a bare slanted stroke, and I / K read as // K. Inter's latin subset has no Δ either.
- The press is a toy 1-D model, not a recording, but it uses all of the paper's constants,
  K_I = 0.5 s^-1 included; the hold is long enough for the integral to settle. The integral can
  pull the target back as well as push it further, as in the demo when the hand first presses
  harder than commanded, so the caption says both. Its curves are thinned to within 0.2 px of the
  simulated ones, which keeps the page at 31 KB instead of 100 KB.
- The colours follow the demo's arrows: violet for f^cmd and cyan for f^meas, darkened to #0b7285
  for 5.6:1 on white, with the page's green for the integral. Tints are opacities rather than
  `color-mix()`, so a browser without it still draws them.
- Plain justification, without hyphenation: Chrome's automatic hyphenation broke three words in
  the six lines of the pipeline caption, and `text-wrap: pretty` opened wide gaps in the first line
  of the experiments' summary to avoid a one-word last line.
### Notes & Caveats
- The Controller section's opening line is now a muted summary like the Experiments section's;
  the demo keeps its own lead-in.
- The diagrams' captions are hand-written in `index.html`; rerunning the script replaces only the
  drawings between the markers, and stops if the markers are missing, doubled or out of order.

## [1.2.1] - 2026-10-07
### Features
- The demo's motion list holds `idle_default` and the five SONIC clips only. The push-gantry clip
  is gone, and so is `scripts/h2r_view_to_holosoma.py`, which only converted that clip (it is in
  the history at d96cf45).
- Each of the three tips has a colour of its own: violet for *Command a force*, as the F^cmd
  arrow, blue for *Push the robot* and the page's green for *Look around*.
- The link to the demo's tab reads *Try the demo in a new tab!*. It is an Amazon-orange (#FF9900)
  pill at twice the former size.
### Design Rationale
- The pill's text is Amazon's dark ink (#232F3E): white on that orange has a contrast of 2.1:1,
  below the 3:1 that even large text needs, and the ink has 6.3:1. The tips take cool colours, which
  leaves the orange to the link.
- Each tip's tint is its colour mixed into the page background, so the dark theme gets dark tints
  from the same rule. The pill's padding is in em and its text shrinks with the viewport, down to
  1 rem, so it keeps to one line on a phone.
### Notes & Caveats
- In `demo/`, only the motion lists (`assets/config.json` and the policy's JSON) and the removed
  clip change; the client, the scene and the ONNX are the same bytes.

## [1.2.0] - 2026-10-07
### Features
- Five more reference motions in the demo, from the SONIC manipulation set: two walking with a
  crate held in both hands (`carry_crate_walk_forward`, `carry_crate_walk_sideways`) and three
  standing two-handed ones (`two_hands_pick_up_box`, `two_hands_put_down_box`,
  `two_hands_lift_heavy_box`). The push-gantry clip is now listed as `push_gantry`.
- The demo's *F meas inertial term* switch starts off.
- The Controller section's three tips are tinted cards with an icon and a short line each. The link
  to the demo's own tab is a solid pill whose letters hop in a wave every 4 s. Its lines break
  between words when the text is enlarged. The motion stops under reduced motion.
- `scripts/prepare_demo_motions.py` copies clips into a demo motion directory under readable names,
  each turned to face the camera.
### Design Rationale
- The clips were picked by tracking, measured with the bundle itself. 586 candidates were screened
  on kinematics to 62 (3–12 s; feet still for the standing kind, at least 1.8 m of walking for the
  walking kind). Each was then played in headless Chrome while the tracking panel was recorded at
  every step. All five keep a mean joint error of 3.4–4.4° and a hand error of 1.4–1.6 cm, close
  to the push-gantry clip's 4.6° and 1.5 cm. The two walks drift least from the reference: 0.3 m
  after 2.5 m, where most other walks drift 0.4–2 m. The three standing clips are the best-tracked of
  three different actions.
- The inertial term is a double finite difference of the hand position. It makes F^meas flicker in
  free space, and the policy observes that F^meas. With it off, F^meas is the wall reaction plus
  F^ext. A 15 N command still settles at 14.8 N. Only the switch's default changes.
- The SONIC clips start facing every which way, and the camera keeps one bearing, so most were seen
  from behind. Each clip is turned to start at the origin facing +x, as `idle_default` does. The
  errors measured before and after turning agree.
- Picking the current entry of a list again, or its x, emptied it. With no motion the robot sinks to
  the floor. `build_demo.py` now turns off `clearable` and `allowDeselect` on the scene, policy and
  motion lists of the built client. It fails if it does not find all three.
### Notes & Caveats
- On the walking clips the robot ends 0.3 m from the reference ghost after about 2.5 m. The
  controller tracks the body relative to itself, not its position in the world.
- The five SONIC clips are third-party motion capture, now downloadable from the page; their
  dataset's terms apply.
- Motions are fetched when picked, so the download before the demo starts is unchanged: 85 MB of
  files, about 55 MB sent, since GitHub Pages compresses them. The start button now says 55 MB.
- `h2r_view_to_holosoma.py` now leaves out MuJoCo's `world` body (32 bodies, as in the training
  clips). The builder rejects motions whose body lists differ.

## [1.1.0] - 2026-10-06
### Features
- Interactive controller demo: the 5-keypoint unified force–position controller (W&B run
  k7ai0nmv, iteration 23000) runs in the visitor's browser in the Controller section. A visitor can
  command a force with either hand, push the robot by dragging it or with F^ext, and switch
  between `idle_default` and a push-gantry reference clip. The recording stays as the preview
  until *Start* is pressed. On phones the button opens the demo in a tab of its own.
- `scripts/build_demo.py`, `scripts/demo_embed.js` and `scripts/h2r_view_to_holosoma.py` rebuild
  `demo/` from the code release.
### Design Rationale
- The bundle is the release's own mjswan build (same browser physics: spring wall, simulated F/T,
  contact phase, integrator), so the demo is the release's evaluation, not a re-implementation.
- It is about 80 MB, so it loads only on request. A veil keeps page scrolling from zooming the
  demo until the visitor clicks into it.
- The build targets the absolute sub-path `/FLoRe/demo/`: with a relative base, mjswan reads the
  first path segment as a project id.
- The FLoRe sources stay untouched. Folding, scaling and the camera are applied to the built
  bundle only.
### Notes & Caveats
- This publishes the stripped low-level policy ONNX (38.8 MB) and one reference clip. The README
  of the release still lists checkpoints as unreleased.
- If the page moves to another path or domain, rebuild with `MJSWAN_DEMO_BASE=/new/path/`.

## [1.0.0] - 2026-10-06
### Features
- First version of the FLoRe project page on an orphan `gh-pages` branch: full-screen hero with a
  looping reel, authors and affiliations, the narrated supplementary video, abstract, pipeline
  figure, low-level controller clip, the three real-world tasks (Wipe Board, Rotate Box, Push
  Gantry) with their force plots, and the robot-free data collection system.
- Content follows the anonymous Google Site (sites.google.com/view/icra2027-flore) in the same
  order, with the paper's abstract and task descriptions added; the layout, typography, scroll
  behaviour and clip handling follow the Workhorse page.
- `scripts/build_media.py` rebuilds every clip and picture from the ICRA2027 working folder.
### Design Rationale
- This page is not anonymous, so it uses the raw footage: Push Gantry clips without head mosaics,
  and the supplementary video re-rendered from the raw recording with the same subtitles and
  narration as `supp_v4.mp4`. The data-collection photo keeps the emoji over the demonstrator's
  face.
- One H.264 file per clip instead of Workhorse's HEVC + H.264 pairs: half the storage in a branch
  that every full clone of the code repository also downloads.
- Trial clips stay 1080p (CRF 24, about 2 Mb/s) because they can be opened full screen; they play
  muted, so their camera audio (which contains speech) is dropped.
- Clips start only while a quarter of them is on screen and release their download when they
  leave it, so a grid of 1080p clips does not load all at once.
### Notes & Caveats
- About 200 MB of media: a plain `git clone` of the repository fetches this branch too; use
  `git clone --single-branch -b main` for the code alone.
- The supplementary video still shows "ICRA 2027 Submission #8324" on its title card, as recorded.
- `og:image` points at https://louis-leee.github.io/FLoRe/, the default Pages address of the repository.
