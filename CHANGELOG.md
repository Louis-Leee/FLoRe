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
