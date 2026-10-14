# FLoRe project page (`gh-pages`)

The project page of *FLoRe: Learning Force-Reactive Humanoid Loco-Manipulation*, served by
GitHub Pages from this branch. It is a static page with no build step for the HTML: `index.html`,
one stylesheet (`static/css/style.css`) and one script (`static/js/site.js`). The layout follows the
[Workhorse page](https://hybridrobotics.github.io/workhorse/).

## Preview locally

```bash
npx http-server -p 8000          # in this folder, then open http://localhost:8000
```

The server has to answer byte-range requests, as GitHub Pages does, or the clips cannot be sought:
Python's `http.server` does not.


## Rebuild the media

Every clip and picture in `static/videos` and `static/images` is made from the paper's working
folder by one script, which never writes into that folder:

```bash
python3 scripts/build_media.py --src PAPER_DIR                       # fills in what is missing
python3 scripts/build_media.py --src PAPER_DIR --only supp --force
```

`PAPER_DIR` is the paper's working folder. It needs Pillow, and an ffmpeg with libass (Homebrew
`ffmpeg-full`) for the subtitles of the narrated video. The sources, encodings and the reason for
each choice are listed at the top of the script. All clips are single H.264 files (1080p trial
clips at CRF 24, the narrated video at 720p CRF 20), so every browser plays them without a second
codec.

The hero is two clips. The teaser joins three teaser clips, one per task, and is played by the
page's scrolling: at 15 fps, with a keyframe every 8 frames and none predicted from a later one, so
that whichever frame the scroll asks for decodes at once. Then the side view of the gantry push
loops under the title. The script reads the teaser clips from this folder's `teaser/` and the
title's rose from `rose.png`, and git ignores both: the clips are 200 MB, and one of them is over
GitHub's 100 MB limit. The build writes the teaser's frame rate and the span of each of its shots
on its `<video>` in `index.html` (`data-rate`, `data-spans`): the page's script maps the scroll to
frames by them, and times each task's name by its shot. The run of page that the teaser plays over
is `--run` screens long, in `style.css`.

The Controller section's two clips of the controller on the robot are made from this folder's
`compliance.MP4` and `forcetracking.MP4`, which git ignores as well: the first alone is 105 MB.

The narrated video's title slide is made again from the clip it plays, rendered with the title
alone; the rest of the recording keeps its frames, and the narration its timing. That clip is made
beside its source in the working folder, with numpy, Pillow and OpenCV:

```bash
cd PAPER_DIR/supp/push_gantry
python3 add_title_reveal.py --occlude --edge-offset 90 --x0 356 --no-subtitle --out IMG_3352_title_occluded_nosub.mp4
```

The data-collection clip is made from a master with the two strips of tape taken off the table
and an arrow drawn on the table round the crate as it turns. The arrow stays where it was drawn,
over everything, while the crate is dragged across it. That master is made beside its source in
the working folder, with numpy, OpenCV and Pillow; its constants (the tapes' lines, the frames that
show them uncovered, the arrow's timing) belong to that one clip:

```bash
python3 scripts/clean_data_collection.py --src PAPER_DIR    # writes PAPER_DIR/supp/data_collection_clean.mp4
```

## Redraw the controller diagrams

The Controller section's two diagrams are inline SVG in `index.html`, written between marker
comments by a script that needs only the standard library; their captions stay hand-written:

```bash
python3 scripts/draw_controller.py
```

In both, the motion's reference p* has put the hand on the surface. The press is a toy 1-D run of
the paper's law with all of its constants, not a recording: the reference brings a hand that lags
its target to the surface, holds it there while the command presses, and retreats. The reference
generator is drawn to scale for that press once it has settled. Captions and figures give no
physical constants. The script refuses to write unless each diagram has exactly one pair of
markers.

## Typeset the controller's equations

The Controller section explains the controller's law in equations, in the paper's notation and along
one axis. They are KaTeX, typeset into `index.html` once rather than in the browser. Each formula's
TeX sits in a comment before its rendering, and one script renders every formula again from its TeX.
It needs Node, and `npx` fetches KaTeX 0.19.0 the first time:

```bash
python3 scripts/render_math.py
```

The page loads only KaTeX's stylesheet and the fonts that its formulas use. Both are in
`static/katex/`: that release's own stylesheet, cut to its woff2 fonts. The script's macros colour
the forces and the integral as the diagrams do, through classes in `style.css`.

## Rebuild the interactive demo

`demo/` is the static browser bundle of the 5-keypoint controller, built by the code release's own
`wbt-realtime-eval` (mjswan: MuJoCo and ONNX Runtime compiled to WebAssembly), then adapted to this
page by `scripts/build_demo.py`: the sub-path `/FLoRe/demo/`, a closer camera, the *F meas
inertial term* off from the start (its switch and the runtime's fallback for it), scene, policy
and motion lists that cannot be emptied, and
`scripts/demo_embed.js`, which lets the panels fold away. The page loads it into a frame only when
the visitor presses *Start*.

1. A FLoRe checkout with its submodule, and the release's browser-viz stack. On Linux that is
   `third_party/holosoma_extensions/scripts/_install_viz_stack.sh`. On macOS, install the four
   packages with `--no-deps` (`holosoma_inference` asks for the Linux-only `evdev`) and then add
   `mujoco==3.8.1 onnx onnxruntime numpy loguru nodeenv rich typer pydantic scipy tyro pyyaml termcolor
   defusedxml websockets filelock importlib_metadata netifaces sshkeyboard wandb pin pyzmq`.
2. The policy: `model_23000.onnx` of W&B run `far-wandb/Holosoma_sihangli/k7ai0nmv`
   (437 MB; the build strips it to the 38.8 MB obs→action graph).
3. The reference clips: a directory of holosoma clips, which all need the same body list.
   `python scripts/prepare_demo_motions.py <motion_dir> <name>=<clip.npz> ...` copies each under
   the name the demo lists and turns it to face the camera. The builder lists every clip in the
   directory, so start from an empty one; `idle_default` is added by the builder. The page uses five
   clips of the SONIC manipulation set, two walking and three standing two-handed:

   | Name in the demo | SONIC manipulation clip |
   | --- | --- |
   | `carry_crate_walk_forward` | `lift_crate_walk_ff_loop_180_R_002__A163` |
   | `carry_crate_walk_sideways` | `lift_crate_walk_ff_loop_270_R_003__A203` |
   | `two_hands_pick_up_box` | `medium_light_two_hands_pick_up_front_medium_R_002__A511` |
   | `two_hands_put_down_box` | `big_heavy_two_hands_put_down_front_medium_R_001__A509` |
   | `two_hands_lift_heavy_box` | `medium_heavy_two_hands_pick_up_front_medium_R_001__A508` |

   They are already holosoma NPZ files. Of 586 crate-carrying and two-handed clips, 62 were
   shortlisted on their kinematics and played through the demo itself in headless Chrome; these
   five tracked best (mean joint error 3.4–4.4°, hand error 1.4–1.6 cm).
4. `python scripts/build_demo.py --model-path model_23000.onnx --motion-dir <motion_dir>` (with
   the viz-stack Python) rewrites `demo/`.

### The in-page demo (`demo-x/`)

The page embeds a simpler build of the same controller, `demo-x/`, and opens the full one, `demo/`,
in a tab of its own:
- only the left hand's command and an extra push on it are offered, both along x;
- its wall stands in front of the hand from the start;
- each arrow carries its value along x, in newtons;
- only the standing motion is kept. The wall stays where the hand started, as in training, and
  the walking and lifting clips would carry the hand into it.

`scripts/build_demo_x.py` builds it after `demo/`, with the same arguments. It leaves `demo/`,
`build_demo.py` and the code release's checkout untouched: it builds from a copy of the release's
mjswan template and command term, changed by `scripts/demo_x.patch`.

```bash
python scripts/build_demo_x.py --model-path model_23000.onnx --motion-dir <motion_dir>
```

The preview that the page plays before *Start* is recorded from the built demo by
`scripts/record_demo_x.mjs`. It needs Playwright and Chrome, and the site served with `FLoRe` as its
folder's name, since the bundle's base path is `/FLoRe/demo-x/`. It writes a master into the paper's
folder, which `build_media.py` then encodes as `controller_x`:

```bash
node scripts/record_demo_x.mjs http://localhost:8765/FLoRe/demo-x/ PAPER_DIR/supp/demo_x_preview.mp4
python3 scripts/build_media.py --src PAPER_DIR --only controller_x --force
```

To embed the full demo again, point the `#demo` frame's `data-src` and its *Start* link back to
`./demo/`, its preview back to `controller.mp4` and `controller.jpg`, restore the paragraph and the
tips above and below it, and name the pill *Try the demo in a new tab!* again. Reverting the commit
that named the pill's tab the full demo, and then the one that switched the frame to `demo-x/`,
does all of that.

## Credits

- Font: [Inter](https://github.com/rsms/inter), SIL Open Font License 1.1 (`static/fonts/OFL.txt`).
- Icon: the 🌺 of [Noto Emoji](https://github.com/googlefonts/noto-emoji) v2.047, Apache License 2.0.
- Equations: [KaTeX](https://katex.org/) 0.19.0, its stylesheet and fonts, MIT License (`static/katex/LICENSE`).
- Page structure: adapted from the [Workhorse](https://hybridrobotics.github.io/workhorse/) project page.
- Interactive demo: [mjswan](https://github.com/ttktjmt/mjswan) (Apache-2.0, `demo/LICENSE`), as vendored and patched in the FLoRe release; the idea of embedding it follows the [PHP](https://php-parkour.github.io/) page.
