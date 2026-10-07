# FLoRe project page (`gh-pages`)

The project page of *FLoRe: Learning Force-Reactive Humanoid Loco-Manipulation*, served by
GitHub Pages from this branch. It is a static page with no build step for the HTML: `index.html`,
one stylesheet (`static/css/style.css`) and one script (`static/js/site.js`). The layout follows the
[Workhorse page](https://hybridrobotics.github.io/workhorse/).

## Preview locally

```bash
python3 -m http.server 8000      # in this folder, then open http://localhost:8000
```


## Rebuild the media

Every clip and picture in `static/videos` and `static/images` is made from the paper's working
folder by one script, which never writes into that folder:

```bash
python3 scripts/build_media.py --src ~/Desktop/ICRA2027              # fills in what is missing
python3 scripts/build_media.py --src ~/Desktop/ICRA2027 --only supp --force
```

It needs Pillow, and an ffmpeg with libass (Homebrew `ffmpeg-full`) for the subtitles
of the narrated video. The sources, encodings and the reason for each choice are listed at the top
of the script. All clips are single H.264 files (1080p trial clips at CRF 24, the narrated video at
720p CRF 20), so every browser plays them without a second codec.

## Rebuild the interactive demo

`demo/` is the static browser bundle of the 5-keypoint controller, built by the code release's own
`wbt-realtime-eval` (mjswan: MuJoCo and ONNX Runtime compiled to WebAssembly), then adapted to this
page by `scripts/build_demo.py`: the sub-path `/FLoRe/demo/`, a closer camera, the *F meas
inertial term* switch starting off, scene, policy and motion lists that cannot be emptied, and
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

## Credits

- Font: [Inter](https://github.com/rsms/inter), SIL Open Font License 1.1 (`static/fonts/OFL.txt`).
- Icon: the 🌿 of [Noto Emoji](https://github.com/googlefonts/noto-emoji) v2.047, Apache License 2.0.
- Page structure: adapted from the [Workhorse](https://hybridrobotics.github.io/workhorse/) project page.
- Interactive demo: [mjswan](https://github.com/ttktjmt/mjswan) (Apache-2.0, `demo/LICENSE`), as vendored and patched in the FLoRe release; the idea of embedding it follows the [PHP](https://php-parkour.github.io/) page.
