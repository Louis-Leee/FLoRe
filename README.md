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
page by `scripts/build_demo.py` (sub-path `/FLoRe/demo/`, a closer camera, and
`scripts/demo_embed.js`, which lets the panels fold away). The page loads it into a frame only when
the visitor presses *Start*.

1. A FLoRe checkout with its submodule, and the release's browser-viz stack. On Linux that is
   `third_party/holosoma_extensions/scripts/_install_viz_stack.sh`. On macOS, install the four
   packages with `--no-deps` (`holosoma_inference` asks for the Linux-only `evdev`) and then add
   `mujoco==3.8.1 onnx onnxruntime numpy loguru nodeenv rich typer pydantic scipy tyro pyyaml termcolor
   defusedxml websockets filelock importlib_metadata netifaces sshkeyboard wandb pin pyzmq`.
2. The policy: `model_23000.onnx` of W&B run `far-wandb/Holosoma_sihangli/k7ai0nmv`
   (437 MB; the build strips it to the 38.8 MB obs→action graph).
3. The reference clips: any directory of holosoma clips. The page uses
   `umi3pt_push_gantry_new_ep025`, converted back from its qpos-only `.h2r` view with
   `python scripts/h2r_view_to_holosoma.py <view_dir> <scene.xml> <out_dir> umi3pt_push_gantry_new_ep025`
   (scene: holosoma's `scene_g1_29dof_wbt_plane_yam.xml`). `idle_default` is added by the builder.
4. `python scripts/build_demo.py --model-path model_23000.onnx --motion-dir <out_dir>` (with the
   viz-stack Python) rewrites `demo/`.

## Credits

- Font: [Inter](https://github.com/rsms/inter), SIL Open Font License 1.1 (`static/fonts/OFL.txt`).
- Icon: the 🌿 of [Noto Emoji](https://github.com/googlefonts/noto-emoji) v2.047, Apache License 2.0.
- Page structure: adapted from the [Workhorse](https://hybridrobotics.github.io/workhorse/) project page.
- Interactive demo: [mjswan](https://github.com/ttktjmt/mjswan) (Apache-2.0, `demo/LICENSE`), as vendored and patched in the FLoRe release; the idea of embedding it follows the [PHP](https://php-parkour.github.io/) page.
