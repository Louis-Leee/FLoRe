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

It needs Pillow and numpy, and an ffmpeg with libass (Homebrew `ffmpeg-full`) for the subtitles
of the narrated video. The sources, encodings and the reason for each choice are listed at the top
of the script. All clips are single H.264 files (1080p trial clips at CRF 24, the narrated video at
720p CRF 20), so every browser plays them without a second codec.

## Credits

- Font: [Inter](https://github.com/rsms/inter), SIL Open Font License 1.1 (`static/fonts/OFL.txt`).
- Icon: the 🌿 of [Noto Emoji](https://github.com/googlefonts/noto-emoji) v2.047, Apache License 2.0.
- Page structure: adapted from the [Workhorse](https://hybridrobotics.github.io/workhorse/) project page.
