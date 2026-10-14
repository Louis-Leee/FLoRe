## [1.9.0] - 2026-10-14
### Features
- The Controller section is rewritten around its law:
  - The heading is "Unified Force–Position Low-Level Controller with Compliant Behavior". The line
    under it, on the five keypoints, is gone.
  - The diagram of the unified reference generator now sits right under the two equations. Its
    labels carry the equations' subscript x: p*_x, p^tgt_x, f^cmd_x, f^meas_x, f^cmd_x/K and I_x/K.
    So do the formulas beside it.
  - The text under the diagram says what each symbol is, I_x included: the integral of the force
    error moves the target further in, or back, until the measured force balances the command.
  - The policy is "trained to reach p^tgt_x while it balances"; the clause about adding nothing
    to its joint commands is gone.
- *The integral at work*: the static press over time ("Soft contact and integrator") is replaced by
  a chart that runs the press in the browser, laid out as tune-to-learn lays out its gains.
  - The plot shows f^cmd_x (dashed), −f^meas_x and I_x in newtons over time, with the contact phase
    shaded.
  - The pad beside it sets K across and K_I up, both on log scales, with the paper's gains at its
    centre and marked by a ring.
  - The pad works by mouse and by touch, without scrolling the page, and by keyboard through two
    sliders that are out of sight. The focused slider lights its axis on the pad. On a phone the
    pad goes under the plot.
  - The approach and the retreat are named over the reference's own times. The shading marks the
    contact as the hysteresis detects it.
  - The caption says that −f^meas_x heads for f^cmd_x whatever K: faster for a higher K_I and a
    lower K. At a low K_I it is still on its way when the command ends.
  - A screen reader is told what the press comes to at the current gains: the force pressed as the
    command ends, and the integral.
- "Force tracking and compliance in one law" becomes *Compliant behavior*. It only says what
  compliance means: in contact, a measured force along an axis where nothing is commanded is
  driven to zero, as far as the integral's bound allows. So the hand tracks the force along the
  force direction and gives way along the others.
- *The external force* block is gone.
- The comparison with other learned controllers is boxed. It sets FLoRe's law beside FALCON's
  (p^tgt_x = p*_x) and the compliance law of FACET, SoftMimic and CHIP (p^tgt_x = p*_x + f^ext_x/K,
  at rest). It ends: only FLoRe measures the force at each wrist and integrates its error.
- The note above the in-page demo ("Try it below: …") is in smaller, muted type.
- The robot clips' titles are a size up, at 1.3rem.
- The hero's Code button links nowhere for now and reads "Code (coming soon)", as Paper and arXiv
  do.
### Design Rationale
- The chart's press is the one the old figure drew, ported from `draw_controller.py`. At four gains
  across the pad it gives the same samples to the last bit, and the same onset and release. It
  runs in well under a millisecond, so it runs again on every frame of a drag.
- Why K_I is the pad's second gain:
  - In this toy, K and the contact's stiffness act only through their ratio. A pad of K against
    the surface's stiffness would therefore give the same curves all along each diagonal.
  - K and K_I are the law's two gains, as K_p and K_d are tune-to-learn's.
  - Across the pad's span (K 100–400 N/m, K_I 0.125–2 s⁻¹) the contact never chatters on and off.
    The integral stays inside its bound.
- The force axis spans all that the pad can reach, so it stays put while the dot moves.
- The chart is SVG drawn at the plot's own pixel size, so its text stays legible on a phone. It
  takes its colours from the page's variables in either theme.
- The comparison drops UniFP: the request named FALCON, FACET, SoftMimic and CHIP.
- The curves' colours were checked with the dataviz palette validator:
  - With the page's f^meas teal, #0b7285, the cyan and green lines failed the normal-vision floor
    (ΔE 9.8, below 15).
  - Lines and arrows now take #0891b2 for f^meas, and #2f9e44 for the integral in the dark theme.
    The violet, cyan and green then pass every check in both themes.
  - Symbols in text keep the darker text colours: the cyan passes for marks (3.7:1) but not for
    text.
  - The generator's arrows take the same mark colours, so f^meas and I_x look alike in both
    figures.
- An inline formula and the punctuation after it are kept on one line: the browsers broke a line
  between them.
- The note above the demo is chosen in CSS (`.figlead:has(+ figure #demo)`), not by a new class.
  So the two reverts that embed the full demo again still stop on the same conflicts the README
  lists.
### Notes & Caveats
- With no script, the chart's place says it runs with JavaScript, and its legend is hidden.
- One pointer drags the dot at a time; a second finger is ignored until the first lets go.
- In Safari, Tab skips the pad's sliders unless the reader turns on full keyboard access, as for
  every range input.
- `draw_controller.py` now draws only the generator, and refuses to run if the press's markers are
  still in the page.
- On a phone, the Code button's longer label takes the buttons to a second row. There the note
  above them has a smaller margin, so the hero still fits a 360×640 screen. At 320×568 it
  overflowed before, and still does.
- The stylesheet is now `style.css?v=18`, and the script `site.js?v=11`.

## [1.8.1] - 2026-10-14
### Features
- The two robot clips are one to a row, at the column's full width, instead of side by side.
- A note under the Force Tracking title says what that clip shows:
  - F^cmd = −10 N along x;
  - the gauge reads the same force as the wrist's F/T sensor, F^meas;
  - when the external force grows or shrinks, the reading strays from 10 N, and the controller
    gradually brings it back.
### Design Rationale
- Side by side, each clip was 441 px wide, too small for the gauge to be read. One to a row, they
  are 900 px wide on a computer.
- The note sits between the title and the clip, as the page's captions sit above their figures, in
  smaller, muted type.
### Notes & Caveats
- On a phone the note takes five lines.
- The stylesheet is now `style.css?v=17`.

## [1.8.0] - 2026-10-14
### Features
- Under the pill to the full demo, the Controller section shows the controller on the robot in two
  clips side by side, each with its title over it:
  - "Compliant Behavior": the hands give way when they are pushed and pulled;
  - "Force Tracking (x direction −10 N)": a force gauge at the hand reads about 10 N.
### Design Rationale
- They are made and played as the Experiments clips are:
  - 1080p H.264 at CRF 24, without their sound;
  - muted, and looping while they are in view;
  - with the page's seek bar and full-screen button.
  On a phone they stack.
- Each title is its clip's caption, centred over it, bold, at the captions' size. It stays centred
  on a phone, where the other captions turn flush left.
- The raw clips stay in this folder, as `compliance.MP4` and `forcetracking.MP4`, and git ignores
  them, as it does `teaser/`. `compliance.MP4` alone is 105 MB, over GitHub's 100 MB limit.
### Notes & Caveats
- `compliance.mp4` is 15.6 MB, and `force_tracking.mp4` is 3.6 MB.
- The update of I wrote I_max with KaTeX's `\max`. That typesets an operator, so the MathML gave its
  subscript three parts, which Firefox reported as invalid markup. It is now `\mathrm{max}`, which
  looks the same on screen.
- The stylesheet is now `style.css?v=16`.
- The robot clips follow the pill. So the two reverts that embed the full demo again now stop on
  conflicts in `index.html`. The README says how to resolve them. Done that way, the switch back
  gives the demo block as it was before demo-x, and keeps the clips.

## [1.7.2] - 2026-10-14
### Features
- The data-collection arrow is drawn over everything, so the crate no longer hides it. It also lies
  a little lower and further left on the table, by 80 px and 40 px in the 1080p master.
- Chen Feng, Guanya Shi and Haozhi Qi carry a †. Beside "§Work done while interning at Amazon FAR",
  the note under the affiliations now reads "†Amazon FAR team co-lead".
### Design Rationale
- Before, the arrow was hidden wherever the crate stood in front of it. As the crate was dragged
  across the ring's front, that hid the arrowhead from 5.5 to 6.2 s. Drawn on top, the head stays
  in sight over the crate until the ring fades.
- Moved lower, the ring's front lies further out on the table, in front of the crate's foot.
- The page's Inter subset has no †. The system fonts' dagger, small in a superscript, read as a +.
  So the dagger is Inter's own, in a 2 KB file of its own (`static/fonts/inter-dagger.woff2`) that
  only U+2020 calls for.
- Each note is kept whole. On a computer the two share one line; on a phone each takes its own.
### Notes & Caveats
- The ring's start, at the crate's right end, is now drawn over the crate too, rather than coming
  out from behind it.
- The master and `data_collection.mp4?v=4` are rebuilt. Away from the arrow's frames, the master
  matches the previous one at 53 dB or more. The poster is unchanged.
- At 320×568 the hero's text was already taller than the hero, and is now about 10 px taller. From
  360×640 up it fits, in Chrome, WebKit and Firefox.
- `inter.css?v=2` and `style.css?v=15`.

## [1.7.1] - 2026-10-14
### Features
- In the data-collection clip, the arrow lies on the table and stays there. Before, it went along
  with the crate. It is laid round the crate where the crate stands when the arrow appears, and the
  crate is turned and dragged left off it. Its direction and timing are unchanged.
- The pill to the full demo reads "Try the full demo in a new tab!", and the line under it is gone.
### Design Rationale
- The ring is laid where the crate starts, not where it comes to rest. Laid at the rest place, its
  first part would show on the bare table beside the crate, pointing at it, just as the crate moves
  the other way. Laid at the start, the arrow begins as before, from behind the crate's right end.
- Lying on the table, the ring is hidden wherever it falls inside the crate's silhouette, since
  there the crate stands in front of it.
  - Before, only its far half could be hidden, which was right only while the ring went round the
    crate.
  - Now, as the crate is dragged across the ring's front, it hides the arrowhead from 5.5 to 6.2 s.
    The head then shows beside the crate until the ring fades.
- The ring moves with the table's drift, as the tapes' fills do: at most 1.6 px over its life.
- The pill's text may now shrink to 14 px. At 320 px the longer label took two lines at 16 px; it
  keeps to one line at every width in Chrome, WebKit and Firefox.
### Notes & Caveats
- The master and `data_collection.mp4?v=3` are rebuilt.
  - Away from the arrow's frames (4.78–6.85 s), the master matches the old one at 53 dB or more.
  - Inside them, only the two rings differ.
  - The poster has no arrow and is unchanged.
- The stylesheet is now `style.css?v=14`.
- The pill's change is a commit of its own. To embed the full demo again, revert it first, then the
  commit "the page's demo is demo-x; the full demo opens in its own tab". The second revert alone
  would conflict, since the line it removes is already gone.

## [1.7.0] - 2026-10-14
### Features
- The demo in the page is now a simpler build, `demo-x/`: the controller on the left hand, along x.
  - Its wall stands in front of the hand from the start.
  - Two sliders: F^cmd, from 0 to 20 N, and an extra push F^ext, at ±20 N.
  - Each arrow (purple F^cmd, blue F^meas, orange F^ext) carries its value along x, in newtons.
  - The panels say what the arrows and the switches are.
  - Only the standing motion is listed.
  - It has its own recorded preview, `controller_x.mp4`.
  - The "Try the demo in a new tab!" pill still opens the full demo, `demo/`, with both hands, all
    three axes and every motion, and a line under it says so.
- The Controller section explains the law in equations, in the manner of tune-to-learn's
  "Position Impedance Controllers and Gains":
  - the paper's Eq. 3 along the task direction x, with its terms labelled under braces (motion
    reference, force command, force feedback; contact phase, force error), followed by a "where …"
    list;
  - the same law as an admittance: force tracking along x, where a force is commanded, and
    compliance along z, where none is;
  - the external force f^ext and what the wrist sensor reads of it (Eq. 2);
  - compliance control, UniFP and FLoRe side by side, with links to FALCON, FACET, SoftMimic, CHIP
    and UniFP.
### Design Rationale
- demo-x is built by `scripts/build_demo_x.py` from cloned copies of the release's mjswan template
  and command term, which `scripts/demo_x.patch` changes. `demo/`, `build_demo.py` and the
  release's checkout are untouched (checked by hash and by `git status --ignored`), so the full
  demo stays as it was.
- The wall is seeded by the runtime at the first step of every episode, from the pelvis's x axis,
  levelled. The runtime publishes the plane, and the drawing shows that plane, centred where the
  hand meets it, instead of seeding a copy of its own.
- F^cmd stops at 20 N because the wall pushes back with at most k_env × 8 cm = 22.6 N. A command
  above that could not be tracked.
- The standing motion only: the wall stays where the hand started, as in training. Played with it,
  the forward walk and the three lifting clips pressed the hand into it with the whole 22.6 N, and
  the sideways walk with 9 N, all with no command.
- F^ext is left out of the simulated F/T reading, as it always was in the full demo. The page now
  says so: the sensor feels the wall and nothing else, so F^ext is
  a disturbance that the policy has to reject. With 15 N held, F^meas settled at −15.0 N, went to
  −14.0 N after a −10 N push and came back to −14.7 N, and came back to −15.0 N after +10 N.
- The labels are sprites drawn on a canvas. Each is placed past the tip of its arrow, in the
  arrow's direction on screen, and moved a row up or down so that labels never overlap. F^ext's
  arrow is drawn 5 cm below the hand point, where F^meas's does not hide it.
- The equations are typeset into the page once by `scripts/render_math.py`, with KaTeX served from
  `static/katex`. The page then needs no script, nothing has to be fetched from a CDN, and the text
  does not reflow once it has loaded. f^cmd, f^meas and I take the diagrams' colours, and f^ext
  takes the orange of the demo's push, in either theme.
- The comparison keeps to the paper's related work:
  - FALCON rejects the force as a disturbance;
  - FACET, SoftMimic and CHIP expose a stiffness;
  - UniFP's shift (f^cmd + f^ext)/K needs the true contact force and stiffness, and its end-effector
    force control is on a quadrupedal manipulator.
- The admittance reading is the law itself: (K/K_I) dΔx/dt = f^cmd + f^meas. On the robot it
  closes on all three axes, as the paper says.
### Notes & Caveats
- index.html is 144 KB, 17 KB gzipped, as KaTeX writes both HTML and MathML for each formula. Only
  four of its fonts are fetched.
- On a phone a wide equation scrolls sideways in its own box, as the diagrams do, and takes keyboard
  focus while it does.
- To embed the full demo again, revert the commit "the page's demo is demo-x; the full demo opens in
  its own tab". demo-x and its preview stay in the repository.
- The stylesheet is now `style.css?v=13`, the script `site.js?v=10`, and KaTeX's stylesheet
  `katex.min.css?v=0.19.0`.

## [1.6.1] - 2026-10-14
### Features
- The bar by each task's name in the teaser is Amazon orange (#ff9900) instead of the leaf green.
### Design Rationale
- It takes `--amazon`, the colour of the link to the demo's tab, so the page has one orange. The
  bar sits on the dark footage in either theme, so it needs no dark variant.
### Notes & Caveats
- The stylesheet is now `style.css?v=12`.

## [1.6.0] - 2026-10-13
### Features
- The hero follows the DoorMan page (doorman-humanoid.github.io). As the page is scrolled, the
  three teaser clips (Wipe Board, Rotate Box, Push Gantry) play frame by frame. Each plays under its
  task's name, set by a leaf-green bar, with a "Scroll to explore" hint at the foot.
- At the end of the teaser the side view of the gantry push fades in and plays by itself. Only then
  do the title, the authors and the links appear. Scrolling on, the page slides up over the hero as
  before.
- Over the hero the nav is its words alone, in white.
- The title ends in the rose of `rose.png`, on its transparent ground, instead of the 🌺. The
  favicon and the touch icon are still the 🌺.
- The data-collection clip no longer shows the green and the yellow strips of tape on the table.
- An orange arrow round the crate shows its turn in the data-collection clip:
  - it grows clockwise with the turn, from 4.95 to 5.87 s;
  - it follows the crate as the crate is dragged;
  - it has faded by 6.85 s.
### Design Rationale
- The teaser is a video sought by the scroll, not DoorMan's sequence of JPEGs: 56 s of footage would
  be hundreds of pictures. It is one H.264 file at 15 fps, with a keyframe every 8 frames and no
  frame predicted from a later one. It is 11.2 MB, and seeks in 3–5 ms in Chrome, WebKit and Firefox
  (median, with the file in memory, on an M2 Pro).
- The alternatives measured: every frame a keyframe was 21 MB and saved about 1 ms; a keyframe
  every 30 frames was 9.0 MB, and WebKit's backward seeks took 12 ms.
- The video is sought one seek at a time, each to the latest frame asked for. The frame sought eases
  after the scroll (time constant 0.12 s), so a wheel's notches play as motion. The task names
  follow the frame on screen.
- The run is 10.6 screens:
  - 9.4 for the teaser, about 6 s of footage a screen;
  - 0.6 for the fade to the title;
  - 0.6 for the title at rest.

  DoorMan's run is 10 screens. The Rotate Box clip keeps its 2× part, which halves the scrolling
  over the drag.
- The fade and the cover have no resting place: a scroll that stops in one goes on to the end it
  was last heading for. A link into the page, Back or Forward lands where it points. The page
  before this one snapped a landing at `#video`, 72 px inside the cover, onto the page, and in
  WebKit sent Back to `#video` up to the hero.
- The teaser is fetched only when the page rests with the hero at least half in view: after load,
  or on the way back up. A link into the page, or a scroll past the hero, fetches nothing. The side
  view (1.4 MB) is fetched once the teaser can play through.
- There is no run under reduced motion, without the script, or on paper. The title is over the side
  view's still from the start, and nothing is fetched. A teaser that fails to load leaves the same,
  with the side view playing.
- The build writes the teaser's frame rate and its shots' spans on its `<video>` in `index.html`
  (`data-rate`, `data-spans`). From them the script maps the scroll to frames and times the names.
  The spans are checked against the teaser's sources on every run, built or not, so a teaser and
  its page cannot part. The run's length is one CSS property (`--run`), which the script reads.
- How the tape comes off:
  - The table under each strip is redrawn once, from frames where nothing covers it, along the seams
    and edges that cross it.
  - Each frame then takes that drawing wherever the tape still shows. The drawing is corrected by
    what changed on the bare table around it, such as light and shadow.
  - The crate, the hands and the cables in front of the tape are kept.
- The result was checked against the source:
  - outside the edits, the master matches the source at 49.6–55.5 dB;
  - the fill changes less from frame to frame than the table beside it (0.22 against 0.62 levels).
### Notes & Caveats
- iOS Safari fetches no frame of a video that has not played. After 2 s without one, the page plays
  the teaser and pauses it at once. This could not be tried on an iPhone here, since Playwright's
  WebKit is the macOS one. Low Power Mode refuses even that, which leaves the poster with the names
  over it.
- In the data-collection clip:
  - from 5.2 to 6.3 s, a few 1–3 px specks remain at the foot of the motion-blurred crate where it
    crosses the old tape lines; they are visible only zoomed in and brightened;
  - about 0.17 s before the turn, the arrowhead peeks out from behind the crate's right end, as a
    cue.
- At 320×568 the hero's text is still taller than the hero, as before. On a phone on its side
  (844×390) the hero is still its minimum 600 px, so the title and the authors show but the
  affiliations and the links fall below the screen, as before.
- Printed, the hero is the title over the side view's still, from wherever the page was scrolled.
- New files:
  - `teaser.mp4` and `teaser.jpg`;
  - `rose.png`;
  - `scripts/clean_data_collection.py`, whose master, `supp/data_collection_clean.mp4`, stays in the
    paper's folder.
- Changed files:
  - `hero.mp4?v=4` is now the side view alone (1.4 MB), with the poster `hero.jpg?v=3`;
  - `data_collection.mp4?v=2` and its poster are rebuilt;
  - `style.css?v=11` and `site.js?v=9`.
- The `rose.png` beside the page stays out of git, like `teaser/`.

## [1.5.1] - 2026-10-13
### Features
- The Rotate Box teaser plays at 2× from 10.45 s, through the camera's circuit as well as the drag,
  instead of 1.6× from 20 s. The reel is 71 s instead of 79 s.
### Design Rationale
- The speed doubles where the camera's first swing round the robot comes to rest. At 10.0 s, the
  time asked for, the camera is still in that swing, and doubling it there would make it lurch;
  across 10.45 s its drift runs on unchanged, and only the robot moves faster.
- At 2× the reel keeps every other frame, an even cadence.
### Notes & Caveats
- The reel is 10.2 MB. Its poster and the link preview are unchanged; the reel is now
  `hero.mp4?v=3`.

## [1.5.0] - 2026-10-13
### Features
- The hero reel plays three teaser clips, one per task in the page's order (Wipe Board, Rotate
  Box, Push Gantry), and then the side view of the gantry push that was the whole reel before. The
  Rotate Box clip plays its drag, filmed from behind, at 1.6× from 20 s in.
- Sihang Li's name carries a §, and a line under the affiliations reads "§Work done while
  interning at Amazon FAR".
### Design Rationale
- One file rather than four played in turn: the hero's logic (fetched after the page has loaded,
  played only while the teaser is on screen, looped) stays as it was, and the file streams, so the
  first clip plays before the rest has arrived.
- The clips cross-fade over 0.6 s, as the old reel's loop did, and the loop still has no seam: the
  first clip's opening 0.6 s fade in over the side view's end, so the last frame leads into the
  first.
- Each clip is cut to a whole number of frames that its source is sure to fill, and the fades'
  offsets are sums of those counts, so every fade starts on a known frame.
- The speed changes at 20.0 s, where the camera has just come to rest behind the robot, so the
  change does not show; the faster part keeps 29.97 fps by dropping frames rather than blending
  them, and stays sharp.
- The link preview (`og.jpg`) is still the side view, now taken from its source: the reel's new
  first frame is a dark close-up of the wiping. The poster is that first frame, so the reel starts
  without a jump.
### Notes & Caveats
- The reel is 11 MB for 79 s, at the old reel's 1280×720 and CRF 28; the old one was 1.4 MB for
  15 s.
- The teaser clips (DJI Osmo Pocket 3, 200 MB, one over GitHub's 100 MB file limit) stay out of
  git, in `teaser/` beside the page; the build script reads them from there, not from the
  paper's folder, so that a checkout under another name finds its own.
- At 320×568 the hero's text was already taller than the hero, and its title slid under the nav;
  the new line adds 14 px to that. Every size from 360×640 up fits.
- `style.css?v=10`, and `?v=2` on the reel and its poster.

## [1.4.0] - 2026-10-13
### Features
- The page's emoji is 🌺 instead of 🌿: in the title, the SVG icon, and the PNG favicon and touch
  icon, made from Noto Emoji's U+1F33A at the same pinned tag.
- Every trial clip has a seek bar along its foot, a thin line that shows where the clip is. A click
  or a drag takes the clip there; once the bar has the focus, so do the arrow keys (2 s), Page Up
  and Page Down (10 s), Home and End, and Space or Enter plays or pauses. It is an ARIA slider
  that reads, for example, "0:05 of 0:31". A playing clip holds still while it is dragged and goes
  on from where it is let go.
- Each task ends with one rollout of the position-only baseline (Wipe Board failure 1, Rotate Box
  failure 1, Push Gantry failure 2), in a red card that says why it fails and what happens.
- The data-collection section ends with a Rotate Box demonstration recorded with the system.
- The narrated video no longer shows the venue and the submission number under its title: the
  title slide is replaced by the clip it plays, rendered again with the title alone, and the
  poster is taken from the new video. The README, the build script and the older entries here no
  longer name the venue either.
### Design Rationale
- A seek bar of its own rather than the browser's control row: the clips have no sound, so a
  volume button would mislead, and the row would cover much of a small tile. In full screen the
  browser's controls still come back.
- The bar is always shown, thin, so that a visitor sees the clip can be moved. Its strip is as tall
  as the full-screen button beside it, so a finger finds it. A finger seeks only once it has moved
  further sideways than up or down, by 6 px at least, or lifts (`touch-action: pan-y`): a vertical
  swipe over the bar still scrolls the page, and a gesture the page takes over puts the clip back.
  The bar follows one pointer at a time.
- While a drag seeks, it owns the playback: whatever starts the clip in the meantime, its turn to
  play on screen or its data arriving, waits until it is let go.
- A clip without its data needs it to show a frame, so a seek there plays it from the point picked;
  a key pressed before its length is known is kept and applied once it has loaded. The point picked
  replaces the one the clip saved when it let go of its download: the browsers refill a released
  clip from their cache, and the saved point would otherwise undo the seek at the next play.
- The baseline comes last in each task, after the plots, which stay next to the clips they belong
  to. Its clip is the size of one of the task's own: at full width the failure would be the
  largest picture of the task.
- The title clip is laid over the recording's first 597 frames, counted by number, rather than
  joined to the rest with concat. The recording's clock starts at 0.1 s; a concat starts at 0 and
  would put every later frame 0.1 s ahead of the narration and the subtitles. The new video has
  the old one's 5334 frames and timestamps, and every frame after the cut is within 40 dB PSNR of
  the old render. Should the title clip end before the cut, its last frame holds, so the old title
  cannot show through.
- The title is rendered with the settings of the clip that the slide plays (the robot in front of
  the text, the reveal at its centre, the text 356 px from the left) and without the line under
  it: the title keeps its place and the reveal its timing.
### Notes & Caveats
- The new clips' camera audio is dropped like the others'. No clip under Experiments or Data
  Collection has an audio track; only the narrated video has sound, its narration.
- The data-collection clip shows the demonstrator's face, as the narrated video already does; the
  photo above it keeps its emoji.
- The branch's history still holds the earlier video with the venue line; Pages serves only the
  current files.
- Python's `http.server` does not answer byte-range requests, so the clips cannot be sought behind
  it; the README now previews with `npx http-server`.
- `style.css?v=9`, `site.js?v=8`, and `?v=2` on the favicons and on the narrated video and its
  poster, so that browsers fetch the new files.

## [1.3.2] - 2026-10-08
### Features
- The unified reference generator's diagram puts the motion's reference p* on the surface, where
  position control brings the hand, and the force command moves the target from there into the
  surface. Before, p* sat in front of the surface, behind the hand.
- The press follows it: the reference brings the hand to the surface, holds it there while the
  command presses, and retreats as the command ramps down. The figure names these phases and calls
  itself an idealised press.
- Both captions say why the part is needed and then how it works, without numbers. The press marks
  where the contact phase switches as *onset* and *release* instead of in newtons; the generator no
  longer gives K, nor repeats its caption in a note.
### Design Rationale
- With p* on the surface, the force command alone makes the hand press, which is how the controller
  is used: the motion puts the hand on the surface and the command sets the force.
- In the toy, the surface stops the hand, which presses the harder the further its lagging target
  lies beyond the surface, at a contact stiffness (150 N/m) below K. The integral therefore moves
  the target further in, as the paper describes, and the generator, drawn to scale, shows the
  settled press. Left on the surface after the command, the hand would go on pressing on what the
  integral holds until it decayed, so the reference retreats during the ramp-down, and the figure
  says so.
### Notes & Caveats
- `style.css?v=8` only drops the rule of the press's old surface line.

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
- Content follows the anonymous project site in the same order, with the paper's abstract and
  task descriptions added; the layout, typography, scroll behaviour and clip handling follow the
  Workhorse page.
- `scripts/build_media.py` rebuilds every clip and picture from the paper's working folder.
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
- The supplementary video still showed the venue and the submission number on its title card, as
  recorded (removed in 1.4.0).
- `og:image` points at https://louis-leee.github.io/FLoRe/, the default Pages address of the repository.
