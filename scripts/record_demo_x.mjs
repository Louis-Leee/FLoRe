// Record the page's preview of the in-page demo (demo-x/) and write it as a master clip, which
// scripts/build_media.py then encodes for the page (static/videos/controller_x.mp4).
//
// The demo runs as the page embeds it (?embed), in headless Chrome at 1280x720: 15 N held, an
// extra push of -8 N and back, the command released (the wall alone), then 15 N again, so that
// the clip loops on the state it starts in.  Its frames come from the DevTools screencast and
// are stamped with the simulation's own clock (?probe), so that the clip plays the physics in
// real time however fast Chrome runs it; ffmpeg then resamples them to 30 fps.
//
// The bundle's base path is /FLoRe/demo-x/, so serve a folder in which FLoRe is this site.
// Node must find Playwright from the current folder (npm i playwright), and Chrome installed:
//
//   node scripts/record_demo_x.mjs http://localhost:8765/FLoRe/demo-x/ PAPER_DIR/supp/demo_x_preview.mp4
import { execFileSync } from 'node:child_process';
import { createRequire } from 'node:module';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const require = createRequire(path.join(process.cwd(), 'noop.js'));
const { chromium } = require('playwright');
const [url, master] = process.argv.slice(2);
if (!url || !master) throw new Error('usage: record_demo_x.mjs URL MASTER.mp4');
const ffmpeg = fs.existsSync('/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg') ? '/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg' : 'ffmpeg';
const frames = fs.mkdtempSync(path.join(os.tmpdir(), 'demo-x-'));
const CMD = 'wbt_force_cmd_local__l_fx';
const EXT = 'wbt_force_ext_local__l_fx';

const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--use-angle=metal'] });
const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
await page.goto(url + '?embed&probe');
await page.waitForSelector('#mjswan-slider-panel input[type=range]', { timeout: 120000 });
await page.waitForTimeout(3000);
const simTime = () => page.evaluate(() => window.__floreProbe.time);
const set = (key, v) =>
  page.evaluate(([key, v]) => {
    const label = [...document.querySelectorAll('#mjswan-slider-panel span')].find((s) => s.title.includes(`(key: ${key})`));
    const input = label.parentElement.querySelector('input[type=range]');
    input.value = String(v);
    input.dispatchEvent(new Event('input', { bubbles: true }));
  }, [key, v]);
const until = async (t) => {
  while ((await simTime()) < t) await page.waitForTimeout(20);
};
const hold = async (d) => until((await simTime()) + d);
// a slider from a to b over d seconds of simulated time, in the slider's steps of 0.5 N
const ramp = async (key, a, b, d) => {
  const t0 = await simTime();
  const n = Math.max(1, Math.round(Math.abs(b - a) / 0.5));
  for (let i = 1; i <= n; i++) {
    await until(t0 + (d * i) / n);
    await set(key, a + ((b - a) * i) / n);
  }
};

await ramp(CMD, 0, 15, 1.5); // the state the clip starts and ends in, settled
await hold(10);
const clock = []; // [wall ms, sim s]
const shots = []; // [file, wall ms]
const cdp = await page.context().newCDPSession(page);
cdp.on('Page.screencastFrame', ({ data, metadata, sessionId }) => {
  const file = path.join(frames, `f${String(shots.length).padStart(5, '0')}.jpg`);
  fs.writeFileSync(file, Buffer.from(data, 'base64'));
  shots.push([file, metadata.timestamp * 1000]);
  cdp.send('Page.screencastFrameAck', { sessionId }).catch(() => {});
});
const sampler = setInterval(() => simTime().then((t) => clock.push([Date.now(), t]), () => {}), 50);
await cdp.send('Page.startScreencast', { format: 'jpeg', quality: 92, maxWidth: 1280, maxHeight: 720 });
const s0 = await simTime();
await hold(2);
await ramp(EXT, 0, -8, 1);
await hold(6);
await ramp(EXT, -8, 0, 1);
await hold(4);
await ramp(CMD, 15, 0, 1.5);
await hold(3);
await ramp(CMD, 0, 15, 1.5);
await hold(6.5);
const s1 = await simTime();
await cdp.send('Page.stopScreencast');
clearInterval(sampler);
await page.waitForTimeout(300);
await browser.close();

// each frame at its simulated time, read off the clock samples by linear interpolation
const simAt = (ms) => {
  let i = clock.findIndex(([w]) => w >= ms);
  if (i < 0) i = clock.length - 1;
  if (i < 1) i = 1;
  const [w0, t0] = clock[i - 1];
  const [w1, t1] = clock[i];
  return t0 + ((t1 - t0) * (ms - w0)) / Math.max(1, w1 - w0);
};
const stamped = shots.map(([f, ms]) => [f, simAt(ms) - s0]).filter(([, t]) => t >= 0 && t <= s1 - s0);
const list = ['ffconcat version 1.0'];
stamped.forEach(([f, t], i) => {
  const next = i + 1 < stamped.length ? stamped[i + 1][1] : s1 - s0;
  list.push(`file '${f}'`, `duration ${Math.max(next - t, 0.001).toFixed(6)}`);
});
list.push(`file '${stamped.at(-1)[0]}'`); // the concat demuxer's last entry has no duration
fs.writeFileSync(path.join(frames, 'list.ffconcat'), list.join('\n') + '\n');
execFileSync(ffmpeg, ['-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', path.join(frames, 'list.ffconcat'),
  '-vf', 'fps=30,format=yuv420p', '-c:v', 'libx264', '-preset', 'slow', '-crf', '12', '-an', master]);
fs.rmSync(frames, { recursive: true, force: true });
console.log(`${stamped.length} frames, ${(s1 - s0).toFixed(1)} s of simulation -> ${master}`);
