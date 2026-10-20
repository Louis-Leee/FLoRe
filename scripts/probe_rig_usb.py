#!/usr/bin/env python3
"""Measure the rig's USB link quality — the A/B test for a cable change.

WHY this exists as a script and not a one-off: the question a cable change raises
("did it start dropping data, did latency get worse") is only answerable as a
DIFFERENCE, so the measurement has to be repeatable and printed in a fixed shape.
``docs/RUNBOOK_hl_serving_mc.md`` §"One hub carries all six devices" holds the
reference numbers this reproduces (40 s, ``--fps 30``, both D405s: 30.008 /
30.012 Hz, **0** drops of 1201, device dt 33.342 ms std 0.001).

Run it with the cable arrangement you have, then again with the old one, and diff
the two ``[verdict]`` blocks.

Three things are measured, and they answer three different questions:

1. **Enumeration / link training** — the sysfs tier path, negotiated ``speed`` and
   lane count of every pinned rig device. A marginal USB-3 cable most often does
   NOT corrupt data: it fails link training and the device falls back to 480 Mb/s,
   or an active extension inserts a hub tier. Both are visible here before any
   traffic flows, and both are catastrophic-but-silent (a D405 at 480 Mb/s still
   streams 640x480, just with a different frame budget).

2. **Sustained-load integrity** — both cameras streaming at once for N seconds,
   counting SDK ``frame_number`` holes. That count IS the drop count: the deploy
   bridges run ``--fps 30 --publish-hz 30`` (``decim=1``), so every device frame is
   published and a hole is a frame that never reached the host. A ``devnum`` change
   during the run is a re-enumeration, i.e. the link actually dropped — polled
   throughout by :class:`LinkWatcher` because the kernel log needs root here
   (``kernel.dmesg_restrict=1``).

3. **Latency** — per-frame ``time_of_arrival - frame_timestamp``. Its ABSOLUTE
   value is meaningless (two clock domains, unknown constant offset), so what is
   reported is the SPREAD and the tail: a cable that is retrying shows up as a
   fatter p99/max, not as a shifted mean. Propagation delay down a longer cable is
   ~5 ns/m and can never be the explanation for anything visible here.

``--stress`` additionally runs a high-bitrate pass (848x480 @ 60 on both cameras,
~4x the deploy bitrate). This is the useful part of a cable test: deploy uses ~6%
of the link, so a marginal cable can pass deploy and still be one warm afternoon
away from failing. The stress pass converts "it works" into a margin.

The F/T sensors and the CAN adapters are INVENTORIED but not loaded here — they
are Full-Speed devices on the hub's USB-2 half, a separate wire from the cameras
(so a camera result says nothing about them), and driving them means opening the
Bota driver / energising the jaws. See the runbook for those two.

Interpreter: the D405 part needs ``pyrealsense2``, which on this box lives in
``third_party/UMI_collection/.venv-vive`` (py3.9) — the same env the real camera
bridge runs in, deliberately, so the measurement is of the deploy path::

    third_party/UMI_collection/.venv-vive/bin/python scripts/probe_rig_usb.py \\
        --seconds 40 --label "two extension cables"

``--inventory-only`` needs no camera env at all and can run under FLoRe's ``.venv``.
"""

from __future__ import annotations

import argparse
import glob
import os
import statistics
import sys
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# The pinned rig devices. Same table as docs/RUNBOOK_hl_serving_mc.md §"The rig's
# devices, pinned" — kept here by (vid:pid, serial) rather than by sysfs path
# because the PATH is one of the things being measured. `expect_path` is what the
# runbook's verified run saw; a mismatch is not automatically wrong (ports get
# swapped) but a DEEPER path means a tier was added, which an active/powered
# extension does and a passive one does not.
# --------------------------------------------------------------------------- #
RIG_DEVICES: Tuple[Dict[str, object], ...] = (
    {"role": "D405 camera LEFT  (:8773)", "vidpid": "8086:0b5b", "serial": "323622270532",
     "asic": "404223070855", "expect_path": "2-1.4.2", "expect_speed": 5000},
    {"role": "D405 camera RIGHT (:8777)", "vidpid": "8086:0b5b", "serial": "353322271384",
     "asic": "402323071249", "expect_path": "2-1.4.3", "expect_speed": 5000},
    {"role": "Bota F/T LEFT     (:8771)", "vidpid": "0483:5740", "serial": "3870355D3532",
     "asic": None, "expect_path": None, "expect_speed": 12},
    {"role": "Bota F/T RIGHT    (:8771)", "vidpid": "0483:5740", "serial": "386B355D3532",
     "asic": None, "expect_path": None, "expect_speed": 12},
    {"role": "DM4310 jaw can0   (:8784)", "vidpid": "1d50:606f", "serial": None,
     "asic": None, "expect_path": "5-11.1.2", "expect_speed": 12},
    {"role": "DM4310 jaw can1   (:8785)", "vidpid": "1d50:606f", "serial": None,
     "asic": None, "expect_path": "5-11.1.3", "expect_speed": 12},
)

#: The gigabit NIC that SHARES the cameras' hub. Not a rig device, but it is on the
#: same SuperSpeed wire, so it is inventoried: saturating it is the one documented
#: un-measured interaction (runbook caveat 1).
NEIGHBOURS: Tuple[Tuple[str, str], ...] = (("0b95:1790", "AX88179 NIC on the cameras' hub"),)

SYSFS_USB = "/sys/bus/usb/devices"
CAM_SERIALS_DEFAULT = "323622270532,353322271384"


def _read(path: str) -> Optional[str]:
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return None


def _usb_nodes() -> List[Dict[str, Optional[str]]]:
    """Every enumerated USB device, as {path, vidpid, serial, speed, devnum, lanes}."""
    out: List[Dict[str, Optional[str]]] = []
    for entry in sorted(glob.glob(os.path.join(SYSFS_USB, "*"))):
        vid, pid = _read(os.path.join(entry, "idVendor")), _read(os.path.join(entry, "idProduct"))
        if not vid or not pid:
            continue  # an interface node (`2-1:1.0`), not a device
        out.append({
            "path": os.path.basename(entry),
            "vidpid": f"{vid}:{pid}",
            "serial": _read(os.path.join(entry, "serial")),
            "speed": _read(os.path.join(entry, "speed")),
            "devnum": _read(os.path.join(entry, "devnum")),
            "rx_lanes": _read(os.path.join(entry, "rx_lanes")),
            "tx_lanes": _read(os.path.join(entry, "tx_lanes")),
            "product": _read(os.path.join(entry, "product")),
        })
    return out


def _match(nodes: Sequence[Dict[str, Optional[str]]], spec: Dict[str, object]) -> List[Dict[str, Optional[str]]]:
    """Rig-device spec -> the sysfs nodes that are it.

    Matched on vid:pid plus serial when the spec pins one. The D405's USB ``serial``
    descriptor is its ASIC serial, NOT the ``rs-enumerate-devices`` module serial the
    launchers pin, so both are accepted — a firmware update changes neither, but only
    the module serial appears in the runbook table.
    """
    want_ser = spec.get("serial")
    want_asic = spec.get("asic")
    hits = []
    for n in nodes:
        if n["vidpid"] != spec["vidpid"]:
            continue
        if want_ser is None:
            hits.append(n)
        elif n["serial"] in (want_ser, want_asic):
            hits.append(n)
    return hits


def inventory(verbose: bool = True) -> Tuple[List[str], List[str]]:
    """Print the link table. Returns (findings, present_paths)."""
    nodes = _usb_nodes()
    findings: List[str] = []
    present: List[str] = []
    if verbose:
        print("── enumeration / link training ──────────────────────────────────────────")
        print(f"{'role':<26} {'sysfs':<10} {'speed':>7} {'lanes':>6} {'devnum':>7}  note")
    used: List[str] = []
    for spec in RIG_DEVICES:
        role = str(spec["role"])
        hits = [h for h in _match(nodes, spec) if h["path"] not in used]
        if not hits:
            if verbose:
                print(f"{role:<26} {'ABSENT':<10} {'-':>7} {'-':>6} {'-':>7}  not on the bus")
            findings.append(f"ABSENT: {role} (serial {spec['serial']}) is not enumerated")
            continue
        node = hits[0]
        used.append(str(node["path"]))
        present.append(str(node["path"]))
        speed = int(node["speed"] or 0)
        lanes = f"{node['rx_lanes'] or '?'}/{node['tx_lanes'] or '?'}"
        notes = []
        exp_speed = spec.get("expect_speed")
        if exp_speed is not None and speed < int(exp_speed):  # type: ignore[arg-type]
            notes.append(f"DOWNGRADED: {speed}M < expected {exp_speed}M — link training failed")
            findings.append(f"SPEED: {role} negotiated {speed}M, expected {exp_speed}M")
        exp_path = spec.get("expect_path")
        if exp_path and node["path"] != exp_path:
            depth_now = str(node["path"]).count(".")
            depth_ref = str(exp_path).count(".")
            if depth_now > depth_ref:
                notes.append(f"DEEPER than reference {exp_path} — a hub tier was ADDED")
                findings.append(f"TIER: {role} is at {node['path']}, reference {exp_path}")
            else:
                notes.append(f"moved ({exp_path} in the runbook) — port swap, same depth")
        if verbose:
            print(f"{role:<26} {str(node['path']):<10} {str(speed) + 'M':>7} {lanes:>6} "
                  f"{str(node['devnum']):>7}  {'; '.join(notes) or 'as referenced'}")
    if verbose:
        for vidpid, what in NEIGHBOURS:
            for n in nodes:
                if n["vidpid"] == vidpid:
                    print(f"{what:<26} {str(n['path']):<10} {str(n['speed']) + 'M':>7} "
                          f"{(n['rx_lanes'] or '?') + '/' + (n['tx_lanes'] or '?'):>6} {str(n['devnum']):>7}  neighbour")
    return findings, present


def _port_dirs() -> List[str]:
    """Every hub port's sysfs dir.

    A hub's ports hang off its INTERFACE node, as ``<hub>/<hub>:1.0/<hub>-portN``
    (root hubs use ``usbN-portM``) — not off the device node, and the device nodes
    in ``/sys/bus/usb/devices`` are symlinks, so a plain ``find`` without ``-L``
    finds nothing at all.
    """
    out: List[str] = []
    for iface in sorted(glob.glob(os.path.join(SYSFS_USB, "*:1.0"))):
        for pd in sorted(glob.glob(os.path.join(os.path.realpath(iface), "*-port*"))):
            if os.path.isdir(pd):
                out.append(pd)
    return out


def _oc_counts() -> Dict[str, int]:
    """path -> over_current_count, for the ports that report one.

    This is the kernel's own count of a hub cutting power to a port because the
    load exceeded what it could supply — the ONE direct, root-free measurement of
    the "the cable caused a power problem" hypothesis. Absent on hubs whose silicon
    has no per-port current sense (many cheap ones), which is why a zero here is
    necessary but not sufficient: see the power table in ``power_budget``.
    """
    out: Dict[str, int] = {}
    for pd in _port_dirs():
        v = _read(os.path.join(pd, "over_current_count"))
        if v is not None and v.isdigit():
            out[pd] = int(v)
    return out


def rig_subtrees() -> List[str]:
    """The top-most hub path on each bus that carries a rig device.

    Derived from where the rig devices are RIGHT NOW, not hardcoded, because the hub
    is the part that gets replaced. `2-1.4.2` -> `2-1`: the first path component pair
    is the hub plugged into the root, so everything under it is "the rig's hub tree",
    however many tiers deep it happens to be.

    Its second job is naming an UPLINK failure. A rig device vanishing is ambiguous
    (unplugged? died?), but one of THESE paths vanishing is not: nothing downstream of
    a hub can take the hub itself off the bus, so a hub in this set disappearing
    localises the fault to its upstream cable, connector or VBUS — and if BOTH halves
    of a USB-3 hub (the SuperSpeed `2-x` node and the USB-2 `5-y` node of one chip) go
    in the same poll, that is the chip losing power rather than a marginal SuperSpeed
    link, which would drop only the SuperSpeed half and keep streaming at 480 Mb/s.
    """
    nodes = _usb_nodes()
    tops = set()
    for spec in RIG_DEVICES:
        for n in _match(nodes, spec):
            path = str(n["path"])
            bus, _, rest = path.partition("-")
            if rest:
                tops.add(f"{bus}-{rest.split('.')[0]}")
    return sorted(tops)


def power_budget(verbose: bool = True) -> List[str]:
    """Print who asks the rig hub for how much current, and who claims to supply it.

    ``bMaxPower`` is a DECLARATION, not a measurement — no host can read actual draw
    — so this bounds the problem rather than settling it. What makes it worth
    printing: it says which device would brown out FIRST. If a longer cable were
    sagging the 5 V rail, the two 720 mA cameras under load are the most sensitive
    thing on it, so a flawless camera stress pass is evidence against a rail story
    that a 100 mA sensor's absence would otherwise suggest.
    """
    nodes = {n["path"]: n for n in _usb_nodes()}
    findings: List[str] = []
    rows: List[Tuple[str, int, str, str]] = []
    total = 0
    subtrees = rig_subtrees()
    for path, n in sorted(nodes.items()):
        # Everything downstream of the hub(s) the rig devices are ACTUALLY on, derived
        # rather than hardcoded: this table was `2-1.*` / `5-11.*` when it was written,
        # and the whole point of the tool is that the hub gets swapped. A hardcoded
        # prefix does not error when the hub moves — it prints an EMPTY budget under a
        # heading that claims to be the budget, which is worse than failing.
        if not any(path == s or path.startswith(s + ".") for s in subtrees):
            continue
        raw = (_read(os.path.join(SYSFS_USB, path, "bMaxPower")) or "0mA").replace("mA", "")
        ma = int(raw) if raw.isdigit() else 0
        attrs = _read(os.path.join(SYSFS_USB, path, "bmAttributes")) or "?"
        try:
            selfp = bool(int(attrs, 16) & 0x40)
        except ValueError:
            selfp = False
        rows.append((path, ma, "self" if selfp else "BUS", str(n["product"] or "")[:34].strip()))
        total += ma
    if verbose:
        print("\n── power: who asks the rig hub for what ────────────────────────────────")
        print(f"{'sysfs':<10} {'bMaxPower':>10} {'powered':>8}  product")
        for path, ma, pw, prod in rows:
            print(f"{path:<10} {str(ma) + ' mA':>10} {pw:>8}  {prod}")
        print(f"{'TOTAL':<10} {str(total) + ' mA':>10}  (declared, not measured)")
    oc = _oc_counts()
    hot = {k: v for k, v in oc.items() if v}
    if verbose:
        if not oc:
            print("  over-current: NO port reports a counter — this silicon has no per-port sense")
        elif hot:
            print(f"  over-current: {len(hot)} port(s) have tripped:")
            for k, v in sorted(hot.items()):
                print(f"      {v:>4} x  {os.path.basename(k)}")
        else:
            print(f"  over-current: 0 trips across {len(oc)} reporting ports")
    for k, v in sorted(hot.items()):
        findings.append(f"OVER-CURRENT: {os.path.basename(k)} has tripped {v} time(s)")
    return findings


def watch(seconds: float, period_s: float = 0.2) -> List[str]:
    """Log every USB connect / disconnect / re-enumeration until `seconds` elapse.

    This is the test for "the cable caused a power problem", because the hypothesis
    makes a specific prediction that a static inventory cannot check. Plug the
    suspect device in while this runs and read which of four shapes appears:

    * appears once and STAYS            -> it was simply unplugged; no power fault
    * appears and vanishes, repeatedly  -> enumeration is failing: power or signal
    * appears, and a NEIGHBOUR's devnum
      or speed changes at the same time -> the 5 V rail sagged; that IS the power story
    * nothing at all                    -> no VBUS or a dead connector; retry the
                                           same port with a known-good device to
                                           separate the port from the device

    Watches EVERY USB device, not just the pinned rig ones, precisely so the third
    shape is catchable — a rail sag shows up on whichever device is least tolerant,
    which is usually not the one being plugged in.
    """
    def snap() -> Dict[str, Tuple[str, str, str]]:
        return {str(n["path"]): (str(n["vidpid"]), str(n["serial"]), f"{n['devnum']}@{n['speed']}M")
                for n in _usb_nodes()}

    def label(path: str, val: Tuple[str, str, str]) -> str:
        vidpid, serial, dn = val
        prod = _read(os.path.join(SYSFS_USB, path, "product")) or ""
        return f"{path:<10} {vidpid} {dn:<10} {prod.strip()[:38]:<38} {serial or ''}"

    print(f"\n── live watch: {seconds:.0f} s, polling every {period_s * 1000:.0f} ms ──────────────────────")
    print("  plug the suspect device in NOW; every connect/disconnect/re-enumeration is logged")
    hubs = set(rig_subtrees())
    base, base_oc = snap(), _oc_counts()
    events: List[str] = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        time.sleep(period_s)
        now, now_oc = snap(), _oc_counts()
        gone = sorted(set(base) - set(now))
        # An UPLINK drop is the one event worth naming rather than listing: when a hub
        # that carries rig devices goes, everything under it goes in the SAME poll, so
        # the interesting fact ("the hub's own upstream link or VBUS failed") arrives as
        # a dozen indistinguishable REMOVED lines you have to read the tree to decode.
        # Collapse it, and say which half of the chip went — both halves at once is
        # power, the SuperSpeed half alone is a marginal SS link that falls back to 480.
        uplink = [p for p in gone if p in hubs]
        if uplink:
            line = (f"t+{time.monotonic() - t0:7.2f}s  !! UPLINK DROP  hub(s) {', '.join(uplink)} "
                    f"left the bus, taking {len(gone)} device(s) with them "
                    f"— fault is UPSTREAM of the hub (cable / connector / VBUS)")
            events.append(line)
            print(f"  {line}", flush=True)
        for path in sorted(set(base) | set(now)):
            was, isnow = base.get(path), now.get(path)
            if was == isnow:
                continue
            if uplink and path in gone:
                continue  # already collapsed into the UPLINK line above
            if was is None:
                line = f"t+{time.monotonic() - t0:7.2f}s  + ADDED    {label(path, isnow)}"
            elif isnow is None:
                line = f"t+{time.monotonic() - t0:7.2f}s  - REMOVED  {label(path, was)}"
            else:
                line = (f"t+{time.monotonic() - t0:7.2f}s  ~ CHANGED  {path:<10} "
                        f"{was[2]} -> {isnow[2]}")
            events.append(line)
            print(f"  {line}", flush=True)
        for pd, v in now_oc.items():
            if v > base_oc.get(pd, v):
                line = (f"t+{time.monotonic() - t0:7.2f}s  ! OVERCURR {os.path.basename(pd)} "
                        f"{base_oc.get(pd)} -> {v}")
                events.append(line)
                print(f"  {line}", flush=True)
        base, base_oc = now, now_oc
    if not events:
        print("  (no events)")
    return events


class LinkWatcher(threading.Thread):
    """Poll the rig devices' ``devnum``/``speed`` and record any change.

    A USB device that re-enumerates gets a NEW ``devnum``, so a change means the
    link actually dropped and came back — the single most important cable symptom,
    and the one that is otherwise invisible because ``kernel.dmesg_restrict=1``
    makes the kernel log unreadable without root. Speed is watched too: a device
    can re-train DOWN without re-enumerating.
    """

    def __init__(self, period_s: float = 0.25) -> None:
        super().__init__(daemon=True)
        self.period_s = period_s
        self.events: List[str] = []
        self.polls = 0
        self._halt = threading.Event()
        self._base = self._snapshot()

    @staticmethod
    def _snapshot() -> Dict[str, Tuple[Optional[str], Optional[str]]]:
        nodes = _usb_nodes()
        snap: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
        # The rig devices AND the hubs carrying them: a hub dropping is what takes every
        # device with it, so watching only the leaves reports six simultaneous failures
        # and not the one cause. `rig_subtrees` names the hubs; hub vid:pids are not in
        # RIG_DEVICES and must not be, or swapping in a different hub brand stops the
        # watch from seeing the uplink at all.
        wanted = {str(s["vidpid"]) for s in RIG_DEVICES}
        hubs = set(rig_subtrees())
        for n in nodes:
            path = str(n["path"])
            if n["vidpid"] in wanted or path in hubs:
                snap[path] = (n["devnum"], n["speed"])
        return snap

    def run(self) -> None:
        t0 = time.monotonic()
        while not self._halt.wait(self.period_s):
            self.polls += 1
            now = self._snapshot()
            for path in sorted(set(self._base) | set(now)):
                was, isnow = self._base.get(path), now.get(path)
                if was != isnow:
                    line = f"t+{time.monotonic() - t0:6.2f}s  {path}: {was} -> {isnow}"
                    self.events.append(line)
                    # Printed as it happens, not only in the summary: this runs for the
                    # whole load test, and on a run measured in tens of minutes a drop
                    # that only surfaces at the end leaves you unable to tell whether it
                    # coincided with something you did in the room.
                    print(f"  !! LINK CHANGE  {line}", flush=True)
            self._base = now

    def stop(self) -> None:
        self._halt.set()
        self.join(timeout=2.0)


def _stats(xs: Sequence[float]) -> Dict[str, float]:
    if not xs:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "p99": float("nan"), "max": float("nan")}
    s = sorted(xs)
    return {
        "n": len(xs),
        "mean": statistics.fmean(xs),
        "std": statistics.pstdev(xs) if len(xs) > 1 else 0.0,
        "p99": s[min(len(s) - 1, int(0.99 * len(s)))],
        "max": s[-1],
    }


class CameraRun(threading.Thread):
    """Stream one D405 and account for every frame it delivers.

    One thread per camera rather than one process (the deploy bridges are separate
    processes) because ``wait_for_frames`` releases the GIL, so two threads do not
    serialise — and running them in ONE process is what makes the two cameras
    provably concurrent on the shared hub, which is the condition being tested.
    """

    def __init__(self, rs, serial: str, width: int, height: int, fps: int, seconds: float) -> None:
        super().__init__(daemon=True)
        self.rs, self.serial = rs, serial
        self.width, self.height, self.fps, self.seconds = width, height, fps, seconds
        self.frame_numbers: List[int] = []
        self.host_t: List[float] = []
        self.dev_ts: List[float] = []
        self.latency_ms: List[float] = []
        self.error: Optional[str] = None
        self.md_used = "none"

    def run(self) -> None:
        rs = self.rs
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        try:
            pipe.start(cfg)
        except Exception as exc:  # noqa: BLE001 — surfaced in the report, not raised into the harness
            self.error = f"start failed: {exc}"
            return
        try:
            deadline = time.monotonic() + self.seconds
            # Discard the first 15 frames: auto-exposure convergence makes the opening
            # device dt genuinely non-uniform, which would pollute the jitter stats with
            # something that is not a link property.
            warm = 15
            while time.monotonic() < deadline:
                try:
                    fs = pipe.wait_for_frames(timeout_ms=2000)
                except Exception as exc:  # noqa: BLE001
                    self.error = f"wait_for_frames: {exc}"
                    break
                f = fs.get_color_frame()
                if not f:
                    continue
                if warm > 0:
                    warm -= 1
                    continue
                self.host_t.append(time.monotonic())
                self.frame_numbers.append(int(f.get_frame_number()))
                dev_us = self._md(f, "frame_timestamp")
                if dev_us is None:
                    self.dev_ts.append(float(f.get_timestamp()) * 1e3)  # ms -> us
                    self.md_used = "get_timestamp (ms)"
                else:
                    self.dev_ts.append(float(dev_us))
                    self.md_used = "frame_timestamp (us)"
                arrival_ms = self._md(f, "time_of_arrival")
                if arrival_ms is not None and dev_us is not None:
                    self.latency_ms.append(float(arrival_ms) - float(dev_us) / 1e3)
        finally:
            try:
                pipe.stop()
            except Exception:  # noqa: BLE001,S110 — teardown of a already-dead pipeline
                pass

    def _md(self, frame, name: str) -> Optional[float]:
        key = getattr(self.rs.frame_metadata_value, name, None)
        if key is None:
            return None
        try:
            if not frame.supports_frame_metadata(key):
                return None
            return float(frame.get_frame_metadata(key))
        except Exception:  # noqa: BLE001
            return None

    def report(self, label: str) -> List[str]:
        if self.error:
            return [f"  {label} {self.serial}: FAILED — {self.error}"]
        n = len(self.frame_numbers)
        if n < 2:
            return [f"  {label} {self.serial}: only {n} frames — nothing to measure"]
        span = self.host_t[-1] - self.host_t[0]
        rate = (n - 1) / span if span > 0 else float("nan")
        gaps = [b - a for a, b in zip(self.frame_numbers, self.frame_numbers[1:])]
        drops = sum(g - 1 for g in gaps if g > 1)
        hist: Dict[int, int] = {}
        for g in gaps:
            hist[g] = hist.get(g, 0) + 1
        hist_s = ", ".join(f"{k}: {100.0 * v / len(gaps):.1f}%" for k, v in sorted(hist.items()))
        dev_dt = _stats([(b - a) / 1e3 for a, b in zip(self.dev_ts, self.dev_ts[1:])])
        host_dt = _stats([1e3 * (b - a) for a, b in zip(self.host_t, self.host_t[1:])])
        lat = _stats(self.latency_ms)
        lines = [
            f"  {label} {self.serial}: {n} frames in {span:.2f} s = {rate:.3f} Hz, "
            f"dropped {drops} / {n + drops}",
            f"      device dt  {dev_dt['mean']:8.3f} ms  std {dev_dt['std']:6.3f}  "
            f"p99 {dev_dt['p99']:8.3f}  max {dev_dt['max']:8.3f}   [{self.md_used}]",
            f"      host   dt  {host_dt['mean']:8.3f} ms  std {host_dt['std']:6.3f}  "
            f"p99 {host_dt['p99']:8.3f}  max {host_dt['max']:8.3f}",
            f"      frame_number gaps  {{{hist_s}}}",
        ]
        if lat["n"] > 1:
            # SUCCESSIVE differences, not the spread about the mean. The device clock and the
            # host clock free-run against each other (measured ~18 ppm on these D405s), so the
            # raw spread grows LINEARLY with run length — 0.78 ms over 40 s but 3.3 ms over
            # 180 s on an identical, drop-free link. That is drift, not transport jitter, and
            # reporting it would make a longer A/B run look like a worse cable. Differencing
            # consecutive frames cancels any constant rate offset, leaving only the per-frame
            # arrival noise a retrying link would inflate.
            jit = _stats([abs(b - a) for a, b in zip(self.latency_ms, self.latency_ms[1:])])
            lines.append(
                f"      sensor->host  jitter |d| mean {jit['mean']:6.3f} ms  p99 {jit['p99']:6.3f}  "
                f"max {jit['max']:6.3f}   (drift-immune; raw spread {lat['std']:.3f} ms is mostly "
                f"clock drift over {span:.0f} s)"
            )
        else:
            lines.append("      sensor->host  unavailable (no time_of_arrival/frame_timestamp metadata)")
        return lines


def measure_cameras(serials: Sequence[str], width: int, height: int, fps: int, seconds: float,
                    label: str) -> Tuple[List[str], List[str]]:
    """Stream every camera concurrently for `seconds`. Returns (report_lines, findings)."""
    try:
        import pyrealsense2 as rs  # noqa: PLC0415 — optional; --inventory-only must not need it
    except ImportError:
        return (["  pyrealsense2 not importable in this interpreter — see the module docstring "
                 "(use third_party/UMI_collection/.venv-vive/bin/python), or pass --inventory-only"],
                ["SKIPPED: camera load test (no pyrealsense2)"])
    runs = [CameraRun(rs, s, width, height, fps, seconds) for s in serials]
    print(f"  streaming {len(runs)} x {width}x{height} rgb8 @ {fps} for {seconds:.0f} s ...", flush=True)
    for r in runs:
        r.start()
    for r in runs:
        r.join(timeout=seconds + 30.0)
    lines: List[str] = []
    findings: List[str] = []
    for r in runs:
        lines.extend(r.report(label))
        if r.error:
            findings.append(f"CAMERA: {r.serial} {r.error}")
            continue
        gaps = [b - a for a, b in zip(r.frame_numbers, r.frame_numbers[1:])]
        drops = sum(g - 1 for g in gaps if g > 1)
        if drops:
            findings.append(f"DROPS: {r.serial} lost {drops} frames at {label}")
    return lines, findings


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=40.0, help="load-test duration (runbook reference: 40).")
    ap.add_argument("--cameras", default=CAM_SERIALS_DEFAULT,
                    help="comma-separated D405 module serials (default = the two ROBOT-mounted ones).")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30, help="deploy rate; 30 with --publish-hz 30 means decim=1.")
    ap.add_argument("--stress", action="store_true",
                    help="ALSO run a high-bitrate pass (--stress-* below) to measure margin, not just pass/fail.")
    ap.add_argument("--stress-width", type=int, default=848)
    ap.add_argument("--stress-height", type=int, default=480)
    ap.add_argument("--stress-fps", type=int, default=60)
    ap.add_argument("--stress-seconds", type=float, default=20.0)
    ap.add_argument("--inventory-only", action="store_true", help="link table only; no streaming, no camera env.")
    ap.add_argument("--watch", type=float, default=0.0, metavar="SECONDS",
                    help="log every USB connect/disconnect/re-enumeration for SECONDS, then exit. "
                         "Plug the suspect device in while it runs — this is the power/contact test.")
    ap.add_argument("--label", default="", help="free text recorded in the header, e.g. \"two extension cables\".")
    args = ap.parse_args(argv)

    print("=" * 78)
    print(f"rig USB probe   {time.strftime('%Y-%m-%d %H:%M:%S')}   label: {args.label or '(none)'}")
    print("=" * 78)
    findings, _ = inventory()
    findings += power_budget()

    if args.watch > 0:
        events = watch(args.watch)
        print(f"\n[verdict]  {len(events)} USB event(s) in {args.watch:.0f} s")
        for f in findings:
            print(f"  ✗ {f}")
        return 0

    if args.inventory_only:
        _verdict(findings, watcher=None)
        return 0

    serials = [s.strip() for s in args.cameras.split(",") if s.strip()]
    watcher = LinkWatcher()
    watcher.start()
    try:
        print("\n── sustained load: deploy settings ─────────────────────────────────────")
        lines, f2 = measure_cameras(serials, args.width, args.height, args.fps, args.seconds,
                                    f"{args.width}x{args.height}@{args.fps}")
        print("\n".join(lines))
        findings += f2
        if args.stress:
            print("\n── sustained load: STRESS (margin, not deploy) ─────────────────────────")
            mult = ((args.stress_width * args.stress_height * args.stress_fps)
                    / float(args.width * args.height * args.fps))
            print(f"  {mult:.1f}x the deploy bitrate")
            lines, f3 = measure_cameras(serials, args.stress_width, args.stress_height, args.stress_fps,
                                        args.stress_seconds,
                                        f"{args.stress_width}x{args.stress_height}@{args.stress_fps}")
            print("\n".join(lines))
            findings += [f"{x} (STRESS)" for x in f3]
    finally:
        watcher.stop()
    _verdict(findings, watcher)
    return 0


def _verdict(findings: Sequence[str], watcher: Optional[LinkWatcher]) -> None:
    print("\n── re-enumeration watch ────────────────────────────────────────────────")
    if watcher is None:
        print("  not run (--inventory-only)")
    elif watcher.events:
        print(f"  {len(watcher.events)} LINK CHANGE(S) over {watcher.polls} polls — the link dropped:")
        for e in watcher.events:
            print(f"    {e}")
    else:
        print(f"  0 changes over {watcher.polls} polls: no device re-enumerated or re-trained.")

    print("\n[verdict]")
    if watcher is not None and watcher.events:
        print(f"  ✗ {len(watcher.events)} link change(s) during the run")
    hard = [f for f in findings if not f.startswith("SKIPPED")]
    if not hard:
        print("  ✓ every pinned device present at its expected speed; no drops; no re-enumeration")
    for f in findings:
        print(f"  {'-' if f.startswith('SKIPPED') else '✗'} {f}")


if __name__ == "__main__":
    sys.exit(main())
