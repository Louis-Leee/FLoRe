"""OFFLINE samples driving a REAL low level, with the mjswan page as the control surface.

The missing half of a two-sided pair the repo already had::

    scripts/infer_s1_wholebody.py   dataset sample -> chunk -> a viser DRAWING.  No controller.
    scripts/hl_serving_wholebody.py live sensors   -> chunk -> the :8770 wire.   No dataset.
    THIS MODULE (+ scripts/hl_offline_drive.py)
                                    dataset sample -> chunk -> the :8770 wire, and the wire's
                                    OTHER end is the mjswan MuJoCo/WASM low level in a browser.

WHY IT IS A THIRD THING AND NOT A FLAG ON EITHER
================================================
The offline session's question is "is this chunk right", answered against a stored target. The
deploy planner's question is "does the robot do it", answered by a controller. Neither can ask
the other's: a drawing has no controller, and a controller fed live sensors has no ground truth
and no repeatability. This path asks the ONE question that needs both — *what does the low level
actually do with a chunk we already know is good?* — and it is repeatable, because the
observation is a stored sample rather than whatever the cameras saw.

That also makes it the cheapest rehearsal that exists for the deploy path: no rig, no trackers,
no Botas, no cameras, no gantry. What it does NOT rehearse is the observation pipeline (the
sample bypasses every bridge) and contact (a browser sim's objects are not the task's).

THE CONTROL SURFACE IS THE PAGE, NOT A SECOND WINDOW
====================================================
``S0Driver`` binds ``:8770`` as the SERVER and the mjswan page is its client, so the page's
existing ``sendCmd`` vocabulary is already a channel back — the same one
``vive/mc_replay_bridge.py``'s episode picker uses. :func:`apply_control` is that protocol, and
it is module-level and PURE for the same reason that one is: an out-of-range index arriving from
a click in a browser must be refused by something a test can call without a socket.

The panel is gated on :data:`PAGE_GATE_KEY` — a wire key nothing but this planner ever sends —
so the rig's page and every other preset's page are untouched by construction, rather than by a
build flag someone has to remember.

TWO FAMILIES, ONE SCRIPT, chosen by what the checkpoint predicts::

    diffusion_umi_wholebody   29 joints  -> DenseWire      -> mode="dense"  (MODE=dense-unified-anneal)
    diffusion_umi_3tracker    3 EE poses -> ThreePointWire -> mode="3pt"    (MODE=3pt-unified-anneal)

The 3-tracker route is DIRECT — no motion completion — and that is the point of it: a
3-keypoint policy and a 3-keypoint low level want the same numbers, so inserting a completion
(as ``scripts/hl_serving_mc.py`` must, for the 5pt/dense families) would add a second model, a
second interpreter and the completion's own error to a handoff that has none. See
:class:`ThreePointWire`.

WHAT THE PAGE DRAWS, per family, and why they differ
====================================================
* **wholebody** — the violet body, free. The page has drawn a LIGHT-PURPLE translucent G1 from
  the wire's ``mc_qpos``/``mc_joint_names``/``mc_root_*`` keys since the completion ghost was
  added, keyed BY JOINT NAME, and a wholebody chunk IS 29 joint angles. So
  :func:`ghost_extras` publishes the staged chunk one step per heartbeat tick and the existing
  page code animates it. No new drawing code, and the ghost is a SEPARATE channel from the
  command — which is what makes "preview" mean preview rather than "publish".
* **3pt** — no body, and that is a statement about the layout rather than a gap. A keypoint
  policy predicts three poses and no posture; the page's own commanded-target triads (magenta L
  TCP / yellow R TCP / green head) are what the wire moves, and they move when the chunk RUNS.
  Before that the panel reports the chunk numerically (per-role excursion, the gate prediction).
  Drawing a body there would mean completing the chunk, i.e. answering with a second model's
  guess about a posture the policy never claimed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from flore.s1_policy.serving.hl_dense import (
    FIVE_POINT_LABELS,
    KEYPOINT_GATE_TOL_M,
    KeypointClutch,
    PostAnchorConfig,
    is_safe_keypoint_command,
    keypoint_gaps_m,
)

#: The wire key that ARMS the page's FLoRe panel. Positive ⇒ "an offline FLoRe driver is on the
#: other end of this socket". Nothing else in the ecosystem sends it: the live VIVE bridge, the
#: MC replay bridge and all three deploy planners publish poses and (at most) the ``episode``
#: family, so every other page never grows the panel and needs no build flag to say so.
PAGE_GATE_KEY = "flore_n_samples"

#: The control verbs the page may send, as ``{"cmd": "flore", "action": <verb>, ...}``.
#:
#: ``predict``  sample one chunk at the cursor and STAGE it (nothing reaches the wire).
#: ``run``      drip the staged chunk onto the wire, one step per 30 Hz tick.
#: ``stop``     interrupt a running drip within one step, and retire an auto loop.
#: ``reset``    clutch + wire + ghost back to the cursor's hold pose. Also drops the staging.
#: ``sample``   move the cursor to ``index`` (absolute dataset anchor). Implies a reset.
#: ``episode``  move the cursor to episode ``index``'s FIRST anchor. Implies a reset.
#: ``auto``     predict → run, repeatedly, until ``stop`` / the wall-clock ceiling.
CONTROL_ACTIONS = ("predict", "run", "stop", "reset", "sample", "episode", "auto")

#: ``cmd`` value this planner answers to. Namespaced so the page can keep the MC replay
#: bridge's ``{"cmd":"episode"}`` / ``{"cmd":"recalibrate"}`` verbs meaning what they mean
#: there — one socket, two vocabularies, and no verb that means different things to different
#: producers.
CONTROL_CMD = "flore"

#: Planner lifecycle, reported to the page as ``flore_state``. A single string rather than a set
#: of booleans because the panel's job is to say which of the four buttons will do something.
STATES = ("hold", "predicting", "staged", "running", "stopped")


# ---------------------------------------------------------------------------
# the control protocol
# ---------------------------------------------------------------------------
def apply_control(state: dict, msg: Any) -> str:
    """Apply one page control message to ``state``; return the line to log (``""`` = silent).

    BOOKKEEPING ONLY, and that is load-bearing rather than tidy. This runs on the WebSocket
    event loop, which also owns the 200 Hz broadcaster: sampling a chunk here (~200-600 ms of
    GPU) would stop the wire for that long, and the low level's 0.25 s stale gate would fire
    mid-command. So every verb sets a REQUEST and the planner's own thread acts on it, exactly
    the way ``mc_replay_bridge.apply_control`` queues an episode switch rather than loading it.

    ``state`` keys read: ``n_samples``, ``n_episodes``, ``sample``. Keys written: ``req``
    (a verb, latest-wins), ``want_sample`` (an absolute anchor index), ``stop`` (a latch).

    ⚠️ ``stop`` is a LATCH set here and cleared by the planner, never the reverse. It is the one
    verb whose whole value is that it cannot be lost: a Stop that arrived while the planner was
    between steps must still end the motion, so it may not be overwritten by a later ``req``.
    """
    if not isinstance(msg, dict):
        return f"ignoring non-object control message {msg!r}"
    cmd = str(msg.get("cmd") or "")
    if cmd == "recalibrate":
        # The page's own Recalibrate button, which exists on every live obs class and cannot be
        # hidden from here. A replay has no operator pose to re-capture, so the honest equivalent
        # is a reset — the same reading ``mc_replay_bridge`` gives it (there: restart the take).
        # Mapped rather than ignored because a button that silently does nothing is worse than
        # either behaviour.
        state["req"] = "reset"
        return "<- recalibrate: treated as reset"
    if cmd != CONTROL_CMD:
        return ""      # someone else's verb on a shared socket — silent, not an error
    action = str(msg.get("action") or "")
    if action not in CONTROL_ACTIONS:
        return f"ignoring unknown action {action!r} (expected one of {list(CONTROL_ACTIONS)})"

    if action in ("sample", "episode"):
        try:
            idx = int(msg["index"])
        except (KeyError, TypeError, ValueError):
            return f"ignoring malformed {action} request {msg!r}"
        if action == "sample":
            n = int(state["n_samples"])
            if not 0 <= idx < n:
                return f"ignoring sample {idx}: this dataset has {n} anchors (0..{n - 1})"
            state["want_sample"] = idx
            state["req"] = "sample"
            return f"<- sample {idx} requested"
        n_ep = int(state["n_episodes"])
        if not 0 <= idx < n_ep:
            return f"ignoring episode {idx}: this dataset has {n_ep} (0..{n_ep - 1})"
        state["want_episode"] = idx
        state["req"] = "episode"
        return f"<- episode {idx} requested"

    if action == "stop":
        # Set, never cleared here. See the docstring: losing a Stop is the one failure this
        # protocol must not have, so it does not go through `req` (latest-wins) at all.
        state["stop"] = True
        state["req"] = "stop"
        return "<- STOP"

    state["req"] = action
    return f"<- {action}"


# ---------------------------------------------------------------------------
# the cursor
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CursorAt:
    """Where the cursor is, in every spelling the page and the log need."""

    index: int                # absolute HLDataset anchor index
    episode: int              # global episode id
    episode_frame: int        # frames into that episode
    episode_index: int        # position of `episode` in the episode list (what the page picks)
    episode_frames: int       # how many ANCHORS this episode has
    episode_offset: int       # position of `index` within this episode's anchors
    split: str                # "train" | "validation"

    def describe(self) -> str:
        return (f"anchor {self.index} = episode {self.episode} frame {self.episode_frame} "
                f"({self.episode_offset + 1}/{self.episode_frames} of that episode's anchors, "
                f"{self.split})")


class SampleCursor:
    """A movable position in an :class:`~flore.s1_policy.inference.common.HLSampleSource`.

    Two coordinate systems, because the page needs both and they are NOT interchangeable:

    * the **anchor index** is what the inference session takes, and it is dense over the whole
      dataset (30k+ of them on a merged session);
    * the **episode** is what an operator thinks in ("show me take 7"), and it is what makes a
      picker usable at all.

    ⚠️ An episode's anchors are NOT a contiguous index range in general — ``drop_glitch_windows``
    and the force/validity filters remove individual anchors, so an episode can be a set with
    holes. This class therefore GROUPS the refs instead of assuming ``[start, start + n)``, which
    an earlier arithmetic version got wrong in exactly the places a QC'd dataset is interesting.
    """

    def __init__(self, refs: Sequence[Any]) -> None:
        if not refs:
            raise ValueError("the dataset has no usable anchors, so there is nothing to drive")
        self._refs = tuple(refs)
        # episode id -> the anchor indices in it, in dataset order.
        by_ep: dict[int, list[int]] = {}
        for i, ref in enumerate(self._refs):
            by_ep.setdefault(int(ref.global_episode_id), []).append(i)
        self._episodes = tuple(sorted(by_ep))
        self._by_ep = {ep: tuple(v) for ep, v in by_ep.items()}
        self._ep_pos = {ep: k for k, ep in enumerate(self._episodes)}
        self._index = 0

    # --- geometry --------------------------------------------------------
    def __len__(self) -> int:
        return len(self._refs)

    @property
    def n_episodes(self) -> int:
        return len(self._episodes)

    @property
    def index(self) -> int:
        return self._index

    @property
    def ref(self) -> Any:
        return self._refs[self._index]

    def at(self) -> CursorAt:
        ref = self.ref
        ep = int(ref.global_episode_id)
        anchors = self._by_ep[ep]
        return CursorAt(
            index=self._index,
            episode=ep,
            episode_frame=int(ref.episode_frame),
            episode_index=self._ep_pos[ep],
            episode_frames=len(anchors),
            episode_offset=anchors.index(self._index),
            split=str(ref.split),
        )

    # --- movement --------------------------------------------------------
    def set_index(self, index: int) -> bool:
        """Move to an absolute anchor. ``False`` (and no move) if out of range or unchanged."""
        idx = int(index)
        if not 0 <= idx < len(self._refs) or idx == self._index:
            return False
        self._index = idx
        return True

    def set_episode(self, episode_index: int) -> bool:
        """Move to the FIRST anchor of the ``episode_index``-th episode.

        The first ANCHOR, not the episode's first frame: a frame the filters dropped is not a
        sample and cannot be inferred from. Landing on the nearest usable one is the only
        behaviour that makes the picker's every entry work.
        """
        k = int(episode_index)
        if not 0 <= k < len(self._episodes):
            return False
        return self.set_index(self._by_ep[self._episodes[k]][0])

    def step(self, delta: int) -> bool:
        """Move ``delta`` anchors, CLAMPED to the dataset (so ◀ at 0 is a no-op, not a wrap)."""
        return self.set_index(int(np.clip(self._index + int(delta), 0, len(self._refs) - 1)))

    def step_episode(self, delta: int) -> bool:
        """Move ``delta`` episodes, clamped, landing on the new episode's first anchor."""
        here = self.at().episode_index
        return self.set_episode(int(np.clip(here + int(delta), 0, len(self._episodes) - 1)))


# ---------------------------------------------------------------------------
# the wire keys the page reads
# ---------------------------------------------------------------------------
def ghost_extras(joints: np.ndarray, joint_names: Sequence[str]) -> dict:
    """The four ``mc_*`` keys that drive the page's LIGHT-PURPLE translucent body.

    Reusing the completion-ghost channel rather than inventing one is deliberate: the page's
    consumer (``WbtTrackingCommand.updateMcGhost``) already keys the joints BY NAME, normalises
    the root quaternion, hides a stale stream and obeys the scene's reference-visible toggle —
    four behaviours worth more than a fresh key's tidiness.

    THE ROOT IS THE IDENTITY, on purpose. The page composes ``mc_root_*`` onto the REFERENCE
    CLIP's root (the held idle pose the low level is anchored on), so an identity root stands the
    ghost exactly at that pelvis and lets its 29 JOINTS be the whole content. That is the honest
    render for this family: a wholebody chunk has no root channel at all — the wire is 29
    positions + 29 finite-differenced velocities — so any root we invented here would be a claim
    the policy never made.
    """
    q = np.asarray(joints, dtype=np.float64).reshape(-1)
    names = [str(n) for n in joint_names]
    if q.size != len(names):
        raise ValueError(
            f"ghost joints ({q.size}) and names ({len(names)}) disagree. The page reads these BY "
            "NAME and simply never applies a name its model lacks, so a mismatch here would "
            "silently freeze part of the body rather than fail.")
    if not np.all(np.isfinite(q)):
        raise ValueError("ghost joints are non-finite; json.dumps would serialize NaN unchecked")
    return {
        "mc_qpos": q.tolist(),
        "mc_joint_names": names,
        "mc_root_pos": [0.0, 0.0, 0.0],
        "mc_root_quat": [1.0, 0.0, 0.0, 0.0],   # wxyz, as the page's normaliser expects
    }


def page_extras(
    *,
    cursor: CursorAt,
    n_samples: int,
    n_episodes: int,
    state: str,
    message: str,
    layout_note: str,
    chunk_steps: int = 0,
    exec_steps: int = 0,
    step: int = -1,
    ghost_step: int = -1,
    ack: int = 0,
) -> dict:
    """The ``flore_*`` telemetry the page's panel renders. Never read by a low level.

    ``flore_n_samples`` is FIRST in spirit and load-bearing in fact: it is
    :data:`PAGE_GATE_KEY`, so a page only grows the panel when this planner is what is feeding
    it. Everything else here is a readout the panel can render as text.

    ``flore_ack`` is the one field that is a PROTOCOL rather than a readout: it counts the control
    requests the planner has finished ACTING on, so a sender can tell "my click has not been
    handled yet" from "it was handled and this is the result". Without it every reader — the page
    and ``scripts/smoke_test_hl_offline.py`` alike — races the planner's loop and reads the
    PREVIOUS state as an answer, which is exactly the class of bug the MC replay picker's
    "show the bridge's own episode, not the selected one" rule exists to avoid.
    """
    if str(state) not in STATES:
        raise ValueError(f"state must be one of {STATES}, got {state!r}")
    return {
        PAGE_GATE_KEY: int(n_samples),
        "flore_sample": int(cursor.index),
        "flore_episode": int(cursor.episode),
        "flore_episode_index": int(cursor.episode_index),
        "flore_episode_frame": int(cursor.episode_frame),
        "flore_episode_offset": int(cursor.episode_offset),
        "flore_episode_anchors": int(cursor.episode_frames),
        "flore_n_episodes": int(n_episodes),
        "flore_split": str(cursor.split),
        "flore_state": str(state),
        "flore_msg": str(message)[:240],
        "flore_layout": str(layout_note)[:120],
        "flore_chunk_steps": int(chunk_steps),
        "flore_exec_steps": int(exec_steps),
        "flore_step": int(step),
        "flore_ghost_step": int(ghost_step),
        "flore_ack": int(ack),
    }


# ---------------------------------------------------------------------------
# the ramp to a hold pose
# ---------------------------------------------------------------------------
def ramp_step(current: np.ndarray, target: np.ndarray, max_step: float) -> tuple[np.ndarray, bool]:
    """One step of a per-element rate-limited move from ``current`` toward ``target``.

    ``(next, arrived)``. PER-ELEMENT and not a normalised L2 step, the same choice the low
    level's own ``HOLOSOMA_VIVE_MAX_JOINT_STEP`` makes and for the same reason: an L2 budget
    couples the channels, so one fast wrist throttles the legs and the whole body creeps.

    Why a ramp exists at all. ``--hold sample`` asks the robot to stand in the DATASET's own
    posture before a chunk is predicted, and that pose is measured **0.514 rad (knee)** from the
    ready crouch the low level holds after **A**. Published in one frame that is a step the low
    level would spend 7+ ticks slewing through with its reference velocity pinned — a lurch, and
    one that happens while the operator is still reading the panel. Ramped, it is a move.
    """
    c = np.asarray(current, dtype=np.float64).reshape(-1)
    t = np.asarray(target, dtype=np.float64).reshape(-1)
    if c.shape != t.shape:
        raise ValueError(f"ramp shapes differ: {c.shape} vs {t.shape}")
    step = float(max_step)
    if not np.isfinite(step) or step <= 0.0:
        return t.copy(), True
    delta = np.clip(t - c, -step, step)
    out = c + delta
    return out, bool(np.all(np.abs(t - out) <= 1e-12))


# ---------------------------------------------------------------------------
# the 3pt wire
# ---------------------------------------------------------------------------
#: How many keypoints a ``3pt`` frame carries, and their operator-facing labels — the first
#: three of the 5pt family's, because they ARE the first three: L hand at the 0.242 m force TCP,
#: R hand likewise, head at ``torso_link`` + 0.35 z. Sliced rather than re-spelled so a preset
#: change in ``hl_dense`` (which re-reads holosoma's own tables under test) propagates here.
THREE_POINT_N = 3
THREE_POINT_LABELS = FIVE_POINT_LABELS[:THREE_POINT_N]


class ThreePointWire:
    """``mode="3pt"``: three keypoint poses, for the ``3pt-*`` family. NO completion.

    The whole reason this class is small. ``diffusion_umi_3tracker`` predicts poses for exactly
    ``("left", "right", "head")``, and ``vr_3point`` is exactly those three points — so the
    handoff is a rename, not a model. Compare the two routes a 3-tracker chunk can take to a
    whole-body controller::

        --low-level 5pt / dense   3 poses -> Motion_Completion -> 5 points / 29 joints
                                  a SECOND model, a SECOND interpreter, ~80 ms/chunk, and the
                                  completion's own null-space choice to clutch away
        --low-level 3pt (HERE)    3 poses -> 3 points
                                  nothing in between

    So ``3pt`` is the right destination for this checkpoint whenever the low level has a keypoint
    front-end, and the completion families exist for the low levels that do NOT.

    ⚠️ WHAT IT SHARES WITH ``FivePointWire``, and what it must not be confused with. Both publish
    keypoints in the ANCHOR's pelvis frame and both take a :class:`KeypointClutch`. The
    difference is that all three points here are OPERATOR-STEERABLE — they are the policy's own
    predictions, not a completion's inference — so ``n_carried == n_points`` (the 5pt wire carries
    only 3 of 5 because re-basing the inferred FEET moves the stance 66-105 mm) and there is no
    derived-point warning to print. That is a property of where the numbers came from, which is
    why it is set here rather than defaulted.

    ⚠️ NOT a ``WireMapper``. That class is LEFT-canonical: it predicts one hand and HOLDS the
    other two points at a home pose, plus a wrist->contact lever for the 0.18 m preset. Here all
    three points are predicted and the chunk is decoded against
    ``ready_pose.resolve_deploy_anchor``, whose poses already sit on the TCP levers the
    ``*-unified-anneal`` presets want (cross-checked against holosoma's own baked
    ``home_pos_pelvis_3pt_tcp`` to 0.1 mm inside that function). Applying a lever here as well
    would double it.
    """

    mode = "3pt"
    low_level_mode = "3pt-unified-anneal"
    gate_note = (f"keypoint-space (HOLOSOMA_VIVE_ARM_TOL, {KEYPOINT_GATE_TOL_M:g} m) on all "
                 f"{THREE_POINT_N} points — every one is PREDICTED, so every one is gated")

    def __init__(self, ready_pos: np.ndarray, ready_orn: np.ndarray, *,
                 clutch: KeypointClutch | None = None,
                 post: PostAnchorConfig | None = None) -> None:
        self._ready_pos = np.asarray(ready_pos, dtype=np.float64).reshape(-1).copy()
        self._ready_orn = np.asarray(ready_orn, dtype=np.float64).reshape(-1).copy()
        if (self._ready_pos.size, self._ready_orn.size) != (3 * THREE_POINT_N, 4 * THREE_POINT_N):
            raise ValueError(
                f"a 3pt ready frame is ({3 * THREE_POINT_N},) positions and "
                f"({4 * THREE_POINT_N},) quats, got {self._ready_pos.size} / "
                f"{self._ready_orn.size}")
        self.clutch = clutch or KeypointClutch(
            self._ready_pos, self._ready_orn, post=post,
            n_points=THREE_POINT_N, n_carried=THREE_POINT_N)

    # --- the frames the planner's loop asks for --------------------------
    @property
    def ready_frame(self) -> tuple:
        return (self._ready_pos.copy(), self._ready_orn.copy())

    def idle(self) -> tuple:
        return self.ready_frame

    def measured(self, rframe) -> tuple | None:
        # NEVER chase a readback, the same rule FivePointWire states: a keypoint wire's pre-arm
        # reference is the frozen reference clip, not the robot's encoders, so tracking measured
        # would move the wire AWAY from what the gate wants.
        del rframe
        return None

    def sync(self, rframe, force_n: float | None = None) -> str:
        del rframe
        # No measured keypoint frame is supplied on this path (there is no RecordTap behind a
        # browser sim), so a CARRYING clutch freezes rather than integrating blind — which is
        # KeypointClutch's documented behaviour for `measured=None` and the safe direction.
        return self.clutch.sync(None, force_n)

    def plan(self, pos: np.ndarray, orn: np.ndarray) -> dict:
        """``(H, 9)`` positions + ``(H, 12)`` xyzw quats -> the per-step frames to publish."""
        p, q = self.clutch.apply(np.asarray(pos, dtype=np.float64),
                                 np.asarray(orn, dtype=np.float64))
        return {"pos": p, "orn": q, "n": int(p.shape[0])}

    def frame(self, plan: dict, i: int) -> tuple:
        return (plan["pos"][i], plan["orn"][i])

    def publish(self, s0, frame: tuple, f_cmd, extra: Mapping | None = None) -> tuple[bool, str]:
        ok, reason = is_safe_keypoint_command(frame[0], frame[1], n_points=THREE_POINT_N)
        if not ok:
            return False, reason
        s0.send(frame[0], frame[1], f_cmd=f_cmd, extra=extra)
        return True, ""

    def commit(self, frame: tuple) -> None:
        self.clutch.commit(frame)

    def gate_report(self, frame: tuple) -> tuple[bool, str]:
        """``(would the low level arm?, a per-point readout)`` for this frame.

        A PREDICTION of the low level's gate against the same reference, not the gate itself.
        All three points are gated here (see the class docstring), so unlike the 5pt readout
        there is no "warned, not gated" tail to explain.
        """
        gaps = keypoint_gaps_m(frame[0], self._ready_pos)
        ok = bool(np.all(gaps <= KEYPOINT_GATE_TOL_M))
        parts = [f"{THREE_POINT_LABELS[i]} {gaps[i] * 1e3:.0f} mm" for i in range(gaps.size)]
        return ok, (f"gate {'PASS' if ok else 'REFUSE'} "
                    f"(tol {KEYPOINT_GATE_TOL_M * 1e3:.0f} mm): " + ", ".join(parts))

    def status(self, frame: tuple) -> str:
        gaps = keypoint_gaps_m(frame[0], self._ready_pos)
        carry = ("" if not self.clutch.post.carrying
                 else f" · carried drift {self.clutch.drift_m() * 1e3:.0f} mm")
        return f"3pt keypoints · worst |dp| vs ready {float(gaps.max()) * 1e3:.0f} mm{carry}"


__all__ = [
    "CONTROL_ACTIONS",
    "CONTROL_CMD",
    "PAGE_GATE_KEY",
    "STATES",
    "THREE_POINT_LABELS",
    "THREE_POINT_N",
    "CursorAt",
    "SampleCursor",
    "ThreePointWire",
    "apply_control",
    "ghost_extras",
    "page_extras",
    "ramp_step",
]
