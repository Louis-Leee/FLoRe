"""The 3-tracker planner's page wiring: the jaw dicts, the closure order, the sequential run.

``scripts/hl_serving_mc.py`` builds its whole page inside one ``_run_viser`` closure over five
sockets, a completion backend in another interpreter and a live G1, so the parts pinned here
cannot be imported and driven — the same reason ``test_hl_serving_jaw_wiring.py`` and
``test_hl_serving_gil_wiring.py`` are source-level. What CAN be driven is the shared jaw state
machine, so the gripper half of this file is functional and only the wiring around it is textual.

EVERY TEST HERE IS A DEFECT THAT REACHED THE RIG (2026-09-01, first bring-up of
``--low-level 5pt`` with both jaws commanded):

1. ``KeyError('overshoot_m')`` on the FIRST jaw command of EVERY Run. The planner hand-wrote its
   ``grip_ctl`` with the CLI's millimetre names (``overshoot_mm``) against a reader that wants
   metres under ``overshoot_m``, and omitted the run-length state entirely. Step 0 was already on
   the wire when it raised, so the chunk's anchor never committed and the arm stopped one step in.
2. ``NameError: cannot access free variable '_record_frame'`` — the heartbeat thread was started
   above that ``def``, so its first tick ran against an unbound cell.
3. No way to run more than one chunk per click, on a path whose chunk is 0.5 s of motion.
4. The HEAD prediction looked absent on the page. It was drawn — as a 2-3 cm polyline at
   ``(-0.004, 0, 1.151)``, i.e. inside the torso mesh, occluded by both the live robot and the
   ghost. The chest is the completion's ROOT conditioning, so that is the path whose
   orientation decides the posture; it now gets a triad long enough to protrude.
"""

from __future__ import annotations

import ast
import re
import threading
from pathlib import Path

_MC_PATH = Path(__file__).resolve().parents[1] / "scripts" / "hl_serving_mc.py"
_MC = _MC_PATH.read_text()
_HS_PATH = Path(__file__).resolve().parents[1] / "scripts" / "hl_serving.py"


def _mc_module():
    """``scripts/hl_serving_mc.py`` as a module. Import-safe (everything runs under
    ``main()``), which is what lets the drawing helpers be tested rather than only grepped."""
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location("_mc_wiring", _MC_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_mc_wiring"] = mod
    spec.loader.exec_module(mod)
    return mod


def _hs_module():
    """``scripts/hl_serving.py`` as a module — the jaw state machine is at module level there
    precisely so tests drive the SHIPPED protocol rather than a re-implementation."""
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location("_hs_mc_wiring", _HS_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_hs_mc_wiring"] = mod
    spec.loader.exec_module(mod)
    return mod


def _fn(name: str, src: str = _MC) -> str:
    """Source of the named function (nested defs included), located via AST for exactness."""
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node) or ""
    raise AssertionError(f"function {name} not found in hl_serving_mc.py")


# ===========================================================================
# 1. the jaw control dict — functional, against the real state machine
# ===========================================================================
def test_new_grip_ctl_supplies_every_key_the_state_machine_reads():
    """The whole point of the constructor: hand its output to ``build_grip_command`` and the
    machine must run end to end. A missing key is not a degraded mode — it is a ``KeyError``
    thrown from the publish path AFTER the first wire frame has gone out."""
    hs = _hs_module()
    ctl = hs.new_grip_ctl(overshoot_mm=10.0, close_below_mm=70.0, confirm_steps=3)
    to_frac, release = hs.build_grip_command(hs.GripperCalib(0.095, 0.0), ctl, threading.Lock())
    for w in (0.090, 0.045, 0.045, 0.045, 0.090):   # open → sustained close → open
        frac, _sq = to_frac(w)
        assert 0.0 <= frac <= 1.0
    release()


def test_new_grip_ctl_converts_the_cli_millimetres_to_metres():
    """The unit seam that caused the outage. The flags are mm (an operator types 10, not 0.010)
    and the machine subtracts the value from a width in METRES, so a dict that passes the flag
    through unconverted would command a 10 METRE overshoot — which saturates to fully closed on
    every predicted width, i.e. a jaw that slams shut and never opens."""
    hs = _hs_module()
    ctl = hs.new_grip_ctl(overshoot_mm=10.0, close_below_mm=70.0, confirm_steps=3)
    assert ctl["overshoot_m"] == 0.010
    assert ctl["close_below_m"] == 0.070
    assert ctl["confirm_steps"] == 3
    assert ctl["hold_tightest"] is True


def test_two_hands_sharing_one_control_dict_share_one_grasp():
    """WHY the 3-tracker planner needs one dict PER HAND, stated as the failure it prevents.

    ``build_grip_command``'s state lives in the dict, not the closure. Share it and the LEFT
    jaw's confirmation run engages the RIGHT jaw's overshoot: the right hand publishes a squeezed
    target on its FIRST close, having never sustained one."""
    hs = _hs_module()
    calib = hs.GripperCalib(0.095, 0.0)
    shared = hs.new_grip_ctl(overshoot_mm=10.0, close_below_mm=70.0, confirm_steps=3)
    lock = threading.Lock()
    left, _ = hs.build_grip_command(calib, shared, lock)
    right, _ = hs.build_grip_command(calib, shared, lock)

    for _ in range(3):
        left(0.045)                     # LEFT earns its confirmation run
    frac_shared, squeezing = right(0.045)   # RIGHT's very first close
    assert squeezing is True, "the shared dict did not leak the left hand's engagement"

    # Per-hand dicts: the same first close is NOT squeezed, because this hand has not earned it.
    own_l = hs.new_grip_ctl(overshoot_mm=10.0, close_below_mm=70.0, confirm_steps=3)
    own_r = hs.new_grip_ctl(overshoot_mm=10.0, close_below_mm=70.0, confirm_steps=3)
    left2, _ = hs.build_grip_command(calib, own_l, lock)
    right2, _ = hs.build_grip_command(calib, own_r, lock)
    for _ in range(3):
        left2(0.045)
    frac_own, squeezing_own = right2(0.045)
    assert squeezing_own is False
    assert frac_own > frac_shared, (
        "with its own dict the right jaw must publish a WIDER (un-squeezed) target on its first "
        "close than it does when it inherits the left hand's engagement")


def test_the_mc_planner_builds_one_control_dict_per_gripper_role():
    """The wiring side of the two tests above, in the file that got it wrong."""
    body = _fn("_run_viser")
    assert re.search(r"grip_ctl\s*=\s*\{role:\s*HL\.new_grip_ctl\(", body), (
        "the 3-tracker planner no longer builds its jaw control dict per role from "
        "HL.new_grip_ctl — a hand-written literal here is what shipped `overshoot_mm` against a "
        "reader that wants `overshoot_m`")
    assert re.search(r"build_grip_command\(calib,\s*grip_ctl\[role\]", body), (
        "both hands are sharing one control dict again — the left jaw's confirm counter will "
        "engage the right jaw's overshoot")
    assert "overshoot_mm\"" not in _MC and "close_below_mm\"" not in _MC, (
        "a millimetre-named key is back in hl_serving_mc.py; the state machine reads metres")


# ===========================================================================
# 2. closure order — the heartbeat's first tick
# ===========================================================================
def test_record_frame_is_defined_before_the_heartbeat_thread_starts():
    """A `def` inside a function does not exist until execution reaches it. The heartbeat is
    spawned from the same body and calls ``_idle_sync_wire`` -> ``_record_frame`` on its first
    tick, so defining the readback further down left that tick raising ``NameError`` and
    publishing from an un-synced base. Recoverable and therefore easy to leave in."""
    body = _fn("_run_viser")
    where_def = body.find("def _record_frame(")
    where_thread = body.find("target=_heartbeat")
    assert where_def >= 0, "_record_frame is gone from _run_viser"
    assert where_thread >= 0, "the heartbeat thread is no longer started in _run_viser"
    assert where_def < where_thread, (
        "_record_frame is defined AFTER the heartbeat thread is started — its first tick will "
        "fail with NameError on an unbound closure cell")
    # And the readback must exist exactly once: a second copy further down would shadow nothing
    # but would drift from this one.
    assert body.count("def _record_frame(") == 1


# ===========================================================================
# 3. the sequential rollout
# ===========================================================================
def test_the_sequential_rollout_is_bounded_four_ways():
    """A policy has no end of its own, so an unbounded predict→execute loop is a robot that
    keeps moving after the operator has looked away. Each of these four is a separate way out and
    none substitutes for another: ⏹ retires the loop at a chunk boundary, STOP interrupts the
    motion within one step, the ceiling fires with nobody present, and a low level that drops off
    :8770 must not be planned against."""
    body = _fn("_run_to_completion")
    assert "while not stop.is_set():" in body, "STOP no longer ends the rollout"
    assert re.search(r'auto_ctl\["gen"\]\s*!=\s*my_gen', body), (
        "the generation check is gone — ⏹ cannot retire an in-flight rollout")
    assert re.search(r"elapsed\s*>=\s*max_s", body), "the wall-clock ceiling is gone"
    assert "s0.num_clients == 0" in body, (
        "the rollout no longer notices the low level disconnecting mid-take")
    assert "busy.acquire(blocking=False)" in body, (
        "the rollout no longer takes the same mutex as the manual buttons")


def test_the_sequential_rollout_clears_the_stop_latch_exactly_once():
    """The subtle half. Clearing ``stop`` per iteration — which is what the single-chunk Run
    branch correctly does — would make STOP unpressable here: the loop would wipe it and re-plan.
    Clearing it never would make a rollout after any earlier abort a silent no-op."""
    body = _fn("_run_to_completion")
    assert body.count("stop.clear()") == 1, (
        "stop.clear() must appear exactly once in the rollout (at entry). More than once and "
        "STOP is erased mid-take; none and a latched stop makes the rollout a no-op")
    entry = body.find("stop.clear()")
    loop = body.find("while not stop.is_set():")
    assert 0 <= entry < loop, "stop is cleared inside the loop rather than at entry"


def test_a_failed_predict_ends_the_rollout_rather_than_spinning():
    """``_predict`` refuses on a stale observation and returns without staging on a failed
    completion. Continuing would re-plan off a frozen scene until the ceiling while the heartbeat
    republishes the last frame — motion that looks deliberate and is not."""
    body = _fn("_run_to_completion")
    assert re.search(r"out\s*=\s*staged\[\"out\"\]", body), (
        "the rollout no longer checks whether the chunk actually staged")
    assert re.search(r"if out is None:(?:.|\n)*?break", body), (
        "a refusal / failed completion no longer breaks the rollout")


def test_the_rollout_runs_on_its_own_thread():
    """It blocks for the whole take. On the viser callback thread it would freeze the page — and
    STOP is a viser callback."""
    body = _fn("_run_viser")
    assert re.search(r"threading\.Thread\(target=_run_to_completion", body), (
        "the sequential rollout is started inline on the button callback; STOP would be "
        "unclickable for the duration")


def test_manual_presses_are_refused_while_a_rollout_is_active():
    """The rollout holds ``busy`` only WITHIN a chunk, so between two iterations a manual press
    would land: Predict would overwrite the chunk about to be dripped, and Reset would clear the
    operator's latched STOP and re-home the decode anchor underneath a live take."""
    body = _fn("_run_viser")
    assert re.search(r"_seq\s*=\s*auto_running\.is_set\(\)", body), (
        "the page loop no longer looks at whether a sequential rollout is active")
    assert re.search(r"if _seq and \(flags\[\"predict\"\] or flags\[\"run\"\] or "
                     r"flags\[\"reset\"\]\)", body), (
        "Predict / Run / Reset are no longer refused during a sequential rollout")


# ===========================================================================
# 4. the HEAD path is actually visible
# ===========================================================================
def test_the_head_gets_a_triad_long_enough_to_escape_the_torso_mesh():
    """The chest keypoint of the ready pose is ``(-0.004, 0, 1.151)`` — INSIDE the torso mesh —
    and it translates ~2-3 cm over a chunk. A polyline there is depth-occluded by the live robot
    AND by the violet ghost, so the head prediction read as "not drawn at all" on the rig. Its
    triad has to protrude from a torso roughly 0.15 m half-deep; a hand is already in open air
    and gets a short one so a reach is not buried in arrows."""
    mc = _mc_module()
    assert set(mc._PLAN_STYLE) >= {"left", "right", "head"}
    head, hand = mc._PLAN_STYLE["head"], mc._PLAN_STYLE["left"]
    assert head["axes"] >= 0.20, (
        f"the head triad is {head['axes']} m — it must clear a ~0.15 m half-deep torso mesh or "
        "it is invisible exactly where it matters")
    assert head["axes"] > hand["axes"], "the head is styled like a hand again"
    assert mc._PLAN_STYLE["left"]["rgb"] != mc._PLAN_STYLE["head"]["rgb"]


def test_triad_samples_always_include_the_last_chunk_step():
    """The final step is where the chunk ENDS, and ``wire.commit`` carries the last EXECUTED
    step forward, so it is the one sample that must never be rounded away to a stride
    boundary."""
    mc = _mc_module()
    for n, stride in ((50, 12), (50, 10), (30, 12), (16, 5), (7, 12), (1, 12)):
        idx = mc._triad_indices(n, stride)
        assert idx[0] == 0, (n, stride, idx)
        assert idx[-1] == n - 1, f"n={n} stride={stride} dropped the last step: {idx}"
        assert len(set(idx.tolist())) == len(idx), f"duplicate sample: {idx}"
        assert idx.max() < n
    assert len(mc._triad_indices(50, 0)) == 50, "a zero stride must not divide by zero"


def test_every_role_is_drawn_three_ways_and_the_old_node_is_removed_first():
    """Removal order is load-bearing: viser addresses nodes by NAME, so adding
    ``/plan/left/line`` while the previous handle still holds that name REPLACES the node, and
    a later ``.remove()`` on the stale handle deletes the one just drawn — a role that vanishes
    on its second chunk."""
    body = _fn("_draw_paths")
    for kind in ("add_line_segments", "add_point_cloud", "add_batched_axes"):
        assert kind in body, f"_draw_paths no longer draws with {kind}"
    where_remove = body.find("old.remove()")
    where_add = body.find("add_line_segments")
    assert 0 <= where_remove < where_add, (
        "_draw_paths adds the new node before removing the old one of the same name")
    assert re.search(r'for kind in \("line", "dots", "triads"\)', body), (
        "the per-role removal no longer covers all three node kinds, so one of them leaks")


def test_the_triad_toggle_reaches_the_already_drawn_chunk():
    """An operator flips this while looking at a staged chunk — the only moment it matters — so
    it cannot be a flag that takes effect on the NEXT prediction."""
    body = _fn("_run_viser")
    assert 'view_ctl: dict = {"triads": True}' in body, (
        "the triad flag is no longer a dict defined above _draw_paths; a closure over the GUI "
        "handle repeats the _record_frame late-binding bug")
    assert "def _set_triads_visible(" in body
    assert re.search(r'if key\.endswith\(":triads"\)', body), (
        "_set_triads_visible no longer walks the live handles")
    assert re.search(r"@c_triads\.on_update(?:.|\n)*?_set_triads_visible", body), (
        "the checkbox is not wired to the live handles")


# ===========================================================================
# 5. Reset restores the violet ghost's ROOT, not just its joints
# ===========================================================================
def test_reset_restores_the_ghost_root_and_not_only_its_joints():
    """A completion's pose is 29 joints PLUS a floating root, and playback writes all three
    (``update_cfg`` + ``ghost_root.position`` + ``ghost_root.wxyz``) so the ghost can take a step
    the live mesh cannot. Reset used to restore only the joints, leaving the violet body standing
    translated and YAWED wherever the last take ended — the scene never looked neutral again.
    The root is display-only, which is why no gate, wire or clutch check catches it."""
    body = _fn("_run_viser")
    helper = _fn("_reset_ghost")
    assert "ghost_root.position" in helper and "ghost_root.wxyz" in helper, (
        "_reset_ghost no longer restores the ghost ROOT — only its joints, which is the bug")
    assert "update_cfg" in helper, "_reset_ghost no longer restores the joints"

    # ONE spelling: the construction site and Reset must call the same helper, or "where the
    # ghost starts" and "where Reset puts it" can drift apart.
    reset_arm = body[body.find('if flags["reset"]:'):body.find('if flags["predict"]:')]
    assert "_reset_ghost()" in reset_arm, "the Reset branch no longer resets the ghost"
    assert "update_cfg(remap_dof_to_urdf(anchor.dof" not in reset_arm, (
        "the Reset branch went back to updating the ghost joints inline, bypassing the root")
    assert body.count("_reset_ghost()") >= 2, (
        "_reset_ghost must be called at construction too, so the initial pose and the reset pose "
        "are one piece of code")
    assert re.search(r'flags\["ghost_step"\] = 0', reset_arm), (
        "Reset no longer rewinds the playback phase, so the next chunk resumes the retired "
        "chunk's step index")


# ===========================================================================
# 6. the live foot-IK switch
# ===========================================================================
def test_the_foot_ik_switch_only_records_a_request_and_predict_applies_it():
    """A live post-process switch has exactly one safe application point.

    ``ChunkCompleter.complete`` holds the completer's lock for the WHOLE chunk (1.4-4.9 s with
    foot IK on) and ``set_post_process`` is deliberately not lock-protected, so calling it from
    the viser callback would either race the settings or freeze the page for seconds. And each
    chunk is its own backend episode, so applying it on the predicting thread before the episode
    opens is what makes a flip land on a chunk BOUNDARY instead of splitting one motion across
    two different bodies."""
    body = _fn("_run_viser")
    cb = body[body.find("@c_foot.on_update"):]
    cb = cb[:cb.find("\n    @")] if "\n    @" in cb else cb
    assert "set_post_process" not in cb, (
        "the foot-IK checkbox reconfigures the completer directly — it must only record a "
        "request; _predict applies it on the predicting thread")
    apply_fn = _fn("_apply_foot_ik")
    assert "set_post_process" in apply_fn
    pred = _fn("_predict")
    assert "_apply_foot_ik()" in pred, "_predict no longer applies a pending foot-IK change"
    assert pred.find("_apply_foot_ik()") < pred.find("infer_chunk"), (
        "the foot-IK change is applied after the chunk is sampled/completed")


def test_the_foot_ik_switch_always_passes_the_iteration_count():
    """``set_post_process``'s ``foot_ik_iters`` is set-or-CLEAR, not leave-alone: naming only
    ``foot_fix`` silently resets the override to the checkpoint's 150, a 3.5x cost change
    (1376 ms -> 4887 ms per chunk) that the operator did not ask for."""
    apply_fn = _fn("_apply_foot_ik")
    assert re.search(r"set_post_process\(foot_fix=fix,\s*foot_ik_iters=iters\)", apply_fn), (
        "_apply_foot_ik no longer passes foot_ik_iters alongside foot_fix")


def test_set_post_process_really_is_set_or_clear_for_the_iteration_count():
    """The functional half of the test above, against the shipped class — so the comment in the
    planner is checked rather than trusted."""
    from fuji.s1_policy.serving import hl_dense
    c = hl_dense.ChunkCompleter(umi_dir="/definitely/not/here")
    c.set_post_process(foot_fix="inertialize", foot_ik_iters=40)
    assert (c._foot_fix, c._foot_ik_iters) == ("inertialize", 40)
    c.set_post_process(foot_fix="inertialize")          # iters omitted
    assert c._foot_ik_iters is None, (
        "omitting foot_ik_iters no longer clears it — the planner's 'pass both every time' "
        "comment would be stale, but keep BOTH in step if this is changed on purpose")


def test_the_foot_ik_panel_names_the_per_family_consequence():
    """One switch serves both families because foot IK runs inside the completion, upstream of
    the wire split — but the consequence is not the same, and "anti-slide" sounds cosmetic. On
    dense the rewritten legs become published joint targets; on 5pt only the two ungated feet
    move. A panel that does not say which invites turning it on to fix a rendering nit."""
    body = _fn("_run_viser")
    panel = body[body.find('add_folder("Motion completion")'):]
    panel = panel[:panel.find('add_folder("Bring-up report")')]
    assert "Foot IK" in panel, "the foot-IK checkbox is gone from the page"
    assert 'wire.mode == "dense"' in panel, (
        "the panel no longer distinguishes the two families' consequences")
    assert "LEGS" in panel and "FOOT keypoints" in panel
    assert "481 ms" in panel, "the panel no longer states the budget it blows"


def test_stop_also_retires_the_rollout():
    """STOP is the operator's one reflex. It must not merely interrupt the chunk in flight and
    let the loop plan another one."""
    body = _fn("_run_viser")
    stop_cb = body[body.find("@b_stop.on_click"):body.find("@b_reset.on_click")]
    assert "stop.set()" in stop_cb
    assert re.search(r'auto_ctl\["gen"\]\s*\+=\s*1', stop_cb), (
        "STOP no longer bumps the rollout generation — the loop could re-plan after it")
