"""The offline five-point body's display variants: the tinted translucent bodies and the
"Real G1 mesh" (URDF materials) toggle. Fakes stand in for viser and the IK; the solve itself is
covered in test_hl_serving_5pt.py."""
from __future__ import annotations

import types

import numpy as np
import pytest

from flore.s1_policy.inference import keypoint_humanoid as KH


class _Geom:
    def __init__(self, rgba):
        self.visual = types.SimpleNamespace(main_color=None if rgba is None else np.array(rgba))


def test_invisible_predicate_is_alpha_zero_only():
    assert KH._urdf_geometry_is_invisible(_Geom([0, 0, 0, 0]))
    assert not KH._urdf_geometry_is_invisible(_Geom([178, 178, 178, 255]))
    assert not KH._urdf_geometry_is_invisible(_Geom([51, 51, 51, 1]))        # faint is still drawn
    assert not KH._urdf_geometry_is_invisible(_Geom([0.0, 0.0, 0.0]))         # RGB-only visual: kept
    assert not KH._urdf_geometry_is_invisible(_Geom(None))                    # no colour: kept
    assert not KH._urdf_geometry_is_invisible(object())                       # no visual at all: kept


# ---- fakes ---------------------------------------------------------------------------------------
class _Frame:
    def __init__(self, name, visible=True):
        self.name, self.visible, self.position, self.wxyz = name, visible, (0, 0, 0), (1, 0, 0, 0)


class _Scene:
    def __init__(self):
        self.frames = {}

    def add_frame(self, name, show_axes=True, visible=True, **_kw):
        self.frames[name] = _Frame(name, visible)
        return self.frames[name]


class _Server:
    def __init__(self):
        self.scene = _Scene()


class _FakeViserUrdf:
    built: list = []

    def __init__(self, server, urdf_or_path, root_node_name="/", mesh_color_override=None, **_kw):
        self.root_node_name, self.mesh_color_override = root_node_name, mesh_color_override
        self.cfgs: list = []
        _FakeViserUrdf.built.append(self)

    def update_cfg(self, cfg):
        self.cfgs.append(np.asarray(cfg).copy())


class _Handle:
    def __init__(self, value):
        self.value = value

    def on_update(self, fn):
        return fn


class _Gui:
    def __init__(self):
        self.handles = {}

    class _Folder:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def add_folder(self, _name):
        return self._Folder()

    def add_text(self, label, initial_value="", **_kw):
        self.handles[label] = _Handle(initial_value)
        return self.handles[label]

    def add_checkbox(self, label, initial_value=False, **_kw):
        self.handles[label] = _Handle(initial_value)
        return self.handles[label]

    def add_dropdown(self, label, options, initial_value=None, **_kw):
        self.handles[label] = _Handle(initial_value if initial_value is not None else options[0])
        return self.handles[label]


def _solved(n=3):
    per = {}
    for k, series in enumerate(KH.SERIES):
        per[series] = types.SimpleNamespace(
            n=n, root_pos=np.tile([0.1 * k, 0.0, 0.75], (n, 1)), root_wxyz=np.tile([1.0, 0, 0, 0], (n, 1)),
            cfg=np.full((n, 31), float(k)), residual_mm=np.zeros((n, 5)), roles=list(KH.IK_ROLES),
        )
    return KH._Solved(generation=0, per_series=per, identity="fake", ms=1.0)


@pytest.fixture
def body(monkeypatch):
    pytest.importorskip("viser")
    import viser.extras

    _FakeViserUrdf.built = []
    monkeypatch.setattr(viser.extras, "ViserUrdf", _FakeViserUrdf)
    monkeypatch.setattr(KH.KeypointIK, "load", classmethod(lambda cls, **kw: (object(), "")))
    monkeypatch.setattr(KH.KeypointHumanoid, "_load_real_urdf", lambda self: (object(), ["geometry_8"]))
    kp = KH.KeypointHumanoid(_Server(), enabled=True, urdf_path="/nonexistent/g1.urdf", log=lambda m: None)
    assert kp.available, kp.disabled_reason
    gui = _Gui()
    kp.build_gui(gui)
    kp._solved = _solved()
    return kp, gui


def test_only_the_tinted_bodies_exist_until_the_checkbox_asks(body):
    kp, gui = body
    kp.render_at(0)
    assert [u.root_node_name for u in _FakeViserUrdf.built] == ["/kp_ik/target", "/kp_ik/prediction"]
    assert all(u.mesh_color_override is not None for u in _FakeViserUrdf.built)
    frames = kp.server.scene.frames
    assert frames["/kp_ik/prediction"].visible and frames["/kp_ik/target"].visible
    assert "/kp_ik_real/prediction" not in frames
    assert gui.handles["mesh"].value == KH._MESH_NOTE_TINTED


def test_real_mesh_toggle_builds_lazily_and_swaps_visibility_both_ways(body):
    kp, gui = body
    kp.render_at(0)
    gui.handles[KH.REAL_MESH_LABEL].value = True
    kp.render_at(1)
    real = [u for u in _FakeViserUrdf.built if u.root_node_name.startswith("/kp_ik_real/")]
    assert sorted(u.root_node_name for u in real) == ["/kp_ik_real/prediction", "/kp_ik_real/target"]
    assert all(u.mesh_color_override is None for u in real)     # the URDF's own materials
    frames = kp.server.scene.frames
    assert frames["/kp_ik_real/prediction"].visible and frames["/kp_ik_real/target"].visible
    assert not frames["/kp_ik/prediction"].visible and not frames["/kp_ik/target"].visible
    # The real body was posed from the same solve, at the requested step.
    pred_real = next(u for u in real if u.root_node_name == "/kp_ik_real/prediction")
    assert len(pred_real.cfgs) == 1 and np.all(pred_real.cfgs[0] == 0.0)
    assert frames["/kp_ik_real/prediction"].position == (0.0, 0.0, 0.75)
    assert gui.handles["mesh"].value == KH._MESH_NOTE_REAL
    # Back to tinted: nothing is rebuilt, the real bodies just hide.
    n_built = len(_FakeViserUrdf.built)
    gui.handles[KH.REAL_MESH_LABEL].value = False
    kp.render_at(1)
    assert len(_FakeViserUrdf.built) == n_built
    assert frames["/kp_ik/prediction"].visible and not frames["/kp_ik_real/prediction"].visible
    assert gui.handles["mesh"].value == KH._MESH_NOTE_TINTED


def test_series_checkboxes_still_pick_the_body_in_real_mode(body):
    kp, gui = body
    gui.handles[KH.REAL_MESH_LABEL].value = True
    gui.handles["Target body"].value = False
    kp.render_at(0)
    frames = kp.server.scene.frames
    assert frames["/kp_ik_real/prediction"].visible
    assert not frames["/kp_ik_real/target"].visible and not frames["/kp_ik/target"].visible
    kp.clear()
    assert not any(f.visible for f in frames.values())


def test_a_failed_real_build_keeps_the_tinted_body_and_says_why(body, monkeypatch):
    kp, gui = body

    def boom(self):
        raise RuntimeError("mesh file missing")

    monkeypatch.setattr(KH.KeypointHumanoid, "_load_real_urdf", boom)
    gui.handles[KH.REAL_MESH_LABEL].value = True
    kp.render_at(0)
    frames = kp.server.scene.frames
    assert frames["/kp_ik/prediction"].visible and "/kp_ik_real/prediction" not in frames
    assert "mesh file missing" in gui.handles["mesh"].value
    # Not retried every tick.
    kp.render_at(1)
    assert len(_FakeViserUrdf.built) == 2
