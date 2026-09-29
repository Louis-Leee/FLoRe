"""Tests for MotionConfig and MotionHandle in mjswan.motion.

Layer: L1 (no I/O, no external dependencies).
"""

from mjswan.motion import MotionConfig, MotionHandle


# ===========================================================================
# MotionConfig — to_summary_dict
# ===========================================================================
class TestMotionConfigToSummaryDict:
    def test_to_summary_dict_minimal(self):
        cfg = MotionConfig(name="Walk")
        result = cfg.to_summary_dict()
        assert result == {"name": "Walk"}
        assert "default" not in result

    def test_to_summary_dict_with_default_true(self):
        cfg = MotionConfig(name="Run", default=True)
        result = cfg.to_summary_dict()
        assert result == {"name": "Run", "default": True}


# ===========================================================================
# MotionConfig — to_dict
# ===========================================================================
class TestMotionConfigToDict:
    def _make_cfg(self, **kwargs) -> MotionConfig:
        defaults = dict(
            name="Spin Kick",
            anchor_body_name="torso_link",
            body_names=("pelvis", "torso_link"),
        )
        defaults.update(kwargs)
        return MotionConfig(**defaults)

    def test_to_dict_required_fields(self):
        cfg = self._make_cfg(fps=30.0)
        result = cfg.to_dict("motions/spin.npz")
        assert result["name"] == "Spin Kick"
        assert result["path"] == "motions/spin.npz"
        assert result["fps"] == 30.0
        assert result["anchor_body_name"] == "torso_link"
        assert result["body_names"] == ["pelvis", "torso_link"]

    def test_to_dict_body_names_serialized_as_list(self):
        cfg = self._make_cfg(body_names=("a", "b", "c"))
        result = cfg.to_dict("x.npz")
        assert isinstance(result["body_names"], list)

    def test_to_dict_omits_dataset_joint_names_when_none(self):
        cfg = self._make_cfg(dataset_joint_names=None)
        result = cfg.to_dict("x.npz")
        assert "dataset_joint_names" not in result

    def test_to_dict_includes_dataset_joint_names_when_set(self):
        cfg = self._make_cfg(dataset_joint_names=["joint_a", "joint_b"])
        result = cfg.to_dict("x.npz")
        assert result["dataset_joint_names"] == ["joint_a", "joint_b"]
        assert isinstance(result["dataset_joint_names"], list)

    def test_to_dict_omits_default_key_when_false(self):
        cfg = self._make_cfg(default=False)
        result = cfg.to_dict("x.npz")
        assert "default" not in result

    def test_to_dict_includes_default_key_when_true(self):
        cfg = self._make_cfg(default=True)
        result = cfg.to_dict("x.npz")
        assert result["default"] is True

    def test_to_dict_uses_custom_fps(self):
        cfg = self._make_cfg(fps=120.0)
        result = cfg.to_dict("x.npz")
        assert result["fps"] == 120.0

    def test_to_dict_omits_metadata_when_empty(self):
        cfg = self._make_cfg()
        result = cfg.to_dict("x.npz")
        assert "metadata" not in result

    def test_to_dict_includes_metadata_when_set(self):
        cfg = self._make_cfg()
        cfg.metadata["source_run"] = "abc/def/xyz"
        result = cfg.to_dict("x.npz")
        assert result["metadata"] == {"source_run": "abc/def/xyz"}


# ===========================================================================
# MotionConfig — defaults
# ===========================================================================
class TestMotionConfigDefaults:
    def test_defaults(self):
        cfg = MotionConfig(name="Test")
        assert cfg.anchor_body_name == ""
        assert cfg.body_names == ()
        assert cfg.fps == 50.0
        assert cfg.default is False
        assert cfg.metadata == {}
        assert cfg.source is None
        assert cfg.data is None
        assert cfg.dataset_joint_names is None


# ===========================================================================
# MotionHandle
# ===========================================================================
class TestMotionHandle:
    def _make_handle(self, **kwargs) -> MotionHandle:
        cfg = MotionConfig(name=kwargs.pop("name", "Walk"), **kwargs)
        return MotionHandle(cfg, policy=None)

    def test_name_property(self):
        handle = self._make_handle(name="Cartwheel")
        assert handle.name == "Cartwheel"

    def test_set_metadata_stores_value(self):
        handle = self._make_handle()
        handle.set_metadata("source_run", "abc123")
        assert handle._config.metadata["source_run"] == "abc123"

    def test_set_metadata_returns_self(self):
        handle = self._make_handle()
        result = handle.set_metadata("key", "value")
        assert result is handle
