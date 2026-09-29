"""Tests for mjswan.Builder — project ID assignment, config JSON structure, and build output.

Layer breakdown:
  L1 (pure logic / lightweight I/O): TestProjectIdAssignment, TestBuilderValidation,
                                     TestSaveConfigJson, TestSaveWebPolicyJson
  L3 slow (triggers frontend build): TestFullBuild

Run only L1 tests (pre-commit):  pytest -m "not slow"
Run all tests (CI):               pytest
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import mujoco
import pytest

import mjswan
from mjswan._build_client import ClientBuilder
from mjswan.builder import Builder
from mjswan.envs.mdp import observations as obs_fns
from mjswan.envs.mdp import terminations as term_fns
from mjswan.envs.mdp.actions import JointEffortActionCfg, JointPositionActionCfg
from mjswan.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjswan.managers.termination_manager import TerminationTermCfg
from mjswan.utils import name2id


# ===========================================================================
# L1 — project ID assignment rules
# ===========================================================================
class TestProjectIdAssignment:
    def test_first_project_without_explicit_id_gets_none(self):
        builder = Builder()
        project = builder.add_project(name="Main Demo")
        assert project.id is None

    def test_second_project_without_explicit_id_gets_auto_id(self):
        builder = Builder()
        builder.add_project(name="Main Demo")
        second = builder.add_project(name="MuJoCo Menagerie")
        assert second.id == name2id("MuJoCo Menagerie")

    def test_auto_id_uses_name2id_transform(self):
        builder = Builder()
        builder.add_project(name="First")
        second = builder.add_project(name="My Project Name")
        assert second.id == "my_project_name"

    def test_explicit_id_used_as_is_on_first_project(self):
        project = Builder().add_project(name="Main Demo", id="custom")
        assert project.id == "custom"

    def test_explicit_id_used_as_is_on_subsequent_project(self):
        builder = Builder()
        builder.add_project(name="First")
        second = builder.add_project(name="Second", id="explicit_id")
        assert second.id == "explicit_id"

    def test_mixed_id_sequence(self):
        builder = Builder()
        p1 = builder.add_project(name="Project A")
        p2 = builder.add_project(name="Project B")
        p3 = builder.add_project(name="Project C", id="custom")
        assert p1.id is None
        assert p2.id == name2id("Project B")
        assert p3.id == "custom"

    def test_get_projects_returns_independent_copy(self):
        builder = Builder()
        builder.add_project(name="Test")
        copy = builder.get_projects()
        copy.clear()
        assert len(builder.get_projects()) == 1


# ===========================================================================
# L1 — GTM ID handling
# ===========================================================================
class TestBuilderGtmId:
    def test_defaults_to_none(self):
        assert Builder()._gtm_id is None

    def test_stored_when_provided(self):
        assert Builder(gtm_id="GTM-W79HQ38W")._gtm_id == "GTM-W79HQ38W"


class TestClientBuilderCustomTerminations:
    def test_preserves_template_imports_when_inlining_local_utils(
        self, tmp_path, monkeypatch
    ):
        project_dir = tmp_path / "template"
        output_dir = project_dir / "src" / "core" / "termination"
        output_dir.mkdir(parents=True)
        source_dir = tmp_path / "custom"
        source_dir.mkdir()

        (source_dir / "utils.ts").write_text(
            "export function helper(): boolean {\n  return true;\n}\n"
        )
        custom_ts = source_dir / "FooTermination.ts"
        custom_ts.write_text(
            "import { TerminationBase, type TerminationConfig } from './TerminationBase';\n"
            "import type { PolicyState } from '../policy/types';\n"
            "import { helper } from './utils';\n"
            "\n"
            "export class FooTermination extends TerminationBase {\n"
            "  constructor(config: TerminationConfig) {\n"
            "    super(config);\n"
            "  }\n"
            "\n"
            "  evaluate(_state: PolicyState): boolean {\n"
            "    return helper();\n"
            "  }\n"
            "}\n"
        )

        monkeypatch.setattr(
            term_fns,
            "_custom_registry",
            {
                "foo": term_fns.TermFunc(
                    ts_name="FooTermination",
                    ts_src=str(custom_ts),
                )
            },
        )

        ClientBuilder(project_dir).generate_custom_terminations()

        generated = (output_dir / "custom_terminations.ts").read_text()
        assert (
            "import { TerminationBase, type TerminationConfig }"
            " from './TerminationBase';"
        ) in generated
        assert "import { helper } from './utils';" not in generated
        assert "function helper(): boolean" in generated
        assert "export function helper" not in generated
        assert "export class FooTermination extends TerminationBase" in generated


# ===========================================================================
# L1 — validation
# ===========================================================================
class TestBuilderValidation:
    def test_build_with_no_projects_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="Cannot build an empty application"):
            Builder().build(tmp_path / "out")

    def test_policy_filename_rejects_empty_string(self):
        with pytest.raises(ValueError):
            Builder()._policy_filename("")

    def test_policy_filename_rejects_forward_slash(self):
        with pytest.raises(ValueError):
            Builder()._policy_filename("path/policy")

    def test_policy_filename_rejects_backslash(self):
        with pytest.raises(ValueError):
            Builder()._policy_filename("path\\policy")

    def test_policy_filename_accepts_plain_name(self):
        assert Builder()._policy_filename("my_policy") == "my_policy"


# ===========================================================================
# L1 — from_mjlab wiring (no mjlab/wandb required, dependencies monkeypatched)
# ===========================================================================
class TestFromMjlab:
    """Verify Builder.from_mjlab forwards run_path to SceneHandle.add_policy_from_wandb."""

    @staticmethod
    def _patch(monkeypatch):
        """Patch add_mjlab_scene to skip the real mjlab loader and return a mock SceneHandle."""
        from mjswan.project import ProjectHandle

        scene_handle = MagicMock(name="SceneHandle")
        monkeypatch.setattr(
            ProjectHandle,
            "add_mjlab_scene",
            lambda self, task_id, *, play=False: scene_handle,
        )
        return scene_handle

    def test_no_run_path_does_not_call_add_policy_from_wandb(self, monkeypatch):
        scene_handle = self._patch(monkeypatch)
        Builder.from_mjlab("go2_flat")
        scene_handle.add_policy_from_wandb.assert_not_called()

    def test_str_run_path_forwarded_with_task_id(self, monkeypatch):
        scene_handle = self._patch(monkeypatch)
        Builder.from_mjlab("go2_flat", run_path="org/proj/abc")
        scene_handle.add_policy_from_wandb.assert_called_once_with(
            "org/proj/abc", task_id="go2_flat"
        )

    def test_list_run_path_forwarded_as_is(self, monkeypatch):
        scene_handle = self._patch(monkeypatch)
        run_paths = ["org/proj/a", "org/proj/b"]
        Builder.from_mjlab("go2_flat", run_path=run_paths)
        scene_handle.add_policy_from_wandb.assert_called_once_with(
            run_paths, task_id="go2_flat"
        )


# ===========================================================================
# L1 — _save_config_json output structure (no frontend build)
# ===========================================================================
class TestSaveConfigJson:
    def _read_config(self, tmp_path: Path) -> dict:
        return json.loads((tmp_path / "assets" / "config.json").read_text())

    def test_config_contains_version(self, tmp_path, minimal_model):
        builder = Builder()
        builder.add_project(name="P").add_scene(name="S", model=minimal_model)
        builder._save_config_json(tmp_path)
        assert self._read_config(tmp_path)["version"] == mjswan.__version__

    def test_config_has_projects_list(self, tmp_path, minimal_model):
        builder = Builder()
        builder.add_project(name="P").add_scene(name="S", model=minimal_model)
        builder._save_config_json(tmp_path)
        config = self._read_config(tmp_path)
        assert isinstance(config["projects"], list)
        assert len(config["projects"]) == 1

    def test_project_name_and_id_in_config(self, tmp_path, minimal_model):
        builder = Builder()
        builder.add_project(name="Main Demo").add_scene(name="S", model=minimal_model)
        builder._save_config_json(tmp_path)
        project = self._read_config(tmp_path)["projects"][0]
        assert project["name"] == "Main Demo"
        assert project["id"] is None

    def test_scene_path_uses_name2id_with_mjb_for_model(self, tmp_path, minimal_model):
        builder = Builder()
        builder.add_project(name="P").add_scene(name="My Scene", model=minimal_model)
        builder._save_config_json(tmp_path)
        scene = self._read_config(tmp_path)["projects"][0]["scenes"][0]
        assert scene["name"] == "My Scene"
        assert scene["path"] == "my_scene/scene.mjb"

    def test_scene_path_uses_mjz_for_spec(self, tmp_path, minimal_spec):
        builder = Builder()
        builder.add_project(name="P").add_scene(name="My Scene", spec=minimal_spec)
        builder._save_config_json(tmp_path)
        scene = self._read_config(tmp_path)["projects"][0]["scenes"][0]
        assert scene["path"] == "my_scene/scene.mjz"

    def test_policy_without_config_path_has_no_config_key(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(name="S", model=minimal_model)
        scene.add_policy(name="Policy", policy=minimal_onnx)
        builder._save_config_json(tmp_path)
        policy = self._read_config(tmp_path)["projects"][0]["scenes"][0]["policies"][0]
        assert policy["name"] == "Policy"
        assert "config" not in policy

    def test_policy_motion_summary_is_included_in_root_config(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(name="S", model=minimal_model)
        scene.add_policy(name="Policy", policy=minimal_onnx).add_motion(
            name="Spin Kick",
            source="motion.npz",
            anchor_body_name="torso_link",
            body_names=("pelvis", "torso_link"),
            default=True,
        )

        builder._save_config_json(tmp_path)
        policy = self._read_config(tmp_path)["projects"][0]["scenes"][0]["policies"][0]
        assert policy["motions"] == [{"name": "Spin Kick", "default": True}]

    def test_multiple_projects_all_present_in_config(self, tmp_path, minimal_model):
        builder = Builder()
        builder.add_project(name="Project A").add_scene(name="S", model=minimal_model)
        builder.add_project(name="Project B").add_scene(name="S", model=minimal_model)
        builder._save_config_json(tmp_path)
        projects = self._read_config(tmp_path)["projects"]
        assert len(projects) == 2
        assert projects[0]["name"] == "Project A"
        assert projects[1]["name"] == "Project B"

    def test_second_project_auto_id_reflected_in_config(self, tmp_path, minimal_model):
        builder = Builder()
        builder.add_project(name="Main").add_scene(name="S", model=minimal_model)
        builder.add_project(name="MuJoCo Menagerie").add_scene(
            name="S", model=minimal_model
        )
        builder._save_config_json(tmp_path)
        projects = self._read_config(tmp_path)["projects"]
        assert projects[0]["id"] is None
        assert projects[1]["id"] == name2id("MuJoCo Menagerie")


# ===========================================================================
# L1 — _save_web: actions/terminations serialization into policy JSON
# ===========================================================================
class TestSaveWebPolicyJson:
    """Tests for _save_web: verify actions/terminations are emitted into the
    generated policy JSON, covering both the no-config_path and config_path
    branches.  The frontend build and template copy are mocked out so these
    tests remain fast (L1).
    """

    @pytest.fixture(autouse=True)
    def _no_frontend(self, monkeypatch):
        """Skip the Node.js frontend build and the large template copytree."""
        monkeypatch.setattr("mjswan.builder.ClientBuilder", MagicMock())
        monkeypatch.setattr("mjswan.builder.shutil.copytree", MagicMock())

    def _run(self, builder: Builder, tmp_path: Path) -> Path:
        """Call _save_web and return the output directory."""
        out = tmp_path / "out"
        builder._save_web(out)
        return out

    def _policy_json(
        self,
        out: Path,
        policy_name: str,
        scene_name: str = "S",
        project_dir: str = "main",
    ) -> dict:
        scene_id = name2id(scene_name)
        policy_id = name2id(policy_name)
        path = out / project_dir / "assets" / scene_id / f"{policy_id}.json"
        return json.loads(path.read_text())

    # -----------------------------------------------------------------------
    # no-config_path branch
    # -----------------------------------------------------------------------

    def test_no_config_path_actions_emitted(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(name="S", model=minimal_model)
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            actions={
                "joint_pos": JointPositionActionCfg(
                    actuator_names=(".*",), scale=0.5, use_default_offset=True
                ),
            },
        )
        data = self._policy_json(self._run(builder, tmp_path), "Policy")
        assert "actions" in data
        assert "joint_pos" in data["actions"]
        assert data["actions"]["joint_pos"]["type"] == "joint_position"
        assert data["actions"]["joint_pos"]["scale"] == 0.5

    def test_no_config_path_terminations_emitted(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(name="S", model=minimal_model)
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            terminations={
                "time_out": TerminationTermCfg(func=term_fns.time_out, time_out=True),
            },
        )
        data = self._policy_json(self._run(builder, tmp_path), "Policy")
        assert "terminations" in data
        assert "time_out" in data["terminations"]
        assert data["terminations"]["time_out"]["name"] == "TimeOut"
        assert data["terminations"]["time_out"]["time_out"] is True

    def test_no_config_path_both_blocks_emitted(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(name="S", model=minimal_model)
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            actions={
                "effort": JointEffortActionCfg(actuator_names=(".*",), scale=2.0),
            },
            terminations={
                "fallen": TerminationTermCfg(
                    func=term_fns.bad_orientation,
                    params={"limit_angle": 1.2},
                ),
            },
        )
        data = self._policy_json(self._run(builder, tmp_path), "Policy")
        assert "actions" in data
        assert "terminations" in data
        assert data["actions"]["effort"]["type"] == "torque"
        assert data["terminations"]["fallen"]["name"] == "BadOrientation"
        assert data["terminations"]["fallen"]["params"]["limit_angle"] == 1.2

    def test_no_config_path_actions_absent_when_not_set(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(name="S", model=minimal_model)
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            terminations={
                "time_out": TerminationTermCfg(func=term_fns.time_out, time_out=True),
            },
        )
        data = self._policy_json(self._run(builder, tmp_path), "Policy")
        assert "actions" not in data

    def test_no_config_path_terminations_absent_when_not_set(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(name="S", model=minimal_model)
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            actions={
                "joint_pos": JointPositionActionCfg(actuator_names=(".*",)),
            },
        )
        data = self._policy_json(self._run(builder, tmp_path), "Policy")
        assert "terminations" not in data

    def test_no_config_path_no_json_without_mdp_components(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(name="S", model=minimal_model)
        scene.add_policy(name="Policy", policy=minimal_onnx)
        out = self._run(builder, tmp_path)
        policy_id = name2id("Policy")
        scene_id = name2id("S")
        json_path = out / "main" / "assets" / scene_id / f"{policy_id}.json"
        assert not json_path.exists()

    def test_no_config_path_onnx_path_in_json(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(name="S", model=minimal_model)
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            actions={"effort": JointEffortActionCfg(actuator_names=(".*",))},
        )
        data = self._policy_json(self._run(builder, tmp_path), "Policy")
        assert data["onnx"]["path"] == "policy.onnx"

    def test_no_config_path_motions_emitted_and_copied(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        motion_file = tmp_path / "spin_kick.npz"
        motion_file.write_bytes(b"motion-bytes")

        builder = Builder()
        scene = builder.add_project(name="P").add_scene(name="S", model=minimal_model)
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            actions={"joint_pos": JointPositionActionCfg(actuator_names=(".*",))},
        ).add_motion(
            name="Spin Kick",
            source=str(motion_file),
            anchor_body_name="torso_link",
            body_names=("pelvis", "torso_link"),
            dataset_joint_names=["joint_a"],
            default=True,
        )

        out = self._run(builder, tmp_path)
        data = self._policy_json(out, "Policy")
        assert data["motions"] == [
            {
                "name": "Spin Kick",
                "path": "policy_spin_kick.npz",
                "fps": 50.0,
                "anchor_body_name": "torso_link",
                "body_names": ["pelvis", "torso_link"],
                "dataset_joint_names": ["joint_a"],
                "default": True,
            }
        ]
        motion_out = out / "main" / "assets" / "s" / "policy_spin_kick.npz"
        assert motion_out.read_bytes() == b"motion-bytes"

    def test_no_config_path_commands_emitted_as_command_terms(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(name="S", model=minimal_model)
        scene.add_policy(name="Policy", policy=minimal_onnx).add_velocity_command()

        data = self._policy_json(self._run(builder, tmp_path), "Policy")
        assert data["commands"]["velocity"]["name"] == "UiCommand"
        assert len(data["commands"]["velocity"]["ui"]["inputs"]) == 3
        assert data["commands"]["velocity"]["ui"]["inputs"][0]["name"] == "lin_vel_x"

    def test_joint_observation_terms_are_enriched_from_scene_spec(
        self, tmp_path, minimal_onnx
    ):
        xml_path = tmp_path / "scene.xml"
        xml_path.write_text(
            '<mujoco model="jointed">'
            "<worldbody>"
            '<body name="robot/base">'
            '<geom type="sphere" size="0.05" mass="1"/>'
            '<body name="robot/link1">'
            '<joint name="robot/joint1" type="hinge"/>'
            '<geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.02" mass="1"/>'
            '<body name="robot/link2">'
            '<joint name="robot/joint2" type="slide"/>'
            '<geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.02" mass="1"/>'
            "</body>"
            "</body>"
            "</body>"
            "</worldbody>"
            '<keyframe><key name="init" qpos="0.25 -0.5"/></keyframe>'
            "</mujoco>"
        )
        spec = mujoco.MjSpec.from_file(str(xml_path))

        builder = Builder()
        scene = builder.add_project(name="P").add_scene(name="S", spec=spec)
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            observations={
                "policy": ObservationGroupCfg(
                    terms={
                        "joint_pos": ObservationTermCfg(func=obs_fns.joint_pos_rel),
                    }
                ),
            },
        )

        data = self._policy_json(self._run(builder, tmp_path), "Policy")
        joint_pos = data["observations"]["policy"][0]
        assert joint_pos["joint_names"] == ["robot/joint1", "robot/joint2"]
        assert joint_pos["default_joint_pos"] == [0.25, -0.5]

    # -----------------------------------------------------------------------
    # config_path branch
    # -----------------------------------------------------------------------

    def test_config_path_actions_merged_into_existing_config(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        config_file = tmp_path / "policy_cfg.json"
        config_file.write_text(
            json.dumps({"onnx": {"path": "old.onnx"}, "existing_key": "kept"})
        )
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(name="S", model=minimal_model)
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            config_path=str(config_file),
            actions={
                "joint_pos": JointPositionActionCfg(
                    actuator_names=(".*",), use_default_offset=True
                ),
            },
        )
        data = self._policy_json(self._run(builder, tmp_path), "Policy")
        assert "actions" in data
        assert data["actions"]["joint_pos"]["type"] == "joint_position"
        assert data["existing_key"] == "kept"

    def test_config_path_terminations_merged_into_existing_config(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        config_file = tmp_path / "policy_cfg.json"
        config_file.write_text(
            json.dumps({"onnx": {"path": "old.onnx"}, "existing_key": "kept"})
        )
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(name="S", model=minimal_model)
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            config_path=str(config_file),
            terminations={
                "fallen": TerminationTermCfg(
                    func=term_fns.bad_orientation,
                    params={"limit_angle": 0.8},
                ),
            },
        )
        data = self._policy_json(self._run(builder, tmp_path), "Policy")
        assert "terminations" in data
        assert data["terminations"]["fallen"]["name"] == "BadOrientation"
        assert data["terminations"]["fallen"]["params"]["limit_angle"] == 0.8
        assert data["existing_key"] == "kept"

    def test_config_path_both_blocks_merged(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        config_file = tmp_path / "policy_cfg.json"
        config_file.write_text(json.dumps({"onnx": {"path": "old.onnx"}}))
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(name="S", model=minimal_model)
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            config_path=str(config_file),
            actions={
                "effort": JointEffortActionCfg(actuator_names=(".*",), scale=1.5),
            },
            terminations={
                "height": TerminationTermCfg(
                    func=term_fns.root_height_below_minimum,
                    params={"minimum_height": 0.3},
                ),
            },
        )
        data = self._policy_json(self._run(builder, tmp_path), "Policy")
        assert data["actions"]["effort"]["type"] == "torque"
        assert data["actions"]["effort"]["scale"] == 1.5
        assert data["terminations"]["height"]["name"] == "RootHeightBelowMinimum"
        assert data["terminations"]["height"]["params"]["minimum_height"] == 0.3

    def test_config_path_overwrites_existing_actions_block(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        """actions from policy.actions fully replaces any pre-existing actions block."""
        config_file = tmp_path / "policy_cfg.json"
        config_file.write_text(
            json.dumps(
                {
                    "onnx": {"path": "old.onnx"},
                    "actions": {"old_action": {"type": "joint_position"}},
                }
            )
        )
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(name="S", model=minimal_model)
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            config_path=str(config_file),
            actions={
                "new_action": JointPositionActionCfg(actuator_names=(".*",)),
            },
        )
        data = self._policy_json(self._run(builder, tmp_path), "Policy")
        assert "new_action" in data["actions"]
        assert "old_action" not in data["actions"]

    def test_config_path_onnx_path_updated(self, tmp_path, minimal_model, minimal_onnx):
        config_file = tmp_path / "policy_cfg.json"
        config_file.write_text(json.dumps({"onnx": {"path": "stale.onnx"}}))
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(name="S", model=minimal_model)
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            config_path=str(config_file),
            actions={"joint_pos": JointPositionActionCfg(actuator_names=(".*",))},
        )
        data = self._policy_json(self._run(builder, tmp_path), "Policy")
        assert data["onnx"]["path"] == "policy.onnx"

    def test_config_path_onnx_path_updated_when_extras_present(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        config_file = tmp_path / "policy_cfg.json"
        config_file.write_text(json.dumps({"onnx": {"path": "stale.onnx"}}))
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(name="S", model=minimal_model)
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            config_path=str(config_file),
            extras={"model_overrides": {"geom_friction": [1.0, 0.5, 0.25]}},
        )
        data = self._policy_json(self._run(builder, tmp_path), "Policy")
        assert data["onnx"]["path"] == "policy.onnx"
        assert data["extras"] == {
            "model_overrides": {"geom_friction": [1.0, 0.5, 0.25]}
        }

    def test_config_path_actions_absent_from_json_when_not_set(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        config_file = tmp_path / "policy_cfg.json"
        config_file.write_text(json.dumps({"onnx": {"path": "old.onnx"}}))
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(name="S", model=minimal_model)
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            config_path=str(config_file),
        )
        data = self._policy_json(self._run(builder, tmp_path), "Policy")
        assert "actions" not in data
        assert "terminations" not in data

    # -----------------------------------------------------------------------
    # Serialization correctness
    # -----------------------------------------------------------------------

    def test_joint_effort_action_scale_serialized(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(name="S", model=minimal_model)
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            actions={"effort": JointEffortActionCfg(actuator_names=(".*",), scale=3.0)},
        )
        data = self._policy_json(self._run(builder, tmp_path), "Policy")
        assert data["actions"]["effort"] == {
            "type": "torque",
            "scale": 3.0,
            "actuator_names": [".*"],
        }

    def test_joint_position_default_offset_serialized(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(name="S", model=minimal_model)
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            actions={
                "joint_pos": JointPositionActionCfg(
                    actuator_names=(".*",), use_default_offset=False
                ),
            },
        )
        data = self._policy_json(self._run(builder, tmp_path), "Policy")
        assert data["actions"]["joint_pos"]["use_default_offset"] is False

    def test_timeout_termination_time_out_flag(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(name="S", model=minimal_model)
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            terminations={
                "time_out": TerminationTermCfg(func=term_fns.time_out, time_out=True),
            },
        )
        data = self._policy_json(self._run(builder, tmp_path), "Policy")
        term = data["terminations"]["time_out"]
        assert term["name"] == "TimeOut"
        assert term.get("time_out") is True
        assert "params" not in term

    def test_bad_orientation_params_serialized(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(name="S", model=minimal_model)
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            terminations={
                "fallen": TerminationTermCfg(
                    func=term_fns.bad_orientation,
                    params={"limit_angle": 1.57},
                ),
            },
        )
        data = self._policy_json(self._run(builder, tmp_path), "Policy")
        assert data["terminations"]["fallen"]["params"]["limit_angle"] == 1.57
        assert "time_out" not in data["terminations"]["fallen"]


# ===========================================================================
# L3 slow — full build pipeline (triggers frontend compilation)
# Run with: pytest -m slow
# ===========================================================================
@pytest.mark.slow
class TestFullBuild:
    def test_build_creates_assets_config_json(self, tmp_path, minimal_model):
        builder = Builder()
        builder.add_project(name="Test").add_scene(name="Scene", model=minimal_model)
        builder.build(tmp_path / "out")
        assert (tmp_path / "out" / "assets" / "config.json").exists()

    def test_build_with_model_creates_mjb_file(self, tmp_path, minimal_model):
        builder = Builder()
        builder.add_project(name="Test").add_scene(name="Scene", model=minimal_model)
        builder.build(tmp_path / "out")
        scene_dir = tmp_path / "out" / "main" / "assets" / "scene"
        assert (scene_dir / "scene.mjb").exists()

    def test_build_with_spec_creates_mjz_file(self, tmp_path, minimal_spec):
        builder = Builder()
        builder.add_project(name="Test").add_scene(name="Scene", spec=minimal_spec)
        builder.build(tmp_path / "out")
        scene_dir = tmp_path / "out" / "main" / "assets" / "scene"
        assert (scene_dir / "scene.mjz").exists()

    def test_build_project_without_id_uses_main_directory(
        self, tmp_path, minimal_model
    ):
        builder = Builder()
        builder.add_project(name="Test").add_scene(name="S", model=minimal_model)
        builder.build(tmp_path / "out")
        assert (tmp_path / "out" / "main").is_dir()

    def test_build_project_with_id_uses_id_as_directory(self, tmp_path, minimal_model):
        builder = Builder()
        builder.add_project(name="Test", id="demo").add_scene(
            name="S", model=minimal_model
        )
        builder.build(tmp_path / "out")
        assert (tmp_path / "out" / "demo").is_dir()

    def test_build_returns_mjswan_app_instance(self, tmp_path, minimal_model):
        builder = Builder()
        builder.add_project(name="Test").add_scene(name="S", model=minimal_model)
        app = builder.build(tmp_path / "out")
        assert isinstance(app, mjswan.mjswanApp)


# ===========================================================================
# L1 — mt parameter: _save_mt_headers / no-headers when mt=False
# ===========================================================================
class TestMtHeaders:
    def test_mt_defaults_to_false(self):
        assert Builder()._mt is False

    def test_mt_true_stored(self):
        assert Builder(mt=True)._mt is True

    def test_save_mt_headers_creates_headers_file(self, tmp_path):
        Builder()._save_mt_headers(tmp_path)
        assert (tmp_path / "_headers").exists()

    def test_save_mt_headers_contains_coop(self, tmp_path):
        Builder()._save_mt_headers(tmp_path)
        content = (tmp_path / "_headers").read_text()
        assert "Cross-Origin-Opener-Policy: same-origin" in content

    def test_save_mt_headers_contains_coep(self, tmp_path):
        Builder()._save_mt_headers(tmp_path)
        content = (tmp_path / "_headers").read_text()
        assert "Cross-Origin-Embedder-Policy: require-corp" in content

    def test_save_mt_headers_applies_wildcard_route(self, tmp_path):
        Builder()._save_mt_headers(tmp_path)
        content = (tmp_path / "_headers").read_text()
        assert content.startswith("/*")

    def test_mt_false_does_not_write_headers(
        self, tmp_path, minimal_model, monkeypatch
    ):
        """_save_web with mt=False must not create _headers."""
        monkeypatch.setattr("mjswan.builder.ClientBuilder", MagicMock())
        monkeypatch.setattr("mjswan.builder.shutil.copytree", MagicMock())
        builder = Builder(mt=False)
        builder.add_project(name="P").add_scene(name="S", model=minimal_model)
        out = tmp_path / "out"
        builder._save_web(out)
        assert not (out / "_headers").exists()

    def test_mt_true_writes_headers(self, tmp_path, minimal_model, monkeypatch):
        """_save_web with mt=True must create _headers with COOP/COEP content."""
        monkeypatch.setattr("mjswan.builder.ClientBuilder", MagicMock())
        monkeypatch.setattr("mjswan.builder.shutil.copytree", MagicMock())
        builder = Builder(mt=True)
        builder.add_project(name="P").add_scene(name="S", model=minimal_model)
        out = tmp_path / "out"
        builder._save_web(out)
        headers_file = out / "_headers"
        assert headers_file.exists()
        content = headers_file.read_text()
        assert "Cross-Origin-Opener-Policy: same-origin" in content
        assert "Cross-Origin-Embedder-Policy: require-corp" in content

    def test_save_web_excludes_mt_template_dir(
        self, tmp_path, minimal_model, monkeypatch
    ):
        """_save_web must not copy the template-only _mt directory into output."""
        monkeypatch.setattr("mjswan.builder.ClientBuilder", MagicMock())
        copytree = MagicMock()
        monkeypatch.setattr("mjswan.builder.shutil.copytree", copytree)

        builder = Builder(mt=False)
        builder.add_project(name="P").add_scene(name="S", model=minimal_model)
        out = tmp_path / "out"
        builder._save_web(out)

        ignore = copytree.call_args.kwargs["ignore"]
        ignored = set(ignore("", ["_mt", "src", "README.md", "__pycache__"]))
        assert "_mt" in ignored


# ===========================================================================
# L3 slow — full mt=True build (triggers frontend compilation)
# Run with: pytest -m slow
# ===========================================================================
@pytest.mark.slow
class TestFullBuildMt:
    def test_mt_false_no_headers_file(self, tmp_path, minimal_model):
        builder = Builder(mt=False)
        builder.add_project(name="Test").add_scene(name="Scene", model=minimal_model)
        builder.build(tmp_path / "out")
        assert not (tmp_path / "out" / "_headers").exists()

    def test_mt_false_no_coi_serviceworker(self, tmp_path, minimal_model):
        builder = Builder(mt=False)
        builder.add_project(name="Test").add_scene(name="Scene", model=minimal_model)
        builder.build(tmp_path / "out")
        assert not (tmp_path / "out" / "coi-serviceworker.js").exists()

    def test_mt_true_writes_headers_file(self, tmp_path, minimal_model):
        builder = Builder(mt=True)
        builder.add_project(name="Test").add_scene(name="Scene", model=minimal_model)
        builder.build(tmp_path / "out")
        headers = tmp_path / "out" / "_headers"
        assert headers.exists()
        content = headers.read_text()
        assert "Cross-Origin-Opener-Policy: same-origin" in content
        assert "Cross-Origin-Embedder-Policy: require-corp" in content

    def test_mt_true_emits_coi_serviceworker(self, tmp_path, minimal_model):
        builder = Builder(mt=True)
        builder.add_project(name="Test").add_scene(name="Scene", model=minimal_model)
        builder.build(tmp_path / "out")
        assert (tmp_path / "out" / "coi-serviceworker.js").exists()

    def test_mt_true_injects_sw_script_into_html(self, tmp_path, minimal_model):
        builder = Builder(mt=True)
        builder.add_project(name="Test").add_scene(name="Scene", model=minimal_model)
        builder.build(tmp_path / "out")
        html = (tmp_path / "out" / "index.html").read_text()
        assert "coi-serviceworker.js" in html
        assert "crossOriginIsolated" in html


@pytest.mark.slow
class TestFullBuildGtmId:
    def test_gtm_snippet_injected_into_all_html_files(self, tmp_path, minimal_model):
        builder = Builder(gtm_id="GTM-SAMPLE123")
        builder.add_project(name="Test").add_scene(name="Scene", model=minimal_model)
        builder.build(tmp_path / "out")
        out = tmp_path / "out"
        for html_file in [out / "index.html", out / "main" / "index.html"]:
            html = html_file.read_text()
            assert "GTM-SAMPLE123" in html
            assert "googletagmanager.com/gtm.js" in html  # <head> script
            assert "googletagmanager.com/ns.html" in html  # <body> noscript

    def test_no_gtm_without_gtm_id(self, tmp_path, minimal_model):
        builder = Builder()
        builder.add_project(name="Test").add_scene(name="Scene", model=minimal_model)
        builder.build(tmp_path / "out")
        html = (tmp_path / "out" / "index.html").read_text()
        assert "googletagmanager.com" not in html
