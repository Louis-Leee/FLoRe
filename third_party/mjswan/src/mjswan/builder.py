"""Builder class for constructing mjswan applications.

This module provides the main Builder class which serves as the entry point
for programmatically creating interactive MuJoCo simulations.
"""

from __future__ import annotations

import gc
import inspect
import json
import shutil
import warnings
from pathlib import Path

import mujoco
import onnx
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)

from . import __version__
from ._build_client import ClientBuilder
from .app import mjswanApp
from .project import ProjectConfig, ProjectHandle
from .scene import SceneConfig
from .splat import SplatConfig
from .utils import collect_spec_assets, name2id, to_zip_deflated


class Builder:
    """Builder for creating mjswan applications.

    The Builder class provides a fluent API for programmatically constructing
    interactive MuJoCo simulations with ONNX policies. It handles projects, scenes, and policies hierarchically.
    """

    def __init__(
        self,
        base_path: str = "/",
        gtm_id: str | None = None,
        mt: bool = False,
        debug: bool = False,
    ) -> None:
        """Initialize a new Builder instance.

        Args:
            base_path: Base path for subdirectory deployment (e.g., '/mjswan/').
            gtm_id: Google Tag Manager ID (e.g., 'GTM-XXXXXXX'). Injects GTM snippet if set.
            mt: Enable multi-threaded MuJoCo WASM. Requires COOP/COEP headers — these are
                written as a ``_headers`` file (Netlify/Cloudflare Pages/Vercel) and a
                service worker (required for GitHub Pages hosting). Defaults to False.
            debug: Keep browser console messages in the built app. Defaults to False
                (console messages are stripped from the production bundle).
        """
        self._projects: list[ProjectConfig] = []
        self._base_path = base_path
        self._gtm_id = gtm_id
        self._mt = mt
        self._debug = debug

    @classmethod
    def from_mjlab(
        cls,
        task_id: str,
        *,
        run_path: str | list[str] | None = None,
        project_name: str = "mjlab",
        play: bool = False,
        base_path: str = "/",
        gtm_id: str | None = None,
        mt: bool = False,
        debug: bool = False,
    ) -> Builder:
        """Create a Builder pre-configured with a single mjlab task.

        This is a convenience factory for the common pattern of visualizing one
        mjlab task. The returned Builder can be further modified before calling
        :meth:`build`.

        Args:
            task_id: mjlab task identifier (e.g. ``"go2_flat"``).
            run_path: Optional W&B run path (``"entity/project/run_id"``) or a
                list of such paths. When provided, all ``model_*.pt``
                checkpoints from each run are fetched and converted to ONNX
                via mjlab+torch (both required). ``task_id`` above is reused
                for the conversion. Defaults to ``None`` (no policy attached).
                For finer control (e.g. ``only_latest=True``, custom
                observations/actions), drop down to
                ``builder.get_projects()[0].scenes[0].add_policy_from_wandb(...)``.
            project_name: Name for the auto-created project. Defaults to ``"mjlab"``.
            play: Whether to load mjlab's play/evaluation config instead of the
                training config for the auto-created scene.
            base_path: Base path for the application (e.g., ``"/mjswan/"``).
            gtm_id: Optional Google Tag Manager container ID.

        Returns:
            Builder with one project and one scene already configured.

        Example:
            ```python
            # Minimal usage
            app = mjswan.Builder.from_mjlab("go2_flat").build()
            app.launch()

            # Load mjlab scene and attach all checkpoints from a W&B run
            app = mjswan.Builder.from_mjlab(
                "Mjlab-Velocity-Flat-Anymal-C",
                run_path="ttktjmt-org/mjlab/dqxvf0eb",
            ).build()

            # Customise before building
            builder = mjswan.Builder.from_mjlab("go2_flat")
            scene = builder.get_projects()[0].scenes[0]  # access SceneConfig
            app = builder.build()
            ```
        """
        builder = cls(base_path=base_path, gtm_id=gtm_id, mt=mt, debug=debug)
        project = builder.add_project(name=project_name)
        scene = project.add_mjlab_scene(task_id, play=play)
        if run_path is not None:
            scene.add_policy_from_wandb(run_path, task_id=task_id)
        return builder

    def add_project(self, name: str, *, id: str | None = None) -> ProjectHandle:
        """Add a new project to the builder.

        Args:
            name: Name for the project (displayed in the UI).
            id: Optional ID for URL routing. If not provided, the first project
                defaults to None (main route), and subsequent projects default to sanitized name.

        Returns:
            ProjectHandle for adding scenes and further configuration.
        """
        # Project ID: explicit > None for first project (main route) > sanitized name
        if id is not None:
            project_id = id
        elif not self._projects:
            project_id = None
        else:
            project_id = name2id(name)

        project = ProjectConfig(name=name, id=project_id)
        self._projects.append(project)
        return ProjectHandle(project, self)

    def build(self, output_dir: str | Path | None = None) -> mjswanApp:
        """Build the application from the configured projects.

        This method finalizes the configuration and creates a mjswanApp
        instance. If output_dir is provided, it also saves the application
        to that directory. If output_dir is not provided, it defaults to
        'dist' in the caller's directory.

        Args:
            output_dir: Optional directory to save the application files.
                       If None, defaults to 'dist' in the caller's directory.

        Returns:
            mjswanApp instance ready to be launched.
        """
        if not self._projects:
            raise ValueError(
                "Cannot build an empty application. "
                "You must add at least one project using builder.add_project() before building.\n"
                "Example:\n"
                "  builder = mwx.Builder()\n"
                "  project = builder.add_project(name='My Project')\n"
                "  scene = project.add_scene(spec=mujoco_spec, name='Scene 1')\n"
                "  app = builder.build()"
            )

        # Get caller's file path
        frame = inspect.stack()[1]
        caller_file = frame.filename
        # Handle REPL or interactive mode where filename might be <stdin> or similar
        if caller_file.startswith("<") and caller_file.endswith(">"):
            base_dir = Path.cwd()
        else:
            base_dir = Path(caller_file).parent

        if output_dir is None:
            output_path = base_dir / "dist"
        else:
            # Resolve relative paths against the caller's directory
            output_path = base_dir / Path(output_dir)

        # TODO: Build with separate function (and then save the web app with _save_web). And set scene.path and policy.path after building.
        self._save_web(output_path)

        return mjswanApp(output_path)

    def _save_config_json(self, output_path: Path) -> None:
        """Save configuration as JSON.

        Creates root assets/config.json with project metadata and structure information.
        Individual project assets (scenes/policies) are saved under project-id/assets/.
        """
        # Create root config with project metadata and structure info
        root_config = {
            "version": __version__,
            "projects": [
                {
                    "name": project.name,
                    "id": project.id,
                    "scenes": [
                        {
                            "name": scene.name,
                            "path": f"{name2id(scene.name)}/{scene.scene_filename}",
                            **(
                                {
                                    "splats": [
                                        self._build_splat_config_dict(scene, s)
                                        for s in scene.splats
                                    ]
                                }
                                if scene.splats
                                else {}
                            ),
                            **(
                                {"splatSection": True}
                                if scene.splat_section and not scene.splats
                                else {}
                            ),
                            **(
                                {"camera": scene.viewer.to_dict()}
                                if scene.viewer and scene.viewer.to_dict()
                                else {}
                            ),
                            **({"events": scene.events} if scene.events else {}),
                            **(
                                {"terrainData": scene.terrain_data}
                                if scene.terrain_data
                                else {}
                            ),
                            "policies": [
                                (
                                    {
                                        "name": policy.name,
                                        **(
                                            {
                                                "config": f"{name2id(scene.name)}/"
                                                f"{name2id(policy.name)}.json"
                                            }
                                            if getattr(policy, "config_path", None)
                                            or getattr(policy, "commands", None)
                                            or getattr(policy, "observations", None)
                                            or getattr(policy, "actions", None)
                                            or getattr(policy, "terminations", None)
                                            or getattr(policy, "motions", None)
                                            else {}
                                        ),
                                        **(
                                            {"source": policy.source_path}
                                            if getattr(policy, "source_path", None)
                                            else {}
                                        ),
                                        **(
                                            {"default": True}
                                            if getattr(policy, "default", False)
                                            else {}
                                        ),
                                        **(
                                            {
                                                "motions": [
                                                    motion.to_summary_dict()
                                                    for motion in policy.motions
                                                ]
                                            }
                                            if getattr(policy, "motions", None)
                                            else {}
                                        ),
                                    }
                                )
                                for policy in scene.policies
                            ],
                        }
                        for scene in project.scenes
                    ],
                }
                for project in self._projects
            ],
        }

        # Save root config.json in assets directory
        assets_dir = output_path / "assets"
        assets_dir.mkdir(exist_ok=True)
        root_config_file = assets_dir / "config.json"
        with open(root_config_file, "w") as f:
            json.dump(root_config, f, indent=2)

    def _save_mt_headers(self, output_path: Path) -> None:
        """Write COOP/COEP response headers needed by multi-threaded MuJoCo.

        Two mechanisms are written so the output works on any static host:
        - ``_headers``: honored by Netlify, Cloudflare Pages, and Vercel.
        - ``coi-serviceworker.js`` (emitted by the Vite build only when mt=True): used by the
          injected inline script for GitHub Pages, which cannot set response headers.
        """
        headers_content = (
            "/*\n"
            "  Cross-Origin-Opener-Policy: same-origin\n"
            "  Cross-Origin-Embedder-Policy: require-corp\n"
            "\n"
        )
        (output_path / "_headers").write_text(headers_content)

    def _build_splat_config_dict(self, scene: SceneConfig, splat: SplatConfig) -> dict:
        """Build the splat dict for config.json.

        When ``source`` is set the file is copied to the scene asset directory
        during :meth:`_save_web`, and the resulting relative path is injected
        here so the frontend can resolve it to a URL.
        """
        d = splat.to_dict()
        if splat.source is not None:
            d["path"] = f"{name2id(scene.name)}/{name2id(splat.name)}.spz"
        return d

    def _policy_filename(self, name: str) -> str:
        if not name or name.strip() == "":
            raise ValueError("Policy name must be a non-empty string.")
        if "/" in name or "\\" in name:
            raise ValueError(
                "Policy name cannot contain path separators ('/' or '\\')."
            )
        return name

    def _motion_filename(self, policy_name: str, motion_name: str) -> str:
        if not motion_name or motion_name.strip() == "":
            raise ValueError("Motion name must be a non-empty string.")
        return f"{name2id(policy_name)}_{name2id(motion_name)}.npz"

    def _save_web(self, output_path: Path) -> None:
        """Save as a complete web application.

        Output structure:
            dist/
            ├── index.html
            ├── logo.svg
            ├── manifest.json
            ├── robots.txt
            ├── assets/
            │   ├── config.json
            │   └── (compiled js/css files)
            └── <project-id>/ (or 'main')
                ├── index.html
                ├── logo.svg
                ├── manifest.json
                └── assets/
                    └── <scene-id>/
                        ├── scene.mjz/.mjb
                        ├── <policy-id>.onnx
                        ├── <policy-id>.json
                        └── <splat-id>.spz  (when local source provided)
        """
        if output_path.exists():
            shutil.rmtree(output_path)

        output_path.mkdir(parents=True, exist_ok=True)

        # Copy template directory
        template_dir = Path(__file__).parent / "template"
        if template_dir.exists():
            # Build client first
            package_json = template_dir / "package.json"
            if package_json.exists():
                print("Building the mjswan application...")
                builder = ClientBuilder(template_dir)
                builder.build(
                    base_path=self._base_path,
                    gtm_id=self._gtm_id,
                    mt=self._mt,
                    debug=self._debug,
                )

            # Copy all files from template to output_path
            shutil.copytree(
                template_dir,
                output_path,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(
                    ".nodeenv", "__pycache__", "*.pyc", ".md", "_mt"
                ),
            )

            # Move built files from nested dist/ to output_path root
            built_dist = output_path / "dist"
            if built_dist.exists() and built_dist.is_dir():
                # Move all files from dist/ to output_path
                for item in built_dist.iterdir():
                    dest = output_path / item.name
                    if dest.exists():
                        if dest.is_dir():
                            shutil.rmtree(dest)
                        else:
                            dest.unlink()
                    shutil.move(str(item), str(output_path))
                # Remove the now-empty dist directory
                built_dist.rmdir()

                # Clean up development files that shouldn't be in production
                dev_files = [
                    "src",
                    "node_modules",
                    ".nodeenv",
                    "package.json",
                    "package-lock.json",
                    "tsconfig.json",
                    "vite.config.ts",
                    "eslint.config.cjs",
                    ".browserslistrc",
                    ".gitignore",
                    "README.md",
                ]
                for dev_file in dev_files:
                    dev_path = output_path / dev_file
                    if dev_path.exists():
                        if dev_path.is_dir():
                            shutil.rmtree(dev_path)
                        else:
                            dev_path.unlink()

                # Remove public directory after build
                public_dir = output_path / "public"
                if public_dir.exists():
                    shutil.rmtree(public_dir)
        else:
            warnings.warn(
                f"Template directory not found at {template_dir}.",
                category=RuntimeWarning,
            )

        # Create root assets directory for shared config
        assets_dir = output_path / "assets"
        assets_dir.mkdir(exist_ok=True)

        # Save root configuration (project metadata and structure)
        self._save_config_json(output_path)

        # Write COOP/COEP headers for multi-threaded MuJoCo (SharedArrayBuffer)
        if self._mt:
            self._save_mt_headers(output_path)

        # Save MuJoCo models and ONNX policies per project
        max_name_len = max(len(p.name) for p in self._projects)
        for project in self._projects:
            # Use 'main' for projects without ID, otherwise use the project ID
            project_dir_name = project.id if project.id else "main"
            project_dir = output_path / project_dir_name
            project_assets_dir = project_dir / "assets"

            # Create directories
            project_assets_dir.mkdir(parents=True, exist_ok=True)

            # Copy index.html to each project directory so direct navigation works
            root_index = output_path / "index.html"
            if root_index.exists():
                shutil.copy(str(root_index), str(project_dir / "index.html"))

            # Copy static root assets
            for static_name in ["manifest.json", "logo.svg", "logo-color.svg"]:
                src_static = output_path / static_name
                if src_static.exists():
                    shutil.copy(str(src_static), str(project_dir / static_name))

            # Save scenes and policies
            with Progress(
                SpinnerColumn(spinner_name="dots"),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("[dim]{task.fields[scene]}"),
            ) as progress:
                task = progress.add_task(
                    project.name.ljust(max_name_len),
                    total=len(project.scenes),
                    scene="",
                )
                for scene in project.scenes:
                    progress.update(task, scene=scene.name)
                    scene_id = name2id(scene.name)
                    scene_dir = project_assets_dir / scene_id
                    scene_dir.mkdir(parents=True, exist_ok=True)
                    scene_path = scene_dir / scene.scene_filename
                    if scene.spec is not None:
                        scene.spec.assets.update(collect_spec_assets(scene.spec))
                        to_zip_deflated(scene.spec, str(scene_path))  # Saves as .mjz
                        scene.spec = None
                    else:
                        if scene.model is None:
                            raise RuntimeError(
                                f"Scene '{scene.name}' has no model to save as .mjb"
                            )
                        mujoco.mj_saveModel(
                            scene.model, str(scene_path)
                        )  # Saves as .mjb
                        scene.model = None
                    gc.collect()

                    # Save policies
                    for policy in scene.policies:
                        policy_id = name2id(policy.name)
                        policy_path = scene_dir / f"{policy_id}.onnx"
                        onnx.save(policy.model, str(policy_path))

                        config_path = getattr(policy, "config_path", None)
                        if config_path:
                            config_src = Path(config_path).expanduser()
                            if not config_src.is_absolute():
                                config_src = (Path.cwd() / config_src).resolve()
                            if config_src.exists():
                                target = policy_path.with_suffix(".json")
                                try:
                                    with open(config_src, "r") as f:
                                        data = json.load(f)
                                    data.setdefault("onnx", {})
                                    if isinstance(data["onnx"], dict):
                                        onnx_config = data["onnx"]
                                        onnx_config["path"] = policy_path.name
                                        meta = dict(onnx_config.get("meta") or {})
                                        if "in_keys" in data and "in_keys" not in meta:
                                            meta["in_keys"] = data["in_keys"]
                                        if (
                                            "out_keys" in data
                                            and "out_keys" not in meta
                                        ):
                                            meta["out_keys"] = data["out_keys"]
                                        if meta:
                                            onnx_config["meta"] = meta
                                    # Serialize commands if any are defined
                                    if policy.commands:
                                        data["commands"] = {
                                            name: cmd.to_dict()
                                            for name, cmd in policy.commands.items()
                                        }
                                    # Merge observation groups into observations
                                    if policy.observations:
                                        obs_config = data.get("observations", {})
                                        for key, group in policy.observations.items():
                                            # Avoid overwriting existing groups
                                            # (e.g. ONNX "policy" group from config_path)
                                            target_key = key
                                            if target_key in obs_config:
                                                target_key = f"{key}_monitor"
                                            obs_config[target_key] = group.to_list()
                                        data["observations"] = obs_config
                                    # Serialize action terms
                                    if policy.actions:
                                        data["actions"] = {
                                            name: cfg.to_dict()
                                            for name, cfg in policy.actions.items()
                                        }
                                    if getattr(policy, "policy_joint_names", None):
                                        data["policy_joint_names"] = (
                                            policy.policy_joint_names
                                        )
                                    if getattr(policy, "default_joint_pos", None):
                                        data["default_joint_pos"] = (
                                            policy.default_joint_pos
                                        )
                                    if getattr(policy, "encoder_bias", None):
                                        data["encoder_bias"] = policy.encoder_bias
                                    if getattr(policy, "initial_qpos", None):
                                        data["initial_qpos"] = policy.initial_qpos
                                    if getattr(policy, "initial_qvel", None):
                                        data["initial_qvel"] = policy.initial_qvel
                                    if getattr(policy, "extras", None):
                                        data["extras"] = policy.extras
                                    if getattr(policy, "motions", None):
                                        data["motions"] = [
                                            motion.to_dict(
                                                self._motion_filename(
                                                    policy.name, motion.name
                                                )
                                            )
                                            for motion in policy.motions
                                        ]
                                    # Serialize termination terms
                                    if policy.terminations:
                                        data["terminations"] = {
                                            name: cfg.to_dict()
                                            for name, cfg in policy.terminations.items()
                                        }
                                    with open(target, "w") as f:
                                        json.dump(data, f, indent=2)
                                except Exception:
                                    shutil.copy(str(config_src), str(target))
                            else:
                                warnings.warn(
                                    f"Policy config path not found: {config_src}",
                                    category=RuntimeWarning,
                                    stacklevel=2,
                                )
                        elif (
                            policy.commands
                            or policy.observations
                            or policy.actions
                            or policy.terminations
                            or policy.policy_joint_names
                            or policy.motions
                        ):
                            # No config_path but MDP components defined
                            target = policy_path.with_suffix(".json")
                            data: dict = {
                                "onnx": {"path": policy_path.name},
                            }
                            if policy.policy_joint_names:
                                data["policy_joint_names"] = policy.policy_joint_names
                            if policy.default_joint_pos:
                                data["default_joint_pos"] = policy.default_joint_pos
                            if policy.encoder_bias:
                                data["encoder_bias"] = policy.encoder_bias
                            if policy.commands:
                                data["commands"] = {
                                    name: cmd.to_dict()
                                    for name, cmd in policy.commands.items()
                                }
                            if policy.observations:
                                data["observations"] = {
                                    key: group.to_list()
                                    for key, group in policy.observations.items()
                                }
                            if policy.actions:
                                data["actions"] = {
                                    name: cfg.to_dict()
                                    for name, cfg in policy.actions.items()
                                }
                            if policy.terminations:
                                terminations = {
                                    name: cfg.to_dict()
                                    for name, cfg in policy.terminations.items()
                                    if cfg.func.unsupported_reason is None
                                }
                                if terminations:
                                    data["terminations"] = terminations
                            if policy.motions:
                                data["motions"] = [
                                    motion.to_dict(
                                        self._motion_filename(policy.name, motion.name)
                                    )
                                    for motion in policy.motions
                                ]
                            with open(target, "w") as f:
                                json.dump(data, f, indent=2)

                        seen_motion_files: set[str] = set()
                        for motion in policy.motions:
                            filename = self._motion_filename(policy.name, motion.name)
                            if filename in seen_motion_files:
                                raise ValueError(
                                    f"Motion filename collision for policy '{policy.name}': "
                                    f"'{motion.name}' sanitizes to '{filename}' which is already used. "
                                    "Rename one of the motions to avoid this conflict."
                                )
                            seen_motion_files.add(filename)
                            target = scene_dir / filename
                            if motion.data is not None:
                                target.write_bytes(motion.data)
                            elif motion.source is not None:
                                src = Path(motion.source).expanduser()
                                if not src.is_absolute():
                                    src = (Path.cwd() / src).resolve()
                                if src.exists():
                                    shutil.copy2(str(src), str(target))
                                else:
                                    warnings.warn(
                                        f"Motion source file not found: {src}",
                                        category=RuntimeWarning,
                                        stacklevel=2,
                                    )

                    # Copy bundled .spz files for each splat with source set
                    for splat in scene.splats:
                        if splat.source is not None:
                            src = Path(splat.source).expanduser()
                            if not src.is_absolute():
                                src = (Path.cwd() / src).resolve()
                            if src.exists():
                                shutil.copy2(
                                    str(src),
                                    str(scene_dir / f"{name2id(splat.name)}.spz"),
                                )
                            else:
                                warnings.warn(
                                    f"Splat source file not found: {src}",
                                    category=RuntimeWarning,
                                    stacklevel=2,
                                )

                    progress.advance(task)

        print(f"✓ Saved mjswan application to: {output_path}")

    def get_projects(self) -> list[ProjectConfig]:
        """Get a copy of all project configurations.

        Returns:
            List of ProjectConfig objects.
        """
        return self._projects.copy()


__all__ = ["Builder"]
