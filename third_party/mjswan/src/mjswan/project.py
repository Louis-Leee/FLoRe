"""Project configuration and management.

This module defines the ProjectConfig dataclass and ProjectHandle class for
managing projects containing multiple scenes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import mujoco

from .adapters import apply_mjlab_sim_options, ensure_mjlab_extensions
from .scene import SceneConfig, SceneHandle
from .utils import collect_spec_assets
from .viewer_config import ViewerConfig

if TYPE_CHECKING:
    from .builder import Builder


@dataclass
class ProjectConfig:
    """Configuration for a project containing multiple scenes."""

    name: str
    """Name of the project."""

    id: str | None = None
    """Optional ID for the project used in URL routing (e.g., 'menagerie' for /#/menagerie/)."""

    scenes: list[SceneConfig] = field(default_factory=list)
    """List of scenes in the project."""


class ProjectHandle:
    """Handle for adding scenes and configuring a project.

    This class provides methods for adding scenes and customizing project properties.
    Similar to viser's server handle, this allows for hierarchical configuration.
    """

    def __init__(self, project_config: ProjectConfig, builder: Builder) -> None:
        self._config = project_config
        self._builder = builder

    @property
    def name(self) -> str:
        """Name of the project."""
        return self._config.name

    @property
    def id(self) -> str | None:
        """Optional ID of the project for URL routing."""
        return self._config.id

    def add_scene(
        self,
        name: str,
        *,
        model: mujoco.MjModel | None = None,
        spec: mujoco.MjSpec | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SceneHandle:
        """Add a MuJoCo scene to this project.

        Provide either ``model`` or ``spec`` (not both).

        Using ``model`` saves the scene as a binary ``.mjb`` file, which loads
        faster in the browser but produces larger files. This is recommended
        when loading speed is a priority and storage size is not a concern.

        Using ``spec`` saves the scene as a compressed ``.mjz`` file, which
        uses significantly less storage but may take slightly longer to load.
        This is recommended when the generated web app exceeds 1 GB of storage
        (e.g., the GitHub Pages deployment limit).

        Args:
            name: Name for the scene (displayed in the UI).
            model: MuJoCo model for the scene (saved as .mjb).
            spec: MuJoCo spec for the scene (saved as .mjz).
            metadata: Optional metadata dictionary for the scene.

        Returns:
            SceneHandle for adding policies and further configuration.

        Example:
            ```
            # Fast loading (larger files):
            project.add_scene(
                model=mujoco.MjModel.from_xml_path("scene.xml"),
                name="My Scene",
            )

            # Compact storage (slower loading):
            project.add_scene(
                spec=mujoco.MjSpec.from_file("scene.xml"),
                name="My Scene",
            )
            ```
        """
        if model is not None and spec is not None:
            raise ValueError("Provide either 'model' or 'spec', not both.")
        if model is None and spec is None:
            raise ValueError("Either 'model' or 'spec' must be provided.")

        if metadata is None:
            metadata = {}

        scene_config = SceneConfig(
            name=name,
            model=model,
            spec=spec,
            metadata=metadata,
        )
        self._config.scenes.append(scene_config)
        return SceneHandle(scene_config, self)

    def add_mjlab_scene(self, task_id: str, *, play: bool = False) -> SceneHandle:
        """Add a MuJoCo scene from an mjlab task.

        Loads the task's MuJoCo spec from the mjlab task registry and adds it
        as a scene to this project. ``mjlab`` must be installed.

        Args:
            task_id: mjlab task identifier (e.g. ``"go2_flat"``).
            play: Whether to load mjlab's play/evaluation config instead of the
                training config. This is useful for demos that should match
                mjlab's randomized play terrain layout.

        Returns:
            SceneHandle for further configuration (add_policy, add_splat, etc.)

        Example:
            ```python
            builder = mjswan.Builder()
            project = builder.add_project(name="My App")
            scene = project.add_mjlab_scene("go2_flat", play=True)
            app = builder.build()
            ```
        """
        try:
            from mjlab.scene import Scene
            from mjlab.tasks.registry import load_env_cfg
        except ImportError as e:
            raise ImportError(
                "mjlab is required for add_mjlab_scene(). "
                "Install it with: pip install mjlab"
            ) from e

        ensure_mjlab_extensions()
        env_cfg = load_env_cfg(task_id, play=play)
        env_cfg.scene.num_envs = 1
        scene = Scene(env_cfg.scene, device="cpu")
        scene.spec.assets.update(_collect_mjlab_scene_assets(env_cfg.scene))
        apply_mjlab_sim_options(scene.spec, getattr(env_cfg, "sim", None))
        handle = self.add_scene(spec=scene.spec, name=task_id)
        viewer_cfg = _adapt_mjlab_viewer_config(getattr(env_cfg, "viewer", None))
        if viewer_cfg is not None:
            handle.set_viewer_config(viewer_cfg)
        terrain_data = _extract_terrain_data(scene)
        if terrain_data:
            handle._config.terrain_data = terrain_data
        events = getattr(env_cfg, "events", None)
        if events:
            handle.set_events(events)
        if terrain_data and handle._config.events:
            _upgrade_spawn_events_for_terrain(handle._config, terrain_data)
        return handle


def _upgrade_spawn_events_for_terrain(
    scene_config: SceneConfig, terrain_data: dict[str, Any]
) -> None:
    """Replace ResetRootStateUniform with ResetRootStateFromFlatPatches when flat patches exist.

    During training, many parallel envs are distributed across terrain tiles so
    per-env x/y randomization can be small. In the browser there is only one env,
    so we upgrade to patch-based spawning to cover the full terrain.
    """
    flat_patches = terrain_data.get("flat_patches", {})
    if not flat_patches:
        return
    patch_name = "spawn" if "spawn" in flat_patches else next(iter(flat_patches))
    events = scene_config.events
    if not events:
        return
    for i, event in enumerate(events):
        if event.get("name") == "ResetRootStateUniform":
            new_params: dict[str, Any] = dict(event.get("params") or {})
            new_params["patch_name"] = patch_name
            events[i] = {"name": "ResetRootStateFromFlatPatches", "params": new_params}


def _extract_terrain_data(scene: Any) -> dict[str, Any] | None:
    """Extract spawn positions from a mjlab Scene for browser-side event execution.

    Tries named flat_patches first (higher-quality sampled positions); falls back
    to terrain_origins (one per sub-terrain tile) when flat_patch_sampling is not
    configured on any sub-terrain.
    """
    terrain = getattr(scene, "terrain", None)
    if terrain is None:
        return None

    # Try explicit flat_patches (only present when flat_patch_sampling is configured).
    flat_patches = getattr(terrain, "flat_patches", None)
    if flat_patches:
        serialized: dict[str, list[list[float]]] = {}
        for name, patches in flat_patches.items():
            # patches: (num_rows, num_cols, num_patches, 3) tensor
            try:
                arr = patches.cpu().numpy()
                rows, cols, n, _ = arr.shape
                positions = arr.reshape(rows * cols * n, 3).tolist()
                serialized[name] = positions
            except Exception:
                pass
        if serialized:
            return {"flat_patches": serialized}

    # Fall back to terrain_origins (one spawn point per sub-terrain tile).
    terrain_origins = getattr(terrain, "terrain_origins", None)
    if terrain_origins is not None:
        try:
            arr = terrain_origins.cpu().numpy()
            # shape: (num_rows, num_cols, 3)
            num_rows, num_cols, _ = arr.shape
            positions = arr.reshape(num_rows * num_cols, 3).tolist()
            return {"flat_patches": {"spawn": positions}}
        except Exception:
            pass

    return None


def _collect_mjlab_scene_assets(scene_cfg: Any) -> dict[str, bytes]:
    """Collect assets from mjlab scene component specs before they are flattened."""
    assets: dict[str, bytes] = {}

    spec_cfgs = [getattr(scene_cfg, "terrain", None)]
    entities = getattr(scene_cfg, "entities", {})
    if isinstance(entities, dict):
        spec_cfgs.extend(entities.values())

    for cfg in spec_cfgs:
        spec_fn = getattr(cfg, "spec_fn", None)
        if not callable(spec_fn):
            continue
        spec = spec_fn()
        if not isinstance(spec, mujoco.MjSpec):
            continue
        assets.update(collect_spec_assets(spec))

    return assets


def _adapt_mjlab_viewer_config(config: Any | None) -> ViewerConfig | None:
    """Convert mjlab's ``ViewerConfig`` dataclass to mjswan's equivalent."""
    if config is None:
        return None

    defaults = ViewerConfig()
    entity_name = getattr(config, "entity_name", None)
    body_name = getattr(config, "body_name", None)
    if entity_name is None and body_name is not None:
        entity_name = "robot"
    origin_type_name = getattr(getattr(config, "origin_type", None), "name", None)
    if isinstance(origin_type_name, str):
        origin_type = getattr(ViewerConfig.OriginType, origin_type_name, None)
    else:
        origin_type = None

    return ViewerConfig(
        lookat=tuple(getattr(config, "lookat", (0.0, 0.0, 0.0))),
        distance=float(getattr(config, "distance", 4.0)),
        fovy=getattr(config, "fovy", None),
        elevation=float(getattr(config, "elevation", -30.0)),
        azimuth=float(getattr(config, "azimuth", 45.0)),
        origin_type=origin_type or defaults.origin_type,
        entity_name=entity_name,
        body_name=body_name,
        env_idx=int(getattr(config, "env_idx", 0)),
        max_extra_envs=int(getattr(config, "max_extra_envs", 2)),
        enable_reflections=bool(getattr(config, "enable_reflections", True)),
        enable_shadows=bool(getattr(config, "enable_shadows", True)),
        height=int(getattr(config, "height", 240)),
        width=int(getattr(config, "width", 320)),
    )


__all__ = ["ProjectConfig", "ProjectHandle"]
