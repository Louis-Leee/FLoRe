"""Common helpers for holosoma extension asset prefix resolution."""

from __future__ import annotations

import os
from dataclasses import replace


def resolve_prefix_path(path: str, token: str, ext_root: str) -> str:
    """Replace a token prefix (e.g., @holosoma_ext_foo) with the extension root."""
    if not path or token not in path:
        return path
    suffix = path.split(token, 1)[1].lstrip("/\\")
    return os.path.join(ext_root, suffix)


def resolve_robot_asset_paths(robot_config, token: str, ext_root: str):
    """Return a robot_config with asset paths resolved for a given token."""
    try:
        asset = getattr(robot_config, "asset", None)
        if asset is None:
            return robot_config

        resolved_root = resolve_prefix_path(asset.asset_root, token, ext_root)
        resolved_urdf = resolve_prefix_path(asset.urdf_file, token, ext_root)
        resolved_xml = resolve_prefix_path(asset.xml_file, token, ext_root) if asset.xml_file else asset.xml_file

        if resolved_root != asset.asset_root or resolved_urdf != asset.urdf_file or resolved_xml != asset.xml_file:
            new_asset = replace(asset, asset_root=resolved_root, urdf_file=resolved_urdf, xml_file=resolved_xml)
            return replace(robot_config, asset=new_asset)
    except Exception:
        return robot_config
    return robot_config


def resolve_training_robot_defaults(defaults: dict, token: str, ext_root: str) -> None:
    """In-place resolution of @-prefixed asset paths in training DEFAULTS dict."""
    for name, cfg in list(defaults.items()):
        try:
            asset = getattr(cfg, "asset", None)
            if asset is None:
                continue
            resolved_root = resolve_prefix_path(asset.asset_root, token, ext_root)
            resolved_urdf = resolve_prefix_path(asset.urdf_file, token, ext_root)
            resolved_xml = resolve_prefix_path(asset.xml_file, token, ext_root) if asset.xml_file else None

            if resolved_root != asset.asset_root or resolved_urdf != asset.urdf_file or resolved_xml != asset.xml_file:
                new_asset = replace(asset, asset_root=resolved_root, urdf_file=resolved_urdf, xml_file=resolved_xml)
                defaults[name] = replace(cfg, asset=new_asset)
        except Exception:
            continue


def resolve_inference_robot_defaults(defaults: dict, token: str, ext_root: str) -> None:
    """In-place resolution of @-prefixed asset paths in inference DEFAULTS dict."""
    for name, cfg in list(defaults.items()):
        try:
            asset_root = getattr(cfg, "asset_root", None) or getattr(getattr(cfg, "asset", None), "asset_root", None)
            asset_file = getattr(cfg, "asset_file", None) or getattr(getattr(cfg, "asset", None), "asset_file", None)
            scene_xml = getattr(cfg, "robot_scene_xml", None)

            resolved_root = resolve_prefix_path(asset_root, token, ext_root) if asset_root else asset_root
            resolved_file = resolve_prefix_path(asset_file, token, ext_root) if asset_file else asset_file
            resolved_scene = resolve_prefix_path(scene_xml, token, ext_root) if scene_xml else scene_xml

            kwargs = {}
            updated = False
            if resolved_root and resolved_root != asset_root:
                kwargs["asset_root"] = resolved_root
                updated = True
            if resolved_file and resolved_file != asset_file:
                kwargs["asset_file"] = resolved_file
                updated = True
            if resolved_scene and resolved_scene != scene_xml:
                kwargs["robot_scene_xml"] = resolved_scene
                updated = True

            if updated:
                defaults[name] = replace(cfg, **kwargs)
        except Exception:
            continue
