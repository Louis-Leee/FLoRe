"""Automatic Node.js environment setup and client build management.

This module handles:
- Creating isolated Node.js environments using nodeenv
- Installing dependencies
- Building TypeScript/JavaScript clients
- Cross-platform compatibility (Windows/macOS/Linux)
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

__all__ = ["ClientBuilder", "ensure_node_env", "build_client"]


class ClientBuilder:
    """Manages isolated Node.js environment and client builds."""

    NODE_VERSION = "25.5.0"

    def __init__(self, project_dir: Path) -> None:
        self.project_dir = Path(project_dir).resolve()
        self.nodeenv_dir = self.project_dir / ".nodeenv"

    def _get_node_bin(self) -> Path:
        if sys.platform == "win32":
            return self.nodeenv_dir / "Scripts" / "node.exe"
        else:
            return self.nodeenv_dir / "bin" / "node"

    def _get_npm_bin(self) -> Path:
        if sys.platform == "win32":
            return self.nodeenv_dir / "Scripts" / "npm.cmd"
        else:
            return self.nodeenv_dir / "bin" / "npm"

    def _node_env(self) -> dict[str, str]:
        """``os.environ`` with the nodeenv's ``bin/`` first on PATH.

        ``npm`` is a ``#!/usr/bin/env node`` script: without this it runs under whichever
        ``node`` the shell finds first, or dies with exit 127 when there is none, and a node
        other than NODE_VERSION silently drops rolldown's optional native binding, so the
        vite build fails later.
        """
        env = os.environ.copy()
        env["PATH"] = str(self._get_node_bin().parent) + os.pathsep + env.get("PATH", "")
        return env

    def _ensure_nodeenv_installed(self) -> None:
        try:
            import nodeenv  # noqa: F401
        except ImportError:
            print("Installing nodeenv...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "nodeenv>=1.9.0"],
                stdout=subprocess.PIPE if not os.getenv("VERBOSE_BUILD") else None,
            )

    def create_env(self, clean: bool = False) -> None:
        if clean and self.nodeenv_dir.exists():
            print(f"Removing existing nodeenv: {self.nodeenv_dir}")
            shutil.rmtree(self.nodeenv_dir)

        if self.nodeenv_dir.exists():
            node_bin = self._get_node_bin()
            if node_bin.exists():
                try:
                    result = subprocess.run(
                        [str(node_bin), "--version"],
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if result.returncode == 0:
                        installed_version = result.stdout.strip().lstrip("v")
                        if installed_version == self.NODE_VERSION:
                            print(f"✓ Node.js {self.NODE_VERSION} already available")
                            return
                except Exception as e:
                    print(f"Warning: Could not verify Node.js version: {e}")

        print(f"Creating Node.js {self.NODE_VERSION} environment in {self.nodeenv_dir}")
        self._ensure_nodeenv_installed()

        # Use nodeenv CLI for robustness across versions
        try:
            cmd = [
                sys.executable,
                "-m",
                "nodeenv",
                str(self.nodeenv_dir),
                "--node",
                self.NODE_VERSION,
            ]
            if os.getenv("VERBOSE_BUILD"):
                cmd.append("--verbose")
            subprocess.check_call(cmd)
        except Exception as e:
            raise RuntimeError(f"Failed to create Node.js environment: {e}")

    def install_dependencies(self, clean: bool = False) -> None:
        npm_bin = self._get_npm_bin()
        package_lock = self.project_dir / "package-lock.json"
        node_modules = self.project_dir / "node_modules"

        if clean:
            # Force a fresh install by removing the lock file and node_modules.
            # Useful when switching platforms or resolving corrupted installs.
            if package_lock.exists():
                package_lock.unlink()
            if node_modules.exists():
                shutil.rmtree(node_modules)

        print("Installing npm dependencies (npm install)...")
        subprocess.check_call([str(npm_bin), "install"], cwd=self.project_dir, env=self._node_env())

    def sync_version_from_python(self) -> None:
        """Sync package.json version with Python package __version__."""
        from mjswan import __version__

        package_json = self.project_dir / "package.json"
        with open(package_json, "r") as f:
            package_data = json.load(f)

        current_version = package_data.get("version", "0.0.0")
        if current_version != __version__:
            print(f"Updating package.json version: {current_version} → {__version__}")
            package_data["version"] = __version__
            # Remove private field if it exists
            package_data.pop("private", None)
            with open(package_json, "w") as f:
                json.dump(package_data, f, indent=2)
                f.write("\n")

    def run_build_script(
        self, script_name: str = "build", env: dict[str, str] | None = None
    ) -> None:
        npm_bin = self._get_npm_bin()
        package_json = self.project_dir / "package.json"
        with open(package_json) as f:
            package_data = json.load(f)
        if script_name not in package_data.get("scripts", {}):
            raise ValueError(
                f"Script '{script_name}' not found in {package_json}. "
                f"Available scripts: {list(package_data.get('scripts', {}).keys())}"
            )
        print(f"Running npm script: {script_name}")
        build_env = self._node_env()
        if env:
            build_env.update(env)
        subprocess.check_call(
            [str(npm_bin), "run", script_name],
            cwd=self.project_dir,
            env=build_env,
        )

    def generate_custom_observations(self) -> None:
        """Generate custom_observations.ts from user-registered ObsFunc sentinels.

        Iterates ``_custom_registry`` and collects entries that have a ``ts_src``
        path.  Each source file is read and its content is inlined into
        ``custom_observations.ts``, which is then re-exported via the
        ``CustomObservations`` map so the browser-side ``PolicyRunner`` can
        resolve the class by name.

        Entries without ``ts_src`` (unsupported sentinels) are ignored — they
        need no JavaScript representation.
        """
        from mjswan.envs.mdp.observations import _custom_registry

        output_path = (
            self.project_dir / "src" / "core" / "observation" / "custom_observations.ts"
        )

        custom_entries = {
            name: sentinel
            for name, sentinel in _custom_registry.items()
            if sentinel.ts_src is not None and sentinel.ts_name
        }

        if not custom_entries:
            output_path.write_text(
                "// Custom observation classes registered via"
                " mjswan.envs.mdp.observations.register_obs_func().\n"
                "// This file is auto-generated at build time — do not edit manually.\n"
                "\n"
                "export const CustomObservations:"
                " Record<string, new (...args: never[]) => unknown> = {};\n"
            )
            return

        lines = [
            "// Custom observation classes registered via"
            " mjswan.envs.mdp.observations.register_obs_func().",
            "// This file is auto-generated at build time — do not edit manually.",
            "",
        ]

        # Collect imports (deduplicated) and class bodies separately
        seen_imports: list[str] = []
        class_bodies: list[str] = []
        class_names: list[str] = []
        seen_sources: set[Path] = set()
        for sentinel in custom_entries.values():
            src_path = Path(sentinel.ts_src).expanduser().resolve()  # type: ignore[arg-type]
            if not src_path.exists():
                raise FileNotFoundError(
                    f"Custom observation ts_src not found: {src_path}"
                )
            class_names.append(sentinel.ts_name)
            if src_path in seen_sources:
                continue
            seen_sources.add(src_path)
            src_lines = src_path.read_text().splitlines()
            body_lines = []
            for src_line in src_lines:
                if src_line.startswith("import "):
                    if src_line not in seen_imports:
                        seen_imports.append(src_line)
                else:
                    body_lines.append(src_line)
            class_bodies.append("\n".join(body_lines).strip())

        # Emit deduplicated imports, then class bodies, then the registry map
        lines.extend(seen_imports)
        lines.append("")
        for body in class_bodies:
            lines.append(body)
            lines.append("")
        lines.append("export const CustomObservations = {")
        for cls in class_names:
            lines.append(f"  {cls},")
        lines.append("};")
        lines.append("")

        output_path.write_text("\n".join(lines))

    def generate_custom_commands(self) -> None:
        """Generate custom_commands.ts from user-registered command terms."""
        from mjswan.command import _custom_registry

        output_path = (
            self.project_dir / "src" / "core" / "command" / "custom_commands.ts"
        )

        custom_entries = {
            name: spec
            for name, spec in _custom_registry.items()
            if spec.ts_src is not None and spec.ts_name
        }

        if not custom_entries:
            output_path.write_text(
                "// Custom command terms registered via"
                " mjswan.register_command_term().\n"
                "// This file is auto-generated at build time — do not edit manually.\n"
                "\n"
                "import type { CommandTermConstructor } from './types';\n"
                "\n"
                "export const CustomCommands:"
                " Record<string, CommandTermConstructor> = {};\n"
            )
            return

        lines = [
            "// Custom command terms registered via mjswan.register_command_term().",
            "// This file is auto-generated at build time — do not edit manually.",
            "",
            "import type { CommandTermConstructor } from './types';",
            "",
        ]

        seen_imports: list[str] = []
        class_bodies: list[str] = []
        class_names: list[str] = []
        seen_sources: set[Path] = set()
        for spec in custom_entries.values():
            src_path = Path(spec.ts_src).expanduser().resolve()  # type: ignore[arg-type]
            if not src_path.exists():
                raise FileNotFoundError(f"Custom command ts_src not found: {src_path}")
            class_names.append(spec.ts_name)
            if src_path in seen_sources:
                continue
            seen_sources.add(src_path)
            src_lines = src_path.read_text().splitlines()
            body_lines = []
            for src_line in src_lines:
                if src_line.startswith("import "):
                    if src_line not in seen_imports:
                        seen_imports.append(src_line)
                else:
                    body_lines.append(src_line)
            class_bodies.append("\n".join(body_lines).strip())

        lines.extend(seen_imports)
        lines.append("")
        for body in class_bodies:
            lines.append(body)
            lines.append("")
        lines.append(
            "export const CustomCommands: Record<string, CommandTermConstructor> = {"
        )
        for cls in class_names:
            lines.append(f"  {cls},")
        lines.append("};")
        lines.append("")

        output_path.write_text("\n".join(lines))

    def generate_custom_terminations(self) -> None:
        """Generate custom_terminations.ts from user-registered terminations."""
        from mjswan.envs.mdp.terminations import _custom_registry

        output_path = (
            self.project_dir / "src" / "core" / "termination" / "custom_terminations.ts"
        )

        custom_entries = {
            name: sentinel
            for name, sentinel in _custom_registry.items()
            if sentinel.ts_src is not None and sentinel.ts_name
        }

        if not custom_entries:
            output_path.write_text(
                "// Custom termination classes registered via"
                " mjswan.envs.mdp.terminations.register_termination_func().\n"
                "// This file is auto-generated at build time — do not edit manually.\n"
                "\n"
                "type TerminationConstructor ="
                " new (config: import('./TerminationBase').TerminationConfig) => import('./TerminationBase').TerminationBase;\n"
                "\n"
                "export const CustomTerminations:"
                " Record<string, TerminationConstructor> = {};\n"
            )
            return

        lines = [
            "// Custom termination classes registered via"
            " mjswan.envs.mdp.terminations.register_termination_func().",
            "// This file is auto-generated at build time — do not edit manually.",
            "",
            "type TerminationConstructor = new (config: import('./TerminationBase').TerminationConfig) => import('./TerminationBase').TerminationBase;",
            "",
        ]

        seen_imports: set[str] = set()
        deduped_imports: list[str] = []
        class_bodies: list[str] = []
        class_names: list[str] = []
        inlined_utils_lines: list[str] = []
        seen_utils_paths: set[str] = set()
        for sentinel in custom_entries.values():
            src_path = Path(sentinel.ts_src).expanduser().resolve()  # type: ignore[arg-type]
            if not src_path.exists():
                raise FileNotFoundError(
                    f"Custom termination ts_src not found: {src_path}"
                )
            src_lines = src_path.read_text().splitlines()
            body_lines = []
            for src_line in src_lines:
                if src_line.startswith("import "):
                    # Detect local sibling imports (e.g. from './utils') and inline
                    # their content instead of emitting an unresolvable import statement.
                    local_match = re.match(
                        r"^import\s+(?:type\s+)?\{[^}]+\}\s+from\s+['\"](\./[^'\"]+)['\"];?$",
                        src_line,
                    )
                    if local_match:
                        utils_rel = local_match.group(1)
                        utils_path = (src_path.parent / (utils_rel + ".ts")).resolve()
                        if utils_path.exists():
                            if str(utils_path) not in seen_utils_paths:
                                seen_utils_paths.add(str(utils_path))
                                for u_line in utils_path.read_text().splitlines():
                                    if not u_line.startswith("import "):
                                        # Strip leading 'export' so symbols become file-local.
                                        stripped = (
                                            u_line[len("export ") :]
                                            if u_line.startswith("export ")
                                            and not u_line.startswith("export default")
                                            else u_line
                                        )
                                        inlined_utils_lines.append(stripped)
                            # Skip the import line — content is inlined above.
                        elif src_line not in seen_imports:
                            seen_imports.add(src_line)
                            deduped_imports.append(src_line)
                    else:
                        if src_line not in seen_imports:
                            seen_imports.add(src_line)
                            deduped_imports.append(src_line)
                else:
                    body_lines.append(src_line)
            class_bodies.append("\n".join(body_lines).strip())
            class_names.append(sentinel.ts_name)

        lines.extend(deduped_imports)
        lines.append("")
        if inlined_utils_lines:
            lines.extend(inlined_utils_lines)
            lines.append("")
        for body in class_bodies:
            lines.append(body)
            lines.append("")
        lines.append(
            "export const CustomTerminations: Record<string, TerminationConstructor> = {"
        )
        for cls in class_names:
            lines.append(f"  {cls},")
        lines.append("};")
        lines.append("")

        output_path.write_text("\n".join(lines))

    def generate_custom_events(self) -> None:
        """Generate custom_events.ts from user-registered event functions."""
        from mjswan.envs.mdp.events import _custom_registry

        output_path = self.project_dir / "src" / "core" / "event" / "custom_events.ts"

        custom_entries = {
            name: sentinel
            for name, sentinel in _custom_registry.items()
            if sentinel.ts_src is not None and sentinel.ts_name
        }

        if not custom_entries:
            output_path.write_text(
                "// Custom event classes registered via"
                " mjswan.envs.mdp.events.register_event_func().\n"
                "// This file is auto-generated at build time — do not edit manually.\n"
                "\n"
                "import type { EventConstructor } from './EventBase';\n"
                "\n"
                "export const CustomEvents: Record<string, EventConstructor> = {};\n"
            )
            return

        lines = [
            "// Custom event classes registered via"
            " mjswan.envs.mdp.events.register_event_func().",
            "// This file is auto-generated at build time — do not edit manually.",
            "",
            "import type { EventConstructor } from './EventBase';",
            "",
        ]

        seen_imports: set[str] = set()
        deduped_imports: list[str] = []
        class_bodies: list[str] = []
        class_names: list[str] = []
        for sentinel in custom_entries.values():
            src_path = Path(sentinel.ts_src).expanduser().resolve()  # type: ignore[arg-type]
            if not src_path.exists():
                raise FileNotFoundError(f"Custom event ts_src not found: {src_path}")
            src_lines = src_path.read_text().splitlines()
            body_lines = []
            for src_line in src_lines:
                if src_line.startswith("import "):
                    if src_line not in seen_imports:
                        seen_imports.add(src_line)
                        deduped_imports.append(src_line)
                else:
                    body_lines.append(src_line)
            class_bodies.append("\n".join(body_lines).strip())
            class_names.append(sentinel.ts_name)

        lines.extend(deduped_imports)
        lines.append("")
        for body in class_bodies:
            lines.append(body)
            lines.append("")
        lines.append("export const CustomEvents: Record<string, EventConstructor> = {")
        for cls in class_names:
            lines.append(f"  {cls},")
        lines.append("};")
        lines.append("")

        output_path.write_text("\n".join(lines))

    def generate_viewer_config_defaults(self) -> None:
        """Generate viewer_config_defaults.ts from Python ViewerConfig defaults."""
        from mjswan.viewer_config import ViewerConfig

        d = ViewerConfig()
        fovy_default = 45  # Python fovy=None means "use 45 degrees"

        lines = [
            "// Auto-generated from Python ViewerConfig defaults. Do not edit manually.",
            "// Regenerated by mjswan._build_client.ClientBuilder.generate_viewer_config_defaults()",
            "",
            "export const VIEWER_CONFIG_DEFAULTS = {",
            f"  lookat: [{d.lookat[0]}, {d.lookat[1]}, {d.lookat[2]}] as [number, number, number],",
            f"  distance: {d.distance},",
            f"  elevation: {d.elevation},",
            f"  azimuth: {d.azimuth},",
            f"  fovy: {fovy_default},",
            f"  originType: '{d.origin_type.name}' as const,",
            f"  enableReflections: {str(d.enable_reflections).lower()},",
            f"  enableShadows: {str(d.enable_shadows).lower()},",
            f"  height: {d.height},",
            f"  width: {d.width},",
            "} as const;",
            "",
        ]
        output_path = (
            self.project_dir / "src" / "core" / "engine" / "viewer_config_defaults.ts"
        )
        output_path.write_text("\n".join(lines))

    def build(
        self,
        clean: bool = False,
        base_path: str = "/",
        gtm_id: str | None = None,
        mt: bool = False,
        debug: bool = False,
    ) -> None:
        try:
            self.create_env(clean=clean)
            self.sync_version_from_python()
            self.generate_custom_observations()
            self.generate_custom_commands()
            self.generate_custom_events()
            self.generate_custom_terminations()
            self.generate_viewer_config_defaults()
            self.install_dependencies(clean=clean)
            env: dict[str, str] = {"MJSWAN_BASE_PATH": base_path}
            if gtm_id:
                env["MJSWAN_GTM_ID"] = gtm_id
            if mt:
                env["MJSWAN_MT"] = "1"
            if debug:
                env["MJSWAN_DEBUG"] = "1"
            self.run_build_script("build", env=env)
            print("✓ Build completed successfully")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Build failed with exit code {e.returncode}") from e
        except Exception as e:
            raise RuntimeError(f"Build failed: {e}") from e

    def cleanup(self) -> None:
        if self.nodeenv_dir.exists():
            print(f"Cleaning up nodeenv: {self.nodeenv_dir}")
            shutil.rmtree(self.nodeenv_dir)


def ensure_node_env(
    project_dir: Path, node_version: str = "25.5.0", clean: bool = False
) -> Path:
    builder = ClientBuilder(project_dir)
    builder.create_env(clean=clean)
    return builder.nodeenv_dir


def build_client(
    project_dir: Path,
    clean: bool = False,
    script: str = "build",
    base_path: str = "/",
    mt: bool = False,
) -> None:
    builder = ClientBuilder(project_dir)
    builder.build(clean=clean, base_path=base_path, mt=mt)
