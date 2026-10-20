"""Resolve a checkpoint's recorded dataset paths on THIS machine.

A checkpoint stores the zarr paths it was trained on (``config.json`` →
``hl_data.zarr_paths``) so every offline tool can rebuild the exact training dataset and
so the planner can derive the URDF↔model frame seam from the data rather than guess it.
Those paths are recorded verbatim, and since training moved to a remote workstation
(datasets under ``/data/flore/datasets``, checkpoints shipped through S3 — see
``docs/RUNBOOK_S1_WORKSTATION.md``) they are routinely absolute paths that exist on the
*training* box and nowhere else. Locally the same file sits in the UMI submodule's
``data/datasets/``. Checkpoints trained before the move carry repo-relative paths, which
already resolved on both machines; nothing here changes those.

So: keep a path that exists, otherwise look for the same **basename** in the known local
dataset directory, and say so when substituting. The substitution is a convenience, not a
trust boundary — every consumer independently verifies the file it actually opened:

* offline inference / eval compare the checkpoint's ``source_content_fingerprint``,
  ``episode_ends_fingerprint`` and ``anchor_indices_fingerprint`` against the dataset they
  loaded (:func:`flore.s1_policy.inference.common.assert_provenance`), so a same-named file
  with different content is refused;
* ``hl_serving.py`` derives the seam from the file's own ``T_tracker_tcp`` attr and hard-
  refuses the pre-2026-08-06 extrinsic.

A path that cannot be found anywhere is returned **unchanged**, so the caller's own
"no such file" names the path the checkpoint recorded — which is the useful thing to see.
"""

from __future__ import annotations

import os
from typing import Callable, Iterable, Sequence

#: Where a relocated dataset is looked for, relative to the repo root. One entry today:
#: the UMI submodule's dataset directory, which is where both the converter writes and
#: ``scripts/fetch_dataset.sh`` downloads.
DEFAULT_SEARCH_DIRS: tuple[str, ...] = ("third_party/UMI_collection/data/datasets",)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def resolve_dataset_path(
    path: str,
    *,
    repo_root: str | None = None,
    search_dirs: Sequence[str] | None = None,
    log: Callable[[str], None] | None = None,
) -> str:
    """One path, resolved on this machine. See the module docstring for the rules."""
    original = str(path)
    if os.path.exists(original):
        return original
    root = os.path.abspath(repo_root or _REPO_ROOT)
    if not os.path.isabs(original):
        repo_relative = os.path.join(root, original)
        if os.path.exists(repo_relative):
            return repo_relative
    name = os.path.basename(original.rstrip("/"))
    if not name:
        return original
    for directory in DEFAULT_SEARCH_DIRS if search_dirs is None else search_dirs:
        candidate = directory if os.path.isabs(directory) else os.path.join(root, directory)
        candidate = os.path.join(candidate, name)
        if os.path.exists(candidate):
            if log is not None:
                log(
                    f"dataset relocated: the checkpoint recorded {original} (another machine); "
                    f"using the local copy at {candidate}. Content is verified separately."
                )
            return candidate
    return original


def resolve_dataset_paths(
    paths: Iterable[str],
    *,
    repo_root: str | None = None,
    search_dirs: Sequence[str] | None = None,
    log: Callable[[str], None] | None = None,
) -> tuple[str, ...]:
    """:func:`resolve_dataset_path` over a sequence, preserving order (file order matters:
    it decides episode order and therefore the validation tail)."""
    return tuple(
        resolve_dataset_path(p, repo_root=repo_root, search_dirs=search_dirs, log=log) for p in paths
    )
