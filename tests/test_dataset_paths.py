"""Relocating a checkpoint's recorded dataset paths onto the machine that deploys it.

Training moved to a remote workstation whose datasets live under ``/data/flore/datasets``,
so ``hl_data.zarr_paths`` in an S3-shipped checkpoint names a path that exists nowhere
locally. These pin the resolution rules in :mod:`flore.utils.dataset_paths`: prefer what
exists, fall back to a basename match in the local dataset dir, and leave a genuinely
missing path ALONE so the caller's own error names what the checkpoint recorded.
"""

from __future__ import annotations

from flore.utils.dataset_paths import (
    DEFAULT_SEARCH_DIRS,
    resolve_dataset_path,
    resolve_dataset_paths,
)


def _fake_repo(tmp_path):
    """A repo root with the real dataset directory layout and one dataset in it."""
    datasets = tmp_path / DEFAULT_SEARCH_DIRS[0]
    datasets.mkdir(parents=True)
    (datasets / "session_9999.01.01_fixture_v3.zarr.zip").write_bytes(b"zarr")
    return datasets


def test_relocates_a_training_machine_absolute_path(tmp_path):
    datasets = _fake_repo(tmp_path)
    logged: list[str] = []
    got = resolve_dataset_path(
        "/data/flore/datasets/session_9999.01.01_fixture_v3.zarr.zip",
        repo_root=str(tmp_path),
        log=logged.append,
    )
    assert got == str(datasets / "session_9999.01.01_fixture_v3.zarr.zip")
    # Substituting silently would make "which dataset am I serving?" unanswerable from a log.
    assert len(logged) == 1 and "/data/flore/datasets/" in logged[0]


def test_existing_path_is_returned_untouched_and_unlogged(tmp_path):
    datasets = _fake_repo(tmp_path)
    real = str(datasets / "session_9999.01.01_fixture_v3.zarr.zip")
    logged: list[str] = []
    assert resolve_dataset_path(real, repo_root=str(tmp_path), log=logged.append) == real
    assert logged == []


def test_missing_everywhere_is_returned_unchanged(tmp_path):
    """No invention: the caller's FileNotFoundError must name the recorded path."""
    _fake_repo(tmp_path)
    recorded = "/data/flore/datasets/session_9999.01.02_absent_v3.zarr.zip"
    logged: list[str] = []
    assert resolve_dataset_path(recorded, repo_root=str(tmp_path), log=logged.append) == recorded
    assert logged == []


def test_repo_relative_path_resolves_against_the_repo_root(tmp_path):
    """Pre-move checkpoints recorded repo-relative paths; those must keep working from any cwd."""
    datasets = _fake_repo(tmp_path)
    recorded = f"{DEFAULT_SEARCH_DIRS[0]}/session_9999.01.01_fixture_v3.zarr.zip"
    assert resolve_dataset_path(recorded, repo_root=str(tmp_path)) == str(
        datasets / "session_9999.01.01_fixture_v3.zarr.zip"
    )


def test_multi_shard_order_is_preserved(tmp_path):
    """File order decides episode order and therefore the validation TAIL, so a resolver that
    reordered (or deduplicated) shards would silently change the split."""
    datasets = _fake_repo(tmp_path)
    for name in ("shard_b.zarr.zip", "shard_a.zarr.zip"):
        (datasets / name).write_bytes(b"zarr")
    recorded = (
        "/data/flore/datasets/shard_b.zarr.zip",
        "/data/flore/datasets/session_9999.01.01_fixture_v3.zarr.zip",
        "/data/flore/datasets/shard_a.zarr.zip",
    )
    got = resolve_dataset_paths(recorded, repo_root=str(tmp_path))
    assert [p.split("/")[-1] for p in got] == [p.split("/")[-1] for p in recorded]


def test_a_same_named_file_outside_the_search_dir_is_not_picked_up(tmp_path):
    """Only the ONE known dataset directory is searched — not a walk of the disk."""
    _fake_repo(tmp_path)
    stray = tmp_path / "elsewhere"
    stray.mkdir()
    (stray / "session_9999.01.03_stray_v3.zarr.zip").write_bytes(b"zarr")
    recorded = "/data/flore/datasets/session_9999.01.03_stray_v3.zarr.zip"
    assert resolve_dataset_path(recorded, repo_root=str(tmp_path)) == recorded
