#!/usr/bin/env bash
#
# bootstrap-holosoma.sh — Clone or update the holosoma dependency.
#
# When holosoma is not checked out as a git submodule, this script replaces
# `git submodule update --init` for the holosoma dependency.
#
# Usage:
#   bash holosoma_extensions/scripts/bootstrap-holosoma.sh                    # clone / update
#   bash holosoma_extensions/scripts/bootstrap-holosoma.sh -b feature-branch  # use specific branch
#   bash holosoma_extensions/scripts/bootstrap-holosoma.sh --status           # check status
#
# Environment variables:
#   HOLOSOMA_REPO   — Override the clone URL (default: git@github.com:amazon-far/holosoma.git)
#   HOLOSOMA_REF    — Checkout a specific ref after cloning (default: main)

set -euo pipefail

# Resolve the directory of this script, following symlinks (works in bash and zsh).
_resolve_script_dir() {
  # shellcheck disable=SC2296
  local source="${BASH_SOURCE[0]:-${(%):-%x}}"
  while [ -h "$source" ]; do
    local dir
    dir="$( cd -P "$( dirname "$source" )" >/dev/null && pwd )"
    source="$(readlink "$source")"
    [[ $source != /* ]] && source="$dir/$source"
  done
  cd -P "$( dirname "$source" )" >/dev/null && pwd
}
SCRIPT_DIR="$(_resolve_script_dir)"
unset -f _resolve_script_dir
ROOT_DIR=$(dirname "$SCRIPT_DIR")

THIRDPARTY_DIR="$ROOT_DIR/thirdparty"
HOLOSOMA_DIR="$THIRDPARTY_DIR/holosoma"

# --- Submodule-aware guard (added for the yam_gripper repro branch) ---------
# On branches where thirdparty/holosoma is a real git submodule (a .gitmodules
# entry exists, or its .git is a gitlink file rather than a directory), the
# checkout is managed by `git submodule update --init --recursive` and pinned
# to a specific Louis-Leee/holosoma commit. The legacy clone/fetch/checkout
# path below would (a) re-point origin at amazon-far via HOLOSOMA_REPO and
# (b) move HEAD off the pinned commit — both undesirable here. So when a
# submodule is detected we ONLY apply the bazel BUILD overlay and exit,
# leaving the submodule checkout untouched. The deploy/main flow has no
# .gitmodules entry, so this guard is a no-op there.
_is_submodule=false
if [[ -f "$ROOT_DIR/.gitmodules" ]] && \
   git config --file "$ROOT_DIR/.gitmodules" --get submodule.thirdparty/holosoma.path >/dev/null 2>&1; then
    _is_submodule=true
elif [[ -f "$HOLOSOMA_DIR/.git" ]]; then
    # A gitlink: `.git` is a file ("gitdir: ...") for a submodule worktree.
    _is_submodule=true
fi

# Read version file first — it may contain "remote ref" or just "ref".
# Precedence: env vars > version file > defaults.
VERSION_FILE="$ROOT_DIR/holosoma.version"
_file_repo=""
_file_ref=""
if [[ -f "$VERSION_FILE" ]]; then
    _version_line="$(tr -d '\n' < "$VERSION_FILE")"
    if [[ "$_version_line" == *" "* ]]; then
        # Two fields: "remote ref"
        _file_repo="${_version_line%% *}"
        _file_ref="${_version_line##* }"
    else
        # Single field: "ref"
        _file_ref="$(echo "$_version_line" | tr -d '[:space:]')"
    fi
    unset _version_line
fi

HOLOSOMA_REPO="${HOLOSOMA_REPO:-${_file_repo:-git@github.com:amazon-far/holosoma.git}}"
if [[ -z "${HOLOSOMA_REF:-}" ]]; then
    HOLOSOMA_REF="${_file_ref:-main}"
fi
unset _file_repo _file_ref

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -b|--branch)
            HOLOSOMA_REF="$2"
            shift 2
            ;;
        --status)
            break
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: bootstrap-holosoma.sh [-b|--branch REF] [--status]" >&2
            exit 1
            ;;
    esac
done

if [[ "${1:-}" == "--status" ]]; then
    if $_is_submodule; then
        echo "[bootstrap] holosoma is a git SUBMODULE at $HOLOSOMA_DIR (managed by 'git submodule update --init --recursive')"
        (cd "$HOLOSOMA_DIR" 2>/dev/null && echo "  remote: $(git remote get-url origin 2>/dev/null || echo 'unknown')" && echo "  branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null)" && echo "  commit: $(git rev-parse --short HEAD 2>/dev/null)") || true
    elif [ -d "$HOLOSOMA_DIR/.git" ]; then
        echo "[bootstrap] holosoma is present at $HOLOSOMA_DIR"
        (cd "$HOLOSOMA_DIR" && echo "  remote: $(git remote get-url origin 2>/dev/null || echo 'unknown')" && echo "  branch: $(git rev-parse --abbrev-ref HEAD)" && echo "  commit: $(git rev-parse --short HEAD)")
    else
        echo "[bootstrap] holosoma is NOT present. Run: bash holosoma_extensions/scripts/bootstrap-holosoma.sh"
    fi
    exit 0
fi

mkdir -p "$THIRDPARTY_DIR"

# Submodule path: do NOT clone/fetch/checkout (that would unpin the submodule
# and re-point origin at amazon-far). Apply only the bazel overlay, then exit.
if $_is_submodule; then
    if [ ! -e "$HOLOSOMA_DIR/.git" ]; then
        echo "[bootstrap] thirdparty/holosoma is a submodule but is not checked out." >&2
        echo "[bootstrap] Run: git submodule update --init --recursive" >&2
        exit 1
    fi
    echo "[bootstrap] holosoma is a git submodule — skipping clone/fetch/checkout (pin-respecting)."
    if [[ -n "${HOLOSOMA_REF:-}" && "$HOLOSOMA_REF" != "main" ]]; then
        echo "[bootstrap] note: -b/HOLOSOMA_REF=$HOLOSOMA_REF ignored for a submodule; the pin is" >&2
        echo "[bootstrap]       managed by 'git submodule update' (use --remote out-of-band to move it)." >&2
    fi
    BAZEL_SUPPORT_DIR="$ROOT_DIR/bazel_support"
    if [[ -d "$BAZEL_SUPPORT_DIR" ]]; then
        while IFS='|' read -r src dst; do
            src_path="$BAZEL_SUPPORT_DIR/$src"
            dst_path="$HOLOSOMA_DIR/$dst"
            [[ -f "$src_path" ]] || { echo "[bootstrap] missing bazel overlay: $src_path" >&2; exit 1; }
            mkdir -p "$(dirname "$dst_path")"
            if [[ -f "$dst_path" ]] && cmp -s "$src_path" "$dst_path"; then continue; fi
            cp "$src_path" "$dst_path" 2>/dev/null && echo "[bootstrap] overlay: $src -> $dst" || \
                echo "[bootstrap] could not overwrite $dst_path (check ownership); skipping" >&2
        done <<'EOF'
holosoma.BUILD|src/holosoma/BUILD.bazel
holosoma_retargeting.BUILD|src/holosoma_retargeting/BUILD.bazel
holosoma_inference.BUILD|src/holosoma_inference/BUILD.bazel
EOF
    fi
    echo "[bootstrap] holosoma ready at $HOLOSOMA_DIR ($(cd "$HOLOSOMA_DIR" && git rev-parse --short HEAD 2>/dev/null))"
    exit 0
fi

if [ -d "$HOLOSOMA_DIR/.git" ]; then
    # Cache short-circuit: if the current checkout already matches
    # HOLOSOMA_REF (either exact SHA match, or tip of a local branch named
    # REF), skip the git fetch/pull. Cuts bootstrap from ~5s to ~50ms when
    # `pi run wbt-teleop` is invoked repeatedly against a pinned
    # holosoma.version. Force a refresh with PI_HOLOSOMA_FORCE_FETCH=1.
    _current_sha="$(git -C "$HOLOSOMA_DIR" rev-parse HEAD 2>/dev/null || true)"
    _current_branch="$(git -C "$HOLOSOMA_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
    _wanted_sha=""
    if [[ "$HOLOSOMA_REF" =~ ^[0-9a-f]{7,40}$ ]]; then
        _wanted_sha="$HOLOSOMA_REF"
    fi
    _skip_fetch=false
    if [[ "${PI_HOLOSOMA_FORCE_FETCH:-0}" != "1" ]]; then
        if [[ -n "$_wanted_sha" && "$_current_sha" == "$_wanted_sha"* ]]; then
            _skip_fetch=true
        elif [[ "$_current_branch" == "$HOLOSOMA_REF" ]]; then
            # Fetch was costly for nothing if we last fetched in the same
            # shell/session; mtime check protects against staler trees.
            _head_age_s=$(( $(date +%s) - $(stat -c %Y "$HOLOSOMA_DIR/.git/FETCH_HEAD" 2>/dev/null || echo 0) ))
            if (( _head_age_s < 600 )); then
                _skip_fetch=true
            fi
        fi
    fi

    if $_skip_fetch; then
        echo "[bootstrap] holosoma at $HOLOSOMA_DIR is already at ref=$HOLOSOMA_REF (cached)"
    else
        echo "[bootstrap] holosoma already cloned at $HOLOSOMA_DIR — updating..."
        # Update remote URL if it changed (e.g. switching between fork and upstream)
        _current_url="$(git -C "$HOLOSOMA_DIR" remote get-url origin 2>/dev/null || true)"
        if [[ -n "$_current_url" && "$_current_url" != "$HOLOSOMA_REPO" ]]; then
            echo "[bootstrap] Updating origin remote: $_current_url -> $HOLOSOMA_REPO"
            git -C "$HOLOSOMA_DIR" remote set-url origin "$HOLOSOMA_REPO"
        fi
        unset _current_url
        (cd "$HOLOSOMA_DIR" && git fetch origin)
        if git -C "$HOLOSOMA_DIR" rev-parse --verify "origin/$HOLOSOMA_REF" >/dev/null 2>&1; then
            # Remote branch — checkout and fast-forward
            (cd "$HOLOSOMA_DIR" && git checkout "$HOLOSOMA_REF" && git pull origin "$HOLOSOMA_REF")
        else
            # Tag or commit hash — just checkout (detached HEAD)
            (cd "$HOLOSOMA_DIR" && git checkout "$HOLOSOMA_REF")
        fi
    fi
    unset _current_sha _current_branch _wanted_sha _skip_fetch _head_age_s
else
    echo "[bootstrap] Cloning holosoma into $HOLOSOMA_DIR..."
    git clone "$HOLOSOMA_REPO" "$HOLOSOMA_DIR"
    (cd "$HOLOSOMA_DIR" && git checkout "$HOLOSOMA_REF")
fi

# Overlay bazel BUILD files into the cloned tree. These live in
# holosoma_extensions/bazel_support/ (with a `.BUILD` suffix so bazel does not
# pick them up in-place) and are copied to BUILD.bazel at the corresponding
# package roots. Keeping them here rather than upstream means holosoma core
# stays bazel-free.
BAZEL_SUPPORT_DIR="$ROOT_DIR/bazel_support"
if [[ -d "$BAZEL_SUPPORT_DIR" ]]; then
    while IFS='|' read -r src dst; do
        src_path="$BAZEL_SUPPORT_DIR/$src"
        dst_path="$HOLOSOMA_DIR/$dst"
        if [[ ! -f "$src_path" ]]; then
            echo "[bootstrap] missing bazel overlay: $src_path" >&2
            exit 1
        fi
        mkdir -p "$(dirname "$dst_path")"
        # Skip the copy when the destination already has identical content.
        # Common when the docker container (running as root) bootstrapped
        # previously and the host user (running pi from outside) lacks
        # write access to the root-owned BUILD.bazel — cp would fail with
        # 'Permission denied' even though the file is already correct.
        if [[ -f "$dst_path" ]] && cmp -s "$src_path" "$dst_path"; then
            continue
        fi
        if ! cp "$src_path" "$dst_path" 2>/dev/null; then
            echo "[bootstrap] could not overwrite $dst_path (check ownership); skipping" >&2
            continue
        fi
        echo "[bootstrap] overlay: $src -> $dst"
    done <<'EOF'
holosoma.BUILD|src/holosoma/BUILD.bazel
holosoma_retargeting.BUILD|src/holosoma_retargeting/BUILD.bazel
holosoma_inference.BUILD|src/holosoma_inference/BUILD.bazel
EOF
fi

echo "[bootstrap] holosoma ready at $HOLOSOMA_DIR ($(cd "$HOLOSOMA_DIR" && git rev-parse --short HEAD))"
