#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.cache/uv}"
export TMPDIR="${OPENTIR_TMP_DIR:-$PWD/.cache/tmp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.cache/matplotlib}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
mkdir -p "$UV_CACHE_DIR" "$TMPDIR" "$MPLCONFIGDIR"
export GITHUB_TOKEN="${GITHUB_TOKEN:-${1:-}}"
uv sync --frozen
uv run --frozen --no-sync python scripts/update_all_repos.py --force
uv run --frozen --no-sync python scripts/comprehensive_workflow.py --force
printf '%s\n' 'Workflow completed. Results: repos/, multi_repo_analysis/, docs/'


