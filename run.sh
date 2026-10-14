#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

advanced=false
force=false
skip_deps=false
while (($#)); do
    case "$1" in
        --advanced-stats|--stats) advanced=true ;;
        --force-reanalysis|--force) force=true ;;
        --skip-deps) skip_deps=true ;;
        --help|-h)
            printf '%s\n' 'Usage: ./run.sh [--advanced-stats] [--force-reanalysis] [--skip-deps]'
            exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
    esac
    shift
done

export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.cache/uv}"
export TMPDIR="${OPENTIR_TMP_DIR:-$PWD/.cache/tmp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.cache/matplotlib}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
mkdir -p "$UV_CACHE_DIR" "$TMPDIR" "$MPLCONFIGDIR"
if ! "$skip_deps"; then
    uv sync --frozen
fi
arguments=(multi-analysis)
if "$force"; then
    arguments+=(--force)
fi
uv run --frozen --no-sync opentir "${arguments[@]}"
if "$advanced"; then
    uv run --frozen --no-sync python examples/advanced_statistical_analysis.py
fi
printf '%s\n' 'Analysis completed. Results: multi_repo_analysis/'
