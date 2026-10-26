# opentir — Agent Guide

## Layout

| Path | Purpose |
|---|---|
| `src/cli.py` | Click CLI group with subcommands (`fetch-repos`, `clone-all`, `analyze`, `generate-docs`, `build-complete`, `status`, `multi-analysis`, `cleanup`) |
| `src/main.py`, `src/config.py`, `src/utils.py` | Orchestration, configuration, logging utilities |
| `src/github_client.py`, `src/repo_manager.py` | GitHub API access and repo cloning/organization |
| `src/code_analyzer.py`, `src/multi_repo_analyzer.py` | Code and multi-repo statistics |
| `src/docs_generator.py`, `src/templates/` | Generated documentation output |
| `scripts/` | Thin workflow wrappers (bash + Python) |
| `tests/` | pytest suite (`pytest.ini`) |

## Conventions observed

- Packaging via `setup.py` with a console entry point `opentir=src.cli:main`;
  requirements in `requirements.txt`.
- Rate limiting matters: dedicated script
  `scripts/run_with_rate_limit_handling.py`; GitHub tokens passed via CLI option.
- Analysis outputs land in `comprehensive_analysis_results/` and
  `multi_repo_analysis/` — treat as generated data, not docs.

## How docs here are maintained

Root `README.md` is the canonical reference; it states methodology and limits
explicitly and avoids ecosystem census claims. Keep `docs/README.md` a short
accurate pointer; do not duplicate the README prose.

## Known issues

- None open. The root `README.md` no longer links to `docs/palantir/**` (those
  links were removed in 1.1.0 because generated docs were never shipped);
  verified 2026-10-25 by filesystem check.

## Notes for agents

- Never commit cloned repositories, tokens, or large analysis result trees.
