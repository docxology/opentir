# opentir — Agent Guide

## Layout

| Path | Purpose |
|---|---|
| `src/cli.py` | Click CLI group with subcommands (`fetch-repos`, `analyze`, `generate-docs`, `status`, `multi-analysis`, `cleanup`) |
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

Root `README.md` (757 lines) is the canonical reference and embeds the
Palantir ecosystem guide. Keep `docs/README.md` a short accurate pointer;
do not duplicate the ecosystem prose.

## Known issues

- 127 links in the root `README.md` point to `docs/palantir/**` paths that do
  not exist in the repo. Verified 2026-08-29 by filesystem check. Fixing them
  would require either restoring generated docs or inlining — an owner
  decision; not silently rewritten.

## Notes for agents

- Never commit cloned repositories, tokens, or large analysis result trees.
