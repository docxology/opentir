# AGENTS.md — `opentir`

> Local-only path under `projects/ongoing/DataTools/` — matched by the
> root `.gitignore` rule `projects/*`; never commit. Repo-wide policy:
> see `/Volumes/external_drive/Git/template/projects/ongoing/AGENTS.md`.

## What this is

`opentir` is a toolkit for analyzing Palantir's open-source ecosystem on GitHub:
fetch the ~250 repositories, clone them locally, analyze code (multi-language
function/complexity extraction), run multi-repo statistics (ANOVA/PCA/clustering),
and generate documentation sites and reports. Packaging via `setup.py` (console
entry point `opentir = src.cli:main`); deps in `requirements.txt`.

## How to run (derived from run.sh / cli.py / main.py source, 2026-08-29)

- `./run.sh` — authoritative orchestrator: creates `venv/`, installs deps, runs
  `main.py` (workspace status + multi-repo analysis over already-cloned repos),
  prints generated report/visualization paths. Options: `--advanced-stats` (also
  runs `examples/advanced_statistical_analysis.py` → `advanced_analysis_results/`),
  `--force-reanalysis` (deletes multi_repo_analysis cross-repo JSON caches),
  `--skip-deps`.
- Click CLI (`src/cli.py`, installed as `opentir` or `python -m src.cli`):
  `fetch-repos [--token] [--output]`, `clone-all [--force] [--token]` (async),
  `analyze [--repo-path] [--output-dir]`, `generate-docs [--analysis-dir]
  [--output-dir] [--token]`, `build-complete [--token]` (fetch → clone → analyze
  → docs), `status`, `multi-analysis [--force] [--output-dir]`, `cleanup
  [--keep-popular]` (with confirmation prompt).
- `python main.py` — standalone orchestrator entry (status check + multi-repo
  analysis over existing repos/). `python demo.py` — capability walkthrough, no
  network. `python comprehensive_analysis.py` — statistical analysis reading
  `multi_repo_analysis/individual_analyses/` and writing
  `comprehensive_analysis_results/` (visualizations/ + data_exports/).

## Invariants / gotchas

- Generated data, not docs: `multi_repo_analysis/`, `comprehensive_analysis_results/`,
  `advanced_analysis_results/`, cloned `repos/` trees.
- GitHub API rate limits: use a token (`--token` / `GITHUB_TOKEN`; see `.env.example`)
  and prefer `scripts/run_with_rate_limit_handling.py` for large runs.
- Root `README.md` links to ~127 `docs/palantir/**` paths that do not exist on
  disk (verified 2026-08-29) — generated docs were not shipped; don't treat
  those links as live.
- Tests: `pytest` via `pytest.ini` in the project root (`tests/`).

## Layout (verified by direct listing, 2026-08-29)

```
comprehensive_analysis_results/, docs/, examples/, multi_repo_analysis/, scripts/, src/, tests/, .cursorignore, .cursorrules, .env.example, .gitattributes, .gitignore, DOCUMENTATION_UPDATES.md, IMPROVEMENTS_SUMMARY.md, README.md, STATISTICAL_ANALYSIS_SUMMARY.md, comprehensive_analysis.py, demo.py, main.py, pytest.ini, requirements.txt, run.sh, setup.py
```
