# AGENTS.md — `opentir`

Public repository `docxology/opentir` (default branch `main`); the local working
copy lives under `projects/ongoing/DataTools/` in the vault, where the `.gitignore`
rule `projects/*` covers scratch data only. Vault-wide rules: see the vault root
`AGENTS.md`.

## What this is

`opentir` is a toolkit for analyzing Palantir's open-source ecosystem on GitHub:
fetch the organization's public repositories, clone them locally, analyze code
(Python AST extraction; regex heuristics for JS/TS/Java/Go), run multi-repo
statistics (ANOVA/PCA/clustering), and generate documentation sites and reports.
Packaging via `setup.py` (console entry point `opentir = src.cli:main`); runtime
deps in `requirements.txt`, dev tools in `pyproject.toml`, pins in `uv.lock`.

## How to run (derived from run.sh / cli.py / main.py source, 2026-10-25)

- `./run.sh` — authoritative orchestrator: `uv sync --frozen`, then
  `uv run opentir multi-analysis` over already-cloned repos, and prints the
  results directory (`multi_repo_analysis/`). Options: `--advanced-stats` (also
  runs `examples/advanced_statistical_analysis.py` → `advanced_analysis_results/`),
  `--force-reanalysis` (passed through as `multi-analysis --force`, recomputes
  instead of reusing cached aggregates), `--skip-deps` (run without `uv sync`).
- Click CLI (`src/cli.py`, installed as `opentir` or `python -m src.cli`):
  `fetch-repos [--token] [--output]`, `clone-all [--force] [--token]` (async),
  `analyze [--repo-path] [--output-dir]`, `generate-docs [--analysis-dir]
  [--output-dir] [--token] [--offline]`, `build-complete [--token]` (fetch → clone
  → analyze → docs), `status`, `multi-analysis [--force] [--output-dir]`, `cleanup
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
- `requirements.txt` still declares packages the code does not import
  (`tree-sitter` and its grammars, `ast-tools`, `asyncio-throttle`, `requests`,
  `plotly`, `kaleido`); dropping them needs `uv lock` plus a full test run
  (a slow operation on external storage). The README's methodology section is
  the source of truth for which languages have real parsers.
- Tests: `pytest` via `pytest.ini` in the project root (`tests/`); CI runs the
  suite on Python 3.10 and 3.12 with a 90% coverage gate.

## Layout (verified by direct listing, 2026-10-25)

```
.github/workflows/, comprehensive_analysis_results/, docs/, examples/, multi_repo_analysis/, scripts/, src/, tests/, .cursorignore, .cursorrules, .env.example, .gitattributes, .gitignore, .python-version, AGENTS.md, CHANGELOG.md, DOCUMENTATION_UPDATES.md, IMPROVEMENTS_SUMMARY.md, LICENSE, MANIFEST.in, README.md, STATISTICAL_ANALYSIS_SUMMARY.md, comprehensive_analysis.py, demo.py, main.py, pyproject.toml, pytest.ini, requirements.txt, run.sh, setup.py, uv.lock
```
