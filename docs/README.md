# opentir — Documentation

Opentir analyzes locally cloned GitHub repositories (default organization:
Palantir): it fetches repository metadata, clones and analyzes code, and
generates documentation, statistics, and visualizations. The root `README.md`
is the canonical reference for install, CLI usage, and methodology/limits; this
file only maps the tree.

## Repository map

- `src/` — application code: `cli.py` (Click CLI), `main.py` (orchestrator),
  `github_client.py`, `repo_manager.py`, `code_analyzer.py`,
  `multi_repo_analyzer.py`, `docs_generator.py`, `config.py`, `utils.py`,
  `templates/`
- `scripts/` — workflow orchestrators (`comprehensive_workflow.py`,
  `run_full_analysis.sh`, `run_with_rate_limit_handling.py`,
  `update_all_repos.py`)
- `multi_repo_analysis/`, `comprehensive_analysis_results/` — generated analysis
  output shipped as sample data
- `examples/` — usage examples (`basic_usage.py`, `advanced_statistical_analysis.py`,
  `cli_usage.md`)
- `tests/` — test suite (`pytest.ini` at root)
- `docs/manuscript/` — publication-readiness decision record

## Running

The package installs a console script (per `setup.py`:
`opentir=src.cli:main`). The locked development environment is the supported path:

```bash
uv sync --frozen
uv run opentir --help
uv run opentir fetch-repos --output ./repository_list.json
uv run opentir analyze --repo-path <path> --output-dir ./analysis_results
```

Subcommands in `src/cli.py`: `fetch-repos`, `clone-all`, `analyze`,
`generate-docs`, `build-complete`, `status`, `multi-analysis`, `cleanup`.
Alternatively run the scripted workflow: `./run.sh` (locked `uv` environment
plus `multi-analysis`), or `scripts/run_full_analysis.sh` for the
update-then-analyze chain. `examples/cli_usage.md` documents the options.

Generated documentation is written as `<output-directory>/` with a sibling
`<output-directory>.mkdocs.yml`; the rendered site lands in
`<output-directory>_site/`.
