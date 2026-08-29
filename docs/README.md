# opentir — Documentation

Opentir is a toolkit for analyzing, organizing, and documenting Palantir's
open-source ecosystem (250+ repositories, per root `README.md`): it fetches
repository metadata, clones and analyzes code, and generates documentation and
statistics. The root `README.md` is the primary reference and contains the
full Palantir ecosystem guide content inline.

## Repository map

- `src/` — application code: `cli.py` (Click CLI), `main.py` (orchestrator),
  `github_client.py`, `repo_manager.py`, `code_analyzer.py`,
  `multi_repo_analyzer.py`, `docs_generator.py`, `config.py`, `utils.py`,
  `templates/`
- `scripts/` — workflow orchestrators (`comprehensive_workflow.py`,
  `run_full_analysis.sh`, `run_with_rate_limit_handling.py`,
  `update_all_repos.py`)
- `multi_repo_analysis/` — multi-repository analysis support
- `examples/` — usage examples
- `comprehensive_analysis_results/` — output of prior analysis runs
- `tests/` — test suite (`pytest.ini` at root)

## Running

The package installs a console script (per `setup.py` entry points:
`opentir=src.cli:main`):

```bash
pip install -r requirements.txt   # or: pip install -e .
opentir --help
opentir --verbose fetch-repos --token <token> --output repos.json
opentir analyze --repo-path <path>
```

Subcommands defined in `src/cli.py`: `fetch-repos`, clone/update, `analyze`,
`generate-docs`, `status`, `multi-analysis`, `cleanup`. Or run the scripted
workflow: `./run.sh` (wraps `scripts/run_full_analysis.sh`).

## Note on `docs/palantir/` links

The root `README.md` links extensively to `docs/palantir/**` pages that are
not present in this repository (they appear to describe generated or
planned documentation). These links are recorded as known-broken; the
ecosystem content itself is included in the README text.
