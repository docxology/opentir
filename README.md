# Opentir

Opentir analyzes locally cloned GitHub repositories and generates source metrics,
cross-repository reports, visualizations, and a MkDocs documentation site.
It defaults to Palantir's public organization. Repository counts and metrics depend
on the checked-out sources; this README does not represent a current ecosystem census.

## Install and verify

Python 3.10 or newer is required. The development interpreter is Python 3.12.
Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```sh
uv sync --frozen
uv run opentir --version
uv run pytest
uv run ruff check src tests scripts main.py setup.py comprehensive_analysis.py examples
```

The runtime dependencies are declared in requirements.txt; development tools are
in pyproject.toml. uv.lock records the resolved environments. Wheels and source
archives are attached to [GitHub releases](https://github.com/docxology/opentir/releases).

## Analyze a local repository without GitHub

```sh
uv run opentir analyze --repo-path ./my-repository --output-dir ./analysis_results
uv run opentir generate-docs --offline --analysis-dir ./analysis_results --output-dir ./generated_docs
uv run mkdocs build --strict -f generated_docs.mkdocs.yml
uv run mkdocs serve -f generated_docs.mkdocs.yml
```

Analysis writes a repository-specific JSON file, all_analyses.json, and
functionality_matrix.json. Nested files and code elements remain structured JSON.
Documentation generation validates that schema and rejects old files containing
stringified code elements; regenerate those files with analyze.

The MkDocs configuration is a sibling of the documentation directory, named
`<output-directory>.mkdocs.yml`. The rendered site is
`<output-directory>_site/`. Installed package templates are read-only.

## Discover and clone repositories

Set GITHUB_TOKEN in your environment for authenticated access. Tokens are not
persisted by the CLI; interactive entry is hidden. GitHub discovery requires network
access and is subject to the account's API limits.

```sh
uv run opentir fetch-repos --output ./repository_list.json
uv run opentir clone-all
uv run opentir analyze
uv run opentir generate-docs --offline
# Fetch, clone, analyze, and generate documentation:
uv run opentir build-complete
```

Repositories live in repos/all_repos/, with language and popularity links under
repos/. clone-all --force updates existing checkouts using fast-forward pulls and
refuses dirty checkouts. It does not discard local changes. Shallow clones limit
available history. A substantial existing collection may bypass discovery unless
--force is passed.

## Cross-repository reports

```sh
./run.sh
./run.sh --force-reanalysis
./run.sh --advanced-stats
uv run opentir multi-analysis --force --output-dir ./reports
```

run.sh uses the locked uv environment. --skip-deps uses the existing environment
without synchronizing it. Errors propagate as nonzero exits. --force-reanalysis
bypasses caches without deleting previous output files.

The multi-analysis command creates JSON metrics, per-repository summaries,
Markdown reports, CSV exports, PNG figures, and pattern analyses. Cached aggregate
metrics can be reused; use --force after changing sources or analysis settings.
The advanced statistics example and comprehensive_analysis.py are exploratory
tools for ANOVA, PCA, and clustering, not evidence of causal relationships.

## Configuration

Copy .env.example to .env or pass a JSON configuration:

```sh
uv run opentir --config-file ./settings.json status
```

Recognized keys include base_dir, github_org, github_api_url, github_token,
repos_dir, docs_dir, cache_dir, logs_dir, log_level, max_concurrent_requests,
include_tests, and include_examples. base_dir is converted to a Path.
Invalid fields and nonpositive concurrency fail with a CLI error. Environment
names are uppercase versions of these keys (BASE_DIR, LOG_LEVEL, and so on);
GITHUB_TOKEN is supported. Unrelated dotenv entries are ignored.
github_api_url applies to both synchronous and asynchronous GitHub requests.

```json
{
  "base_dir": "./workspace",
  "github_org": "palantir",
  "include_tests": false,
  "include_examples": false
}
```

For work on external storage, bind UV_PYTHON to an interpreter on that volume,
and keep UV_CACHE_DIR, TMPDIR, MPLCONFIGDIR, and the virtual environment there.
Use UV_LINK_MODE=copy across volumes. Verify both sys.prefix and sys.base_prefix.

## CLI commands

| Command | Purpose |
| --- | --- |
| fetch-repos | Discover repositories and metadata |
| clone-all | Clone missing repositories; --force updates clean checkouts |
| analyze | Analyze a local path or repos/all_repos |
| generate-docs | Generate documentation from saved analysis; --offline skips GitHub |
| build-complete | Discovery, cloning, analysis, documentation |
| status | Report workspace state |
| multi-analysis | Generate aggregate reports, figures, and patterns |
| cleanup | Delete cloned repositories after confirmation; --keep-popular protects linked popular repositories |

Use `uv run opentir <command> --help` for options. Cleanup is destructive;
review the target workspace first.

## Additional entry points

| File | Purpose |
| --- | --- |
| main.py | Analyze already-cloned repositories and report workspace status |
| demo.py | Capability walkthrough |
| comprehensive_analysis.py | Exploratory statistics over saved individual summaries |
| examples/basic_usage.py | Python API examples |
| examples/advanced_statistical_analysis.py | Advanced source statistics |
| scripts/update_all_repos.py | Repository update orchestration |
| scripts/comprehensive_workflow.py | Complete workflow orchestration |
| scripts/run_with_rate_limit_handling.py | Rate-limit-aware orchestration |
| scripts/run_full_analysis.sh | Run update and comprehensive workflows through uv |

Generated outputs, cloned repositories, caches, and documentation sites are
workspace data. Existing summary documents describe historical runs and must
not be treated as fresh measurements.

## Methodology and limitations

- Python extraction uses the AST, including sync/async methods, positional-only
  and keyword-only parameters, and return annotations. JavaScript, TypeScript,
  Java, and Go use heuristic regular expressions. Rust and other extensions
  may be recognized but do not currently have extraction parsers.
- Lines of code count nonblank source lines, including comments. Functionality
  categories come from names and docstrings. Complexity is an approximation,
  not a comparable quality score across languages.
- Virtual environments, dependency/build directories, oversized files, and
  external file symlinks are excluded. include_tests and include_examples
  control directory inclusion.
- Dependencies are inferred from imports; they are not resolved dependency
  versions. Documentation size groups are not measurements of maintenance activity.
- Statistical patterns describe the sampled repositories. Selection, missing
  parsers, shallow clones, and correlated observations limit inference.

## Development and releases

CI runs the offline test suite with a 90% coverage gate on Python 3.10 and 3.12, checks Python syntax
errors with Ruff, builds distributions, and checks package metadata with Twine.
Regression tests exercise real local Git, a local HTTP server, CLI subprocesses,
and a strict MkDocs build. Existing legacy unit tests also use mocks.
Python 3.10 on macOS uses SciPy below 1.15 because the resolved 1.15.3 wheel
failed to load on the release-validation machine.

```sh
uv run pytest --cov=src --cov-report=term-missing
uv build
uv run twine check dist/*
```

See [CHANGELOG.md](CHANGELOG.md) for release changes.
Licensed under [Apache 2.0](LICENSE).
