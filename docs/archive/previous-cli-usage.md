# Opentir CLI Usage Guide

All commands below are the ones defined in `src/cli.py`; run
`uv run opentir <command> --help` for the authoritative option list.

## Installation

```bash
# Locked development environment (recommended; installs uv first)
uv sync --frozen
uv run opentir --version

# Or install the package itself
pip install -r requirements.txt   # or: pip install -e .
```

## GitHub Token Setup

Discovery, cloning, and `generate-docs` without `--offline` need GitHub API
access (60 requests/hour unauthenticated, 5,000 with a token).

```bash
# 1. Interactive prompt (used when no token is configured and stdin is a TTY)
uv run opentir fetch-repos

# 2. Environment variable
export GITHUB_TOKEN=your_github_token_here

# 3. Explicit option
uv run opentir fetch-repos --token your_github_token_here
```

Create tokens at https://github.com/settings/tokens with the `public_repo`
scope. Tokens are never persisted by the CLI and interactive entry is hidden.

## Commands

| Command | Purpose |
| --- | --- |
| `fetch-repos [--token T] [--output FILE]` | Discover organization repositories and metadata |
| `clone-all [--force] [--token T]` | Clone missing repositories; `--force` fast-forward updates clean checkouts |
| `analyze [--repo-path DIR] [--output-dir DIR]` | Analyze one repository or `repos/all_repos` |
| `generate-docs [--analysis-dir DIR] [--output-dir DIR] [--token T] [--offline]` | Build documentation from saved analysis |
| `build-complete [--token T]` | Fetch, clone, analyze, and generate documentation |
| `status` | Report workspace state |
| `multi-analysis [--force] [--output-dir DIR]` | Aggregate statistics, figures, and reports |
| `cleanup [--keep-popular]` | Delete cloned repositories after confirmation |

Global options: `--config-file/-c FILE`, `--verbose/-v`, `--version`.

### Complete workflow

```bash
uv run opentir build-complete --token YOUR_GITHUB_TOKEN
```

### Step-by-step workflow

```bash
uv run opentir fetch-repos --output ./repository_list.json
uv run opentir clone-all
uv run opentir analyze
uv run opentir generate-docs --offline
```

### Local repository without GitHub

```bash
uv run opentir analyze --repo-path ./my-repository --output-dir ./analysis_results
uv run opentir generate-docs --offline --analysis-dir ./analysis_results --output-dir ./generated_docs
uv run mkdocs build --strict -f generated_docs.mkdocs.yml
```

### Cross-repository analysis

```bash
./run.sh                       # uv sync + multi-analysis
./run.sh --force-reanalysis    # ignore cached aggregate metrics
./run.sh --advanced-stats      # also run examples/advanced_statistical_analysis.py
uv run opentir multi-analysis --force --output-dir ./reports
```

## Configuration

Settings live in `src/config.py`. Pass a JSON file, use environment variables
(uppercase key names: `BASE_DIR`, `LOG_LEVEL`, `GITHUB_TOKEN`, ...), or a `.env`
file; `GITHUB_TOKEN` is also accepted directly.

```bash
uv run opentir --config-file ./settings.json status
```

```json
{
  "base_dir": "./workspace",
  "github_org": "palantir",
  "include_tests": false,
  "include_examples": false,
  "max_concurrent_requests": 10
}
```

There are no `opentir config`/`clear-cache`/`list-repos` subcommands; edit the
JSON file (or the environment) and re-run.

## Output structure

```
repos/
├── all_repos/           # cloned repositories
├── by_language/         # symlinks grouped by detected language
├── popular/             # symlinks for popular repositories
└── by_category/         # symlinks for archived/category groupings
analysis_results/        # per-repository JSON, all_analyses.json, functionality_matrix.json
docs/                    # generated documentation, docs.mkdocs.yml, docs_site/
multi_repo_analysis/     # cross-repo JSON, reports/, visualizations/, individual_analyses/
logs/                    # <logger>.log, created on first log record
```

`generate-docs --output-dir X` writes the MkDocs configuration next to the
directory as `X.mkdocs.yml` and the rendered site to `X_site/`.

## Troubleshooting

```bash
# Verbose logging
uv run opentir --verbose analyze

# Inspect logs (file is created lazily)
tail -f logs/opentir_cli.log

# Rate limits: use a token, or throttle large runs
uv run python scripts/run_with_rate_limit_handling.py --token YOUR_TOKEN

# Reclaim space from cloned repositories
uv run opentir cleanup --keep-popular

# Cache/aggregate metrics went stale
./run.sh --force-reanalysis
```

## Tips

1. Always use a GitHub token for discovery and cloning.
2. Cloning the full organization takes substantial disk space; check with `opentir status`.
3. Keep `UV_CACHE_DIR`, `TMPDIR`, and `MPLCONFIGDIR` on the same volume as the workspace on external storage.
4. View generated documentation with `uv run mkdocs serve -f docs.mkdocs.yml`.
5. `clone-all --force` fast-forwards clean checkouts only; commit or stash local changes first.
