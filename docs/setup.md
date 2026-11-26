# Setup and configuration

Use Python 3.11 or newer. The checkout's [.python-version](../.python-version)
pins the development interpreter. Install [uv](https://docs.astral.sh/uv/)
and prepare the locked environment:

```sh
uv sync --frozen
uv run --no-sync opentir --version
uv run --no-sync opentir --help
```

[Runtime requirements](../requirements.txt), [development tools](../pyproject.toml),
and [resolved versions](../uv.lock) are maintained separately.

Use `--config-file settings.json` before the command to load workspace
configuration. The [local walkthrough](local-analysis.md) uses a separate
`base_dir`, an explicit source path, and separate destinations for analysis
and documentation. `include_tests` and `include_examples` control which
directories the analyzer visits. Environment settings such as `BASE_DIR`
can also configure the default workspace.

Opentir's generated JSON, MkDocs files, and built site are working outputs.
The pre-existing [analysis snapshot](../multi_repo_analysis/README.md) and
[statistical snapshot](../comprehensive_analysis_results/README.md) describe
earlier runs.
