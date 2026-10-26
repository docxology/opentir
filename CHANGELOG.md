# Changelog

## Unreleased - 2026-10-25

- Remove dead imports: the unused networkx/plotly/orchestrator imports in
  `examples/advanced_statistical_analysis.py` (the example imported `networkx`,
  which was never declared or locked, so `./run.sh --advanced-stats` crashed at
  import time), unused seaborn and regressor imports in `comprehensive_analysis.py`,
  and unused `Template` and `RateLimitExceededException` imports in `src/`.
- Replace deprecated `asyncio.get_event_loop()` calls in `src/repo_manager.py`
  with `asyncio.get_running_loop()`.
- Refresh CI pins: `actions/checkout@v7` (was v4) and `astral-sh/setup-uv`
  v10.3.2 (was v10.3.0).
- Correct stale documentation: `examples/cli_usage.md` documented subcommands
  that do not exist (`config`, `info`, `list-repos`, `clear-cache`, and others)
  and now lists the real commands and flags; `docs/README.md`, `docs/AGENTS.md`,
  and `AGENTS.md` now match current paths and workflows; the `src/`, `tests/`,
  and `examples/` READMEs no longer describe the checkout as local-only scratch.
- Note for the next release, dependency declarations to fix in one pass: commit
  7877379 trimmed nothing on purpose, so `requirements.txt` still declares
  packages no code imports (`tree-sitter` and its five grammars, `ast-tools`,
  `asyncio-throttle`, `requests`, `plotly`, `kaleido`, 16 lock entries), and
  `PyYAML` is imported directly by `src/docs_generator.py` yet declared only
  transitively through `mkdocs`. Both need `uv lock` plus a full test run, which
  does not fit a documentation-only maintenance pass on slow storage.

## 1.1.0 - 2026-10-14

- Add a locked uv project, Python 3.12 development binding, Python 3.10 minimum,
  packaged documentation templates, Apache license file, and CI.
- Repair Jinja syntax and missing templates; generate a buildable MkDocs site
  without modifying the installed package.
- Preserve nested analysis data and add offline documentation generation,
  version reporting, and validated CLI configuration.
- Await callable-produced coroutines in the orchestrator and share its GitHub
  client with repository management.
- Protect dirty clones, use fast-forward updates, correct clone result summaries,
  support repositories without a detected language, and repair organization links.
- Extract Python methods, complete parameter kinds, and return annotations;
  prune environment/build directories and honor test/example inclusion settings.
- Derive documentation dependencies from actual imports instead of placeholder data.
- Replace the activity-chart placeholder with latest local commit dates, and mark
  undefined constant-input correlations explicitly instead of emitting NaN.
- Repair small-collection cache reuse, scope aggregate outputs to the chosen workspace,
  close replaced log handlers, and defer log-file creation until logging occurs.
- Correct retry counts, binary detection, duration/byte formatting, and filename sanitization.
- Propagate shell and advanced-analysis failures; replace stale README statistics
  and broken generated-documentation links with verified usage and limitations.
- Add real CLI, Git, HTTP, schema, and strict MkDocs regression coverage;
  correct stale legacy test fixtures and result schemas.
- Apply configured API URLs to both GitHub transports, preserve active async
  sessions during nested fetches, and test full workflows against a local server.
- Constrain SciPy on Python 3.10/macOS after reproducing a native wheel import failure.

## 1.0.0

Initial toolkit version. No prior GitHub release was published.
