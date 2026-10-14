# Changelog

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
