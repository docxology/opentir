# AGENTS.md — `opentir/tests`

> Repository-wide policy for template projects is maintained in the template repository, not restated here.

## What this is

pytest suite for the `src/` modules and repository workflows. From the
repository root, run `uv sync --frozen` followed by
`uv run --frozen --no-sync pytest`. The root [pytest configuration](../pytest.ini)
selects `tests/`. See the [test guide](README.md) for the CI coverage command.

## Layout

```
__init__.py, test_code_analyzer.py, test_docs_generator.py, test_github_client.py, test_github_workflows.py, test_main.py, test_multi_repo_analyzer.py, test_release_workflows.py, test_repo_manager.py, test_utils.py
```
