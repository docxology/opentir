# tests/

Test suite.

Contents: __init__.py, test_code_analyzer.py, test_docs_generator.py,
test_github_client.py, test_github_workflows.py, test_main.py,
test_multi_repo_analyzer.py, test_release_workflows.py, test_repo_manager.py,
test_utils.py

From the repository root, prepare the locked development environment and run
the suite using the root [pytest configuration](../pytest.ini).

```sh
uv sync --frozen
uv run --frozen --no-sync pytest
```

The Python workflow configures a 90% coverage floor.

```sh
uv run --frozen --no-sync pytest --cov=src --cov-report=term-missing
```

[Project README](../README.md)
