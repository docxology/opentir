# AGENTS.md — `opentir/tests`

> Repository-wide policy for template projects is maintained in the template repository, not restated here.

## What this is

pytest suite for the `src/` modules — one test file per module
(`test_github_client.py`, `test_repo_manager.py`, `test_code_analyzer.py`,
`test_docs_generator.py`, `test_multi_repo_analyzer.py`, `test_main.py`,
`test_utils.py`). Run from the project root: `pytest` (config in `pytest.ini`);
inside the venv created by `./run.sh`: `source venv/bin/activate && pytest`.

## Layout (verified by direct listing, 2026-08-29)

```
__init__.py, test_code_analyzer.py, test_docs_generator.py, test_github_client.py, test_main.py, test_multi_repo_analyzer.py, test_repo_manager.py, test_utils.py
```
