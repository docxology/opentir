# AGENTS.md — `opentir/tests`

> Local-only path under `projects/ongoing/DataTools/` — matched by the
> root `.gitignore` rule `projects/*`; never commit. Repo-wide policy:
> see `/Volumes/external_drive/Git/template/projects/ongoing/AGENTS.md`.

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
