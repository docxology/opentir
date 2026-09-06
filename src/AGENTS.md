# AGENTS.md — `opentir/src`

> Local-only path under `projects/ongoing/DataTools/` — matched by the
> root `.gitignore` rule `projects/*`; never commit. Repo-wide policy:
> see `/Volumes/external_drive/Git/template/projects/ongoing/AGENTS.md`.

## What this is

opentir source code (`templates/` holds the markdown templates used by the
docs generator):

- `cli.py` — Click CLI group (`fetch-repos`, `clone-all`, `analyze`,
  `generate-docs`, `build-complete`, `status`, `multi-analysis`, `cleanup`);
  async commands wrapped via `async_command` decorator.
- `main.py` — `OpentirOrchestrator`: workspace status + stepwise pipeline.
- `github_client.py` — GitHub API access with rate limiting.
- `repo_manager.py` — cloning and by-language/by-category organization into
  `repos/`.
- `code_analyzer.py` — multi-language function/class/complexity extraction.
- `multi_repo_analyzer.py` — cross-repo statistics; writes `multi_repo_analysis/`.
- `docs_generator.py` — renders `src/templates/*.md` into generated docs.
- `config.py` — paths/tokens/logging config (env + `.env.example`).
- `utils.py` — logging/helpers.

## Layout (verified by direct listing, 2026-08-29)

```
templates/, __init__.py, cli.py, code_analyzer.py, config.py, docs_generator.py, github_client.py, main.py, multi_repo_analyzer.py, repo_manager.py, utils.py
```
