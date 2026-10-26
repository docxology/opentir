# AGENTS: `scripts/` — Thin Orchestrator Scripts

Location: `scripts/` of the repo `docxology/opentir` (default branch `main`).
Repo-wide policy: see `../AGENTS.md`.

## Thin-Orchestrator Contract

- Scripts in this directory are orchestration ONLY: argparse, `sys.path`
  bootstrap to the repo root, logging, and delegated calls into `src/`
  entrypoints.
- Business/data/plot/analysis logic lives in `src/` (importable and covered by
  `tests/`); do not add logic here.
- New or removed scripts must be reflected in the inventory tables in this
  file and in `README.md` in the same change.

## Script Inventory

| Script | Pattern | Delegates to |
|--------|---------|--------------|
| `update_all_repos.py` | Thin orchestrator (argparse + bootstrap + async entry) | `src.main.OpentirOrchestrator` |
| `comprehensive_workflow.py` | Thin orchestrator (argparse + bootstrap + async entry) | `src.main.OpentirOrchestrator`, `src.multi_repo_analyzer.MultiRepositoryAnalyzer` |
| `run_with_rate_limit_handling.py` | Thin orchestrator (argparse + bootstrap + async entry) | `src.github_client.GitHubClient`, `src.main.OpentirOrchestrator`, `src.multi_repo_analyzer.MultiRepositoryAnalyzer` |
| `run_full_analysis.sh` | Bash chain of the two Python workflows above | `scripts/update_all_repos.py`, `scripts/comprehensive_workflow.py` |

Run commands: see `README.md`.

## Gotchas

- Every Python script bootstraps with
  `sys.path.insert(0, Path(__file__).parent.parent)`; run them from the repo
  root so workspace paths (`repos/`, `multi_repo_analysis/`, `docs/`) resolve
  via `src.config`.
- `update_all_repos.py` reaches into private `OpentirOrchestrator._*` helpers
  (`_fetch_organization_info`, `_fetch_repository_list`,
  `_analyze_all_repositories`) — fragile across `src/main.py` refactors.
- GitHub API rate limits: pass `--token` or export `GITHUB_TOKEN` (see
  `.env.example`); prefer `run_with_rate_limit_handling.py` for large runs.
- `run_full_analysis.sh` and `./run.sh` both drive the locked uv environment
  (`uv sync --frozen` + `uv run --frozen --no-sync`); the Python scripts can
  also run under any interpreter that has `requirements.txt` installed.
- Not covered by unit tests; smoke-check with `--help` before relying on a
  full network run.
