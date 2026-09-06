# scripts/

Thin orchestrator scripts for `opentir`: argparse, path bootstrap, logging,
and delegated calls into `src/` entrypoints. Business/data/plot/analysis logic
lives in `src/` (importable, tested under `tests/`). Contract details and
gotchas: see `AGENTS.md`.

## Inventory

| Script | Purpose | Delegates to | Run command |
|--------|---------|--------------|-------------|
| `update_all_repos.py` | Fetch org/repo list from GitHub, clone/update all repositories, validate integrity, optional analysis check | `src.main.OpentirOrchestrator` (`_fetch_organization_info`, `_fetch_repository_list`, `repo_manager.clone_all_repositories`, `_analyze_all_repositories`) | `python3 scripts/update_all_repos.py [--token T] [--force] [--skip-validation] [--skip-analysis] [--base-dir D]` |
| `comprehensive_workflow.py` | End-to-end fetch -> clone -> analyze -> docs workflow plus cross-repo statistics, then prints a summary | `src.main.OpentirOrchestrator.execute_complete_workflow`, `src.multi_repo_analyzer.MultiRepositoryAnalyzer` | `python3 scripts/comprehensive_workflow.py [--token T] [--force] [--skip-docs] [--base-dir D]` |
| `run_with_rate_limit_handling.py` | Same workflow with a GitHub rate-limit pre-check and fallback to analyzing already-cloned repositories (preferred for large runs) | `src.github_client.GitHubClient` (rate-limit check), `src.main.OpentirOrchestrator`, `src.multi_repo_analyzer.MultiRepositoryAnalyzer` | `python3 scripts/run_with_rate_limit_handling.py [--token T] [--force]` |
| `run_full_analysis.sh` | Bash chain: update repositories, then run the comprehensive analysis workflow, with a dependency pre-check | `scripts/update_all_repos.py`, `scripts/comprehensive_workflow.py` | `bash scripts/run_full_analysis.sh [$GITHUB_TOKEN]` |

All Python scripts accept `--token/-t` or fall back to the `GITHUB_TOKEN`
environment variable (see `.env.example`). Run them from the repo root.
