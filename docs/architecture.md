# Package map

Generated on 2026-11-12 from the `import` statements under `src/`. An arrow from A to B means code in A imports B; arrows already implied by a longer path are left out, so the picture shows layering rather than every edge. It shows 9 of the package's 10 top-level parts: the largest ones that import or are imported by others.

```mermaid
flowchart TB
    accTitle: Internal dependency map of the src package in opentir
    accDescr: Generated from import statements in src. Each box is a module of the package; an arrow from A to B means code in A imports B; arrows already implied by a longer path are omitted. Most-imported: config, utils, code_analyzer.
    n_multi_repo_analyzer["multi_repo_analyzer.py"]
    n_code_analyzer["code_analyzer.py"]
    n_docs_generator["docs_generator.py"]
    n_repo_manager["repo_manager.py"]
    n_github_client["github_client.py"]
    n_cli["cli.py"]
    n_main["main.py"]
    n_utils["utils.py"]
    n_config["config.py"]
    n_utils --> n_config
    n_cli --> n_repo_manager
    n_cli --> n_docs_generator
    n_cli --> n_multi_repo_analyzer
    n_code_analyzer --> n_utils
    n_docs_generator --> n_code_analyzer
    n_github_client --> n_utils
    n_main --> n_repo_manager
    n_main --> n_docs_generator
    n_multi_repo_analyzer --> n_code_analyzer
    n_repo_manager --> n_github_client
```
