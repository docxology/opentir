# Analyze a local fixture

Prepare the locked environment using the [setup guide](setup.md). The commands
below read the checked-in [inventory source](examples/fixture/stock.py) as text.
Run them from the repository root:

```sh
(
  set -eu
  root="$(pwd -P)"
  work="$root/output/local-analysis"
  mkdir -p "$work"
  cd "$work"
  unset GITHUB_TOKEN
  export BASE_DIR="$work"

  "$root/.venv/bin/opentir" analyze \
    --repo-path "$root/docs/examples/fixture" --output-dir "$work/analysis"
  "$root/.venv/bin/opentir" generate-docs --offline \
    --analysis-dir "$work/analysis" --output-dir "$work/site-source"
  "$root/.venv/bin/mkdocs" build --strict -f "$work/site-source.mkdocs.yml"

  python3 - <<'PY'
import json
from pathlib import Path

analysis = json.loads(Path("analysis/fixture_analysis.json").read_text())
elements = {(element["name"], element["type"]) for element in analysis["all_elements"]}
assert analysis["total_files"] == 1
assert elements == {("item_count", "function"), ("Inventory", "class"), ("is_empty", "method")}
assert Path("analysis/all_analyses.json").is_file()
assert Path("analysis/functionality_matrix.json").is_file()
assert Path("site-source/index.md").is_file()
assert Path("site-source_site/index.html").is_file()
print("PASS: one file, three source elements, and a built offline documentation site")
PY
)
```

The analyzer writes repository JSON, an aggregate analysis file, and a
functionality matrix. Documentation generation reads those structured
files and writes Markdown plus `site-source.mkdocs.yml`. MkDocs builds
`site-source_site/`. The explicit `--offline` flag skips organization lookup
during documentation generation.

The workflow exercises the analyzer's Python parser and local reporting.
The inventory module is never imported or run. Older analysis files with
stringified code elements need to be regenerated before report generation.
