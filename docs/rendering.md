# Render the technical note

Run the [local analysis walkthrough](local-analysis.md) to verify the example.
The [canonical manuscript](manuscript/README.md) uses the external
[docxology/template](https://github.com/docxology/template) engine. Install uv,
Python 3.14, Pandoc with a matching `pandoc-crossref`, and XeLaTeX with its
packages.

From the root of a committed checkout named `opentir`, choose an unused path
for the renderer clone:

```sh
OPENTIR_ROOT="$(pwd -P)"
TEMPLATE_ROOT="$OPENTIR_ROOT/../opentir-render-template"
git clone https://github.com/docxology/template.git "$TEMPLATE_ROOT"
git -C "$TEMPLATE_ROOT" switch --detach 87704feae0a4490ef7dad9c931d268e8c30273c2

cd "$TEMPLATE_ROOT"
uv sync --locked --no-dev --python 3.14
mkdir -p projects/working
ln -s "$OPENTIR_ROOT" projects/working/opentir
uv run --no-sync python scripts/pipeline/stage_03_render.py --project working/opentir --skip-manuscript-hydration
uv run --no-sync python scripts/pipeline/stage_04_validate.py --project working/opentir
```

Find the combined PDF at `output/pdf/opentir_combined.pdf`, HTML at
`output/web/index.html`, and validation and provenance receipts in
`output/reports/` in the Opentir checkout.
