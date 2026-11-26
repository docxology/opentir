# Extraction and reporting {#sec:method}

The Python parser maps syntax to an abstract syntax tree, as documented by
the Python standard library [@python_ast]. Opentir walks that tree and records
elements and their source locations. It derives file and repository summaries
from those records. Pattern-based extraction for other languages uses
different rules.

The command-line analysis emits per-repository and aggregate JSON. Its
offline documentation command reads that JSON to generate Markdown and a
MkDocs configuration. MkDocs builds the site from those sources
[@mkdocs_user_guide].
