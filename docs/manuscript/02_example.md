# Worked local example {#sec:example}

The source fixture models a small inventory and has a function and a class
with a method. The analyzer reported one Python file and these source
elements:

| Name | Extracted type |
| --- | --- |
| `item_count` | Function |
| `Inventory` | Class |
| `is_empty` | Method |

The verification checks the paths to the generated analysis, Markdown, and
built HTML. It confirms that the expected identifiers and their types
survive the local extraction and reporting workflow. The fixture remains
source input throughout the run.
