# Method and limitations

The [analyzer](../src/code_analyzer.py) uses Python's abstract syntax tree to
extract functions, classes, methods, parameters, and annotations. JavaScript,
TypeScript, Java, and Go use regular-expression heuristics. The list of
recognized extensions is wider than the list of supported parsers.

Line counts include nonblank comments. Categories come from source names
and docstrings. Import records describe lexical references. They do not
resolve installed dependency versions. Complexity values are parser-specific
proxies. Treat any comparison across source languages or historical snapshots
as exploratory.

The local walkthrough examines one authored source file and its generated
documentation. Results describe that file and its parser behavior. The
[technical note](manuscript/README.md) records the observed elements.
