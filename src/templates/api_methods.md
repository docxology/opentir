# Functions and methods

{{ total_methods }} extracted functions and methods.

{% for item in methods %}
## {{ item.repository }}: {{ item.element.name }}

`{{ item.element.file_path }}:{{ item.element.line_number }}`

Parameters: {{ item.element.parameters|join(', ') }}

{{ item.element.docstring }}
{% endfor %}
