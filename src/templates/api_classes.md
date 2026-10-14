# Classes

{{ total_classes }} extracted classes.

{% for item in classes %}
## {{ item.repository }}: {{ item.element.name }}

`{{ item.element.file_path }}:{{ item.element.line_number }}`

{{ item.element.docstring }}
{% endfor %}
