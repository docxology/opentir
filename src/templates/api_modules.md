# Modules

{{ total_modules }} source modules containing extracted elements.

{% for module, elements in modules.items() %}
## {{ module }}

{% for element in elements %}
- {{ element.type }}: `{{ element.name }}` (line {{ element.line_number }})
{% endfor %}
{% endfor %}
