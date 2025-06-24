# {{ repo.repo_name }}

## Overview

- **Total Files**: {{ repo.total_files }}
- **Total Lines**: {{ repo.total_lines:,}}
- **Languages**: {{ repo.languages.keys()|list|join(", ") }}

## Functionality Summary

- **Total Elements**: {{ repo.functionality_summary.total_elements }}
- **Public Methods**: {{ repo.functionality_summary.public_methods|length }}
- **Classes**: {{ repo.functionality_summary.classes|length }}
- **Async Functions**: {{ repo.functionality_summary.async_functions|length }}

## Top Elements

{% for element_type, elements in top_elements.items() %}
### {{ element_type.title() }}

{% for element in elements[:10] %}
- **{{ element.name }}** ({{ element.file }}) - Complexity: {{ element.complexity }}
  {% if element.docstring %}
  > {{ element.docstring[:100] }}...
  {% endif %}
{% endfor %}

{% endfor %}

## Metrics

- **Average File Complexity**: {{ "%.2f"|format(repo.metrics.average_file_complexity or 0) }}
- **Documentation Ratio**: {{ "%.1f"|format((repo.metrics.documentation_ratio or 0) * 100) }}%
