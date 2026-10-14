# Source metrics

{% if metrics %}
- Mean summed repository complexity: {{ metrics.average_repository_complexity }}
- Fraction of elements with documentation: {{ metrics.documentation_coverage }}

{% for category, count in metrics.complexity_distribution.items() %}
- {{ category }} complexity: {{ count }} repositories
{% endfor %}
{% else %}
No repositories were analyzed.
{% endif %}
