# {{ category.replace('_', ' ').title() }}

These groups describe source size, not maintenance activity or archival status.

{% for repo in repositories %}
- [{{ repo.repo_name }}](../{{ repo.repo_name }}.md): {{ repo.total_lines }} nonblank lines
{% endfor %}
