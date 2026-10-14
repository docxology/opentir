# {{ language.title() }} repositories

{{ total_repos }} repositories; {{ total_elements }} extracted elements.

{% for repo in repositories %}
- [{{ repo.repo_name }}](../{{ repo.repo_name }}.md)
{% endfor %}
