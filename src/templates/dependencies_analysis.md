# Import dependencies

Dependencies are inferred from source imports, not resolved package manifests.

{% for repo, dependencies_list in dependencies.dependency_graph.items() %}
## {{ repo }}

{% for dependency in dependencies_list %}
- `{{ dependency }}`
{% endfor %}
{% endfor %}
