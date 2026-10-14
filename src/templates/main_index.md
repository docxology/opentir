# {{ org_info.name or "Palantir" }} Open Source Ecosystem

Welcome to the comprehensive documentation for Palantir's open source ecosystem.

## Overview

- **Total Repositories**: {{ total_repositories }}
- **Total Files Analyzed**: {{ "{:,}".format(total_files) }}
- **Total Lines of Code**: {{ "{:,}".format(total_lines) }}
- **Generated**: {{ generation_date }}

## Top Programming Languages

{% for language, count in top_languages %}
- **{{ language.title() }}**: {{ count }} files
{% endfor %}

## Popular Repositories

{% for repo in popular_repositories %}
- [{{ repo.repo_name }}](repositories/{{ repo.repo_name }}.md) - {{ repo.total_files }} files, {{ "{:,}".format(repo.total_lines) }} lines
{% endfor %}

## Navigation

- [Repository Documentation](repositories/index.md)
- [Functionality Matrix](analysis/functionality_matrix.md)
- [API Reference](api_reference/methods/index.md)
- [Analysis Reports](analysis/metrics.md)
