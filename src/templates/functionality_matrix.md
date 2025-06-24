# Comprehensive Functionality Matrix

## Global Overview

- **Total Repositories**: {{ total_repositories }}
- **Total Languages**: {{ global_summary.languages|length }}
- **Generated**: {{ generation_date }}

## Repository Comparison

| Repository | Languages | Files | Lines | Elements | Public Methods | Classes | Complexity |
|------------|-----------|-------|-------|----------|----------------|---------|------------|
{% for repo in functionality_table %}
| {{ repo.repository }} | {{ repo.languages|join(", ") }} | {{ repo.total_files }} | {{ repo.total_lines:,}} | {{ repo.total_elements }} | {{ repo.public_methods }} | {{ repo.classes }} | {{ repo.complexity }} |
{% endfor %}

## Method Frequency Analysis

### Most Common Methods

{% for method, count in method_analysis.most_common_methods[:20] %}
- **{{ method }}**: Used {{ count }} times
{% endfor %}

### Method Distribution

- **Unique Methods**: {{ method_analysis.method_distribution.unique_methods:,}}
- **Methods Used Once**: {{ method_analysis.method_distribution.methods_used_once:,}}
- **Methods Used 5+ Times**: {{ method_analysis.method_distribution.methods_used_5_plus:,}}
- **Methods Used 10+ Times**: {{ method_analysis.method_distribution.methods_used_10_plus:,}}

### Common Naming Patterns

{% for pattern in method_analysis.common_patterns %}
- **{{ pattern.pattern }}**: {{ pattern.count }} occurrences
{% endfor %}
