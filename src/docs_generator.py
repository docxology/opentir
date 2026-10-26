"""
Documentation generator for creating comprehensive docs with vast functionality tables.
Generates organized documentation structure with MkDocs and detailed analysis reports.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from jinja2 import Environment, FileSystemLoader
import pandas as pd

from .config import config, DOCS_TEMPLATE_STRUCTURE
from .utils import Logger, FileUtils, format_bytes, format_duration, sanitize_filename
from .code_analyzer import RepositoryAnalysis, CodeElement

class DocumentationGenerator:
    """
    Comprehensive documentation generator for Palantir ecosystem analysis.
    Creates organized docs with functionality matrices and detailed analysis.
    """
    
    def __init__(self, docs_dir: Optional[Path] = None):
        """Initialize documentation generator."""
        self.docs_dir = docs_dir or (config.base_dir / config.docs_dir)
        self.logger = Logger("docs_generator")
        
        # Ensure docs directory exists
        FileUtils.ensure_directory(self.docs_dir)
        self._setup_docs_structure()
        
        # Setup Jinja2 template environment
        template_dir = Path(__file__).parent / "templates"
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=False
        )
        
        # Initialize template directory if it doesn't exist
        self._setup_templates()
    
    def _setup_templates(self) -> None:
        """Setup template directory with default templates."""
        for name in (
            "main_index.md", "repository_detail.md", "functionality_matrix.md",
            "language_index.md", "category_index.md", "api_methods.md",
            "api_classes.md", "api_modules.md", "dependencies_analysis.md",
            "metrics_analysis.md",
        ):
            self.jinja_env.get_template(name)
    
    def generate_complete_documentation(
        self, 
        repository_analyses: List[RepositoryAnalysis],
        functionality_matrix: Dict[str, Any],
        org_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate complete documentation suite with all components.
        Creates comprehensive docs with vast functionality tables.
        """
        self.logger.info("Generating complete documentation suite...")
        for analysis in repository_analyses:
            if analysis.repo_name in {".", ".."} or Path(analysis.repo_name).name != analysis.repo_name:
                raise ValueError("Repository name must be a single directory name")
        
        # Setup directory structure
        self._setup_docs_structure()
        
        # Generate main documentation components
        results = {
            "main_index": self._generate_main_index(org_info, repository_analyses),
            "repository_docs": self._generate_repository_documentation(repository_analyses),
            "functionality_matrix": self._generate_functionality_matrix_docs(functionality_matrix),
            "api_reference": self._generate_api_reference(repository_analyses),
            "analysis_reports": self._generate_analysis_reports(repository_analyses),
            "mkdocs_config": self._generate_mkdocs_config(),
        }
        
        # Generate summary statistics
        results["generation_summary"] = self._generate_summary_stats(repository_analyses)
        
        self.logger.info("Documentation generation completed successfully")
        return results
    
    def _setup_docs_structure(self) -> None:
        """Setup organized documentation directory structure."""
        structure_dirs = [
            "repositories/by_language",
            "repositories/by_category", 
            "repositories/popular",
            "api_reference/methods",
            "api_reference/classes",
            "api_reference/modules",
            "analysis/functionality",
            "analysis/metrics",
            "analysis/dependencies",
            "guides",
            "assets/images",
            "assets/data",
        ]
        
        for dir_path in structure_dirs:
            FileUtils.ensure_directory(self.docs_dir / dir_path)
    
    def _generate_main_index(self, org_info: Dict[str, Any], analyses: List[RepositoryAnalysis]) -> str:
        """Generate main index page with overview and navigation."""
        template = self.jinja_env.get_template("main_index.md")
        
        # Calculate overview statistics
        total_repos = len(analyses)
        total_files = sum(a.total_files for a in analyses)
        total_lines = sum(a.total_lines for a in analyses)
        languages = {}
        
        for analysis in analyses:
            for lang, count in analysis.languages.items():
                languages[lang] = languages.get(lang, 0) + count
        
        top_languages = sorted(languages.items(), key=lambda x: x[1], reverse=True)[:10]
        popular_repos = sorted(analyses, key=lambda x: len(x.all_elements), reverse=True)[:10]
        
        content = template.render(
            org_info=org_info,
            total_repositories=total_repos,
            total_files=total_files,
            total_lines=total_lines,
            top_languages=top_languages,
            popular_repositories=popular_repos,
            generation_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        
        index_path = self.docs_dir / "index.md"
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return str(index_path)
    
    def _generate_repository_documentation(self, analyses: List[RepositoryAnalysis]) -> Dict[str, List[str]]:
        """Generate detailed documentation for each repository."""
        self.logger.info("Generating individual repository documentation...")
        
        generated_files = {
            "individual": [],
            "by_language": [],
            "by_category": [],
        }
        
        # Generate individual repository docs
        repo_template = self.jinja_env.get_template("repository_detail.md")
        
        for analysis in analyses:
            content = repo_template.render(
                repo=analysis,
                top_elements=self._get_top_elements_for_display(analysis.all_elements),
                functionality_summary=analysis.functionality_summary,
                metrics=analysis.metrics,
            )
            
            repo_file = self.docs_dir / "repositories" / f"{analysis.repo_name}.md"
            with open(repo_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            generated_files["individual"].append(str(repo_file))

        index_file = self.docs_dir / "repositories" / "index.md"
        index_file.write_text(
            "# Repositories\n\n" + "\n".join(
                f"- [{a.repo_name}]({a.repo_name}.md)" for a in analyses
            ) + "\n", encoding="utf-8",
        )
        
        # Generate language-based organization
        language_groups = self._group_analyses_by_language(analyses)
        language_template = self.jinja_env.get_template("language_index.md")
        
        for language, repos in language_groups.items():
            content = language_template.render(
                language=language,
                repositories=repos,
                total_repos=len(repos),
                total_elements=sum(len(r.all_elements) for r in repos),
            )
            
            lang_file = self.docs_dir / "repositories" / "by_language" / f"{sanitize_filename(language)}.md"
            with open(lang_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            generated_files["by_language"].append(str(lang_file))
        
        # Generate category-based organization
        category_groups = self._group_analyses_by_category(analyses)
        category_template = self.jinja_env.get_template("category_index.md")
        
        for category, repos in category_groups.items():
            content = category_template.render(
                category=category,
                repositories=repos,
                total_repos=len(repos),
            )
            
            cat_file = self.docs_dir / "repositories" / "by_category" / f"{category}.md"
            with open(cat_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            generated_files["by_category"].append(str(cat_file))
        
        return generated_files
    
    def _generate_functionality_matrix_docs(self, functionality_matrix: Dict[str, Any]) -> str:
        """Generate vast functionality matrix documentation."""
        self.logger.info("Generating comprehensive functionality matrix...")
        
        # Create main functionality matrix page
        matrix_template = self.jinja_env.get_template("functionality_matrix.md")
        
        # Prepare data for display
        global_summary = functionality_matrix.get("global_summary", {})
        repositories = functionality_matrix.get("repositories", {})
        
        # Create functionality comparison table
        functionality_table = self._create_functionality_comparison_table(repositories)
        
        # Create method frequency analysis
        method_analysis = self._analyze_method_frequency(global_summary)
        
        content = matrix_template.render(
            global_summary=global_summary,
            functionality_table=functionality_table,
            method_analysis=method_analysis,
            total_repositories=len(repositories),
            generation_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        
        matrix_file = self.docs_dir / "analysis" / "functionality_matrix.md"
        with open(matrix_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Generate detailed CSV exports for analysis
        self._export_functionality_data(functionality_matrix)
        
        return str(matrix_file)
    
    def _create_functionality_comparison_table(self, repositories: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create comprehensive functionality comparison table."""
        table_data = []
        
        for repo_name, repo_data in repositories.items():
            basic_info = repo_data.get("basic_info", {})
            functionality = repo_data.get("functionality", {})
            metrics = repo_data.get("metrics", {})
            
            row = {
                "repository": repo_name,
                "languages": list(basic_info.get("languages", {}).keys()),
                "total_files": basic_info.get("total_files", 0),
                "total_lines": basic_info.get("total_lines", 0),
                "total_elements": functionality.get("total_elements", 0),
                "public_methods": len(functionality.get("public_methods", [])),
                "classes": len(functionality.get("classes", [])),
                "async_functions": len(functionality.get("async_functions", [])),
                "complexity": metrics.get("total_complexity", 0),
                "documentation_ratio": metrics.get("documentation_ratio", 0),
                "main_categories": list(functionality.get("functionality_categories", {}).keys())[:5],
            }
            
            table_data.append(row)
        
        # Sort by total elements descending
        table_data.sort(key=lambda x: x["total_elements"], reverse=True)
        
        return table_data
    
    def _analyze_method_frequency(self, global_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze method frequency across all repositories."""
        top_methods = global_summary.get("top_methods", {})
        
        return {
            "most_common_methods": list(top_methods.items())[:50],
            "method_distribution": {
                "unique_methods": len(top_methods),
                "methods_used_once": len([m for m, count in top_methods.items() if count == 1]),
                "methods_used_5_plus": len([m for m, count in top_methods.items() if count >= 5]),
                "methods_used_10_plus": len([m for m, count in top_methods.items() if count >= 10]),
            },
            "common_patterns": self._identify_common_patterns(top_methods),
        }
    
    def _identify_common_patterns(self, methods: Dict[str, int]) -> List[Dict[str, Any]]:
        """Identify common naming patterns in methods."""
        patterns = {}
        
        for method_name in methods.keys():
            # Analyze prefixes
            if method_name.startswith(('get_', 'set_', 'is_', 'has_', 'can_')):
                prefix = method_name.split('_')[0] + '_'
                patterns[prefix] = patterns.get(prefix, 0) + 1
            
            # Analyze suffixes
            if method_name.endswith(('_test', '_util', '_helper', '_factory')):
                suffix = '_' + method_name.split('_')[-1]
                patterns[suffix] = patterns.get(suffix, 0) + 1
        
        return [
            {"pattern": pattern, "count": count}
            for pattern, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:20]
        ]
    
    def _export_functionality_data(self, functionality_matrix: Dict[str, Any]) -> None:
        """Export functionality data to CSV/JSON for external analysis."""
        export_dir = self.docs_dir / "assets" / "data"
        
        # Export repository comparison data
        repositories = functionality_matrix.get("repositories", {})
        comparison_data = self._create_functionality_comparison_table(repositories)
        
        df = pd.DataFrame(comparison_data)
        df.to_csv(export_dir / "repository_comparison.csv", index=False)
        
        # Export method frequency data
        global_summary = functionality_matrix.get("global_summary", {})
        top_methods = global_summary.get("top_methods", {})
        
        methods_df = pd.DataFrame([
            {"method": method, "frequency": freq}
            for method, freq in top_methods.items()
        ])
        methods_df.to_csv(export_dir / "method_frequency.csv", index=False)
        
        # Export full functionality matrix as JSON
        with open(export_dir / "full_functionality_matrix.json", 'w') as f:
            json.dump(functionality_matrix, f, indent=2, default=str)
    
    def _generate_api_reference(self, analyses: List[RepositoryAnalysis]) -> Dict[str, str]:
        """Generate comprehensive API reference documentation."""
        self.logger.info("Generating API reference documentation...")
        
        generated_files = {}
        
        # Collect all elements by type
        all_methods = []
        all_classes = []
        all_modules = {}
        
        for analysis in analyses:
            for element in analysis.all_elements:
                if element.type in ["function", "method"]:
                    all_methods.append({
                        "element": element,
                        "repository": analysis.repo_name,
                    })
                elif element.type == "class":
                    all_classes.append({
                        "element": element,
                        "repository": analysis.repo_name,
                    })
                
                # Group by file/module
                module_key = f"{analysis.repo_name}:{element.file_path}"
                if module_key not in all_modules:
                    all_modules[module_key] = []
                all_modules[module_key].append(element)
        
        # Generate methods reference
        methods_template = self.jinja_env.get_template("api_methods.md")
        methods_content = methods_template.render(
            methods=sorted(all_methods, key=lambda x: x["element"].name),
            total_methods=len(all_methods),
        )
        
        methods_file = self.docs_dir / "api_reference" / "methods" / "index.md"
        with open(methods_file, 'w', encoding='utf-8') as f:
            f.write(methods_content)
        generated_files["methods"] = str(methods_file)
        
        # Generate classes reference
        classes_template = self.jinja_env.get_template("api_classes.md")
        classes_content = classes_template.render(
            classes=sorted(all_classes, key=lambda x: x["element"].name),
            total_classes=len(all_classes),
        )
        
        classes_file = self.docs_dir / "api_reference" / "classes" / "index.md"
        with open(classes_file, 'w', encoding='utf-8') as f:
            f.write(classes_content)
        generated_files["classes"] = str(classes_file)
        
        # Generate modules reference
        modules_template = self.jinja_env.get_template("api_modules.md")
        modules_content = modules_template.render(
            modules=all_modules,
            total_modules=len(all_modules),
        )
        
        modules_file = self.docs_dir / "api_reference" / "modules" / "index.md"
        with open(modules_file, 'w', encoding='utf-8') as f:
            f.write(modules_content)
        generated_files["modules"] = str(modules_file)
        
        return generated_files
    
    def _generate_analysis_reports(self, analyses: List[RepositoryAnalysis]) -> Dict[str, str]:
        """Generate comprehensive analysis reports."""
        self.logger.info("Generating analysis reports...")
        
        generated_files = {}
        
        # Dependencies analysis
        dependencies_data = self._analyze_dependencies(analyses)
        deps_template = self.jinja_env.get_template("dependencies_analysis.md")
        deps_content = deps_template.render(dependencies=dependencies_data)
        
        deps_file = self.docs_dir / "analysis" / "dependencies.md"
        with open(deps_file, 'w', encoding='utf-8') as f:
            f.write(deps_content)
        generated_files["dependencies"] = str(deps_file)
        
        # Metrics analysis
        metrics_data = self._analyze_metrics(analyses)
        metrics_template = self.jinja_env.get_template("metrics_analysis.md")
        metrics_content = metrics_template.render(metrics=metrics_data)
        
        metrics_file = self.docs_dir / "analysis" / "metrics.md"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            f.write(metrics_content)
        generated_files["metrics"] = str(metrics_file)
        
        return generated_files
    
    def _generate_mkdocs_config(self) -> str:
        """Generate MkDocs configuration file."""
        import yaml
        config_content = yaml.safe_dump({
            "site_name": config.docs_site_name,
            "site_description": config.docs_site_description,
            "docs_dir": self.docs_dir.name,
            "site_dir": f"{self.docs_dir.name}_site",
            "theme": {"name": config.docs_theme},
            "nav": [
                {"Home": "index.md"},
                {"Repositories": "repositories/index.md"},
                {"Functions": "api_reference/methods/index.md"},
                {"Classes": "api_reference/classes/index.md"},
                {"Modules": "api_reference/modules/index.md"},
                {"Functionality": "analysis/functionality_matrix.md"},
                {"Dependencies": "analysis/dependencies.md"},
                {"Metrics": "analysis/metrics.md"},
            ],
            "markdown_extensions": ["tables", "admonition", "toc"],
        }, sort_keys=False)

        config_file = self.docs_dir.parent / f"{self.docs_dir.name}.mkdocs.yml"
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        return str(config_file)
    
    def _create_default_templates(self, template_dir: Path) -> None:
        """Copy bundled templates into a caller-owned directory without overwriting."""
        source_dir = Path(__file__).parent / "templates"
        template_dir.mkdir(parents=True, exist_ok=True)
        for source in source_dir.glob("*.md"):
            if source.name in {"AGENTS.md", "README.md"}:
                continue
            destination = template_dir / source.name
            if not destination.exists():
                shutil.copyfile(source, destination)
        destination = template_dir / "mkdocs.yml"
        if not destination.exists():
            destination.write_text("site_name: Opentir\n", encoding="utf-8")

    
    def _get_top_elements_for_display(self, elements: List[CodeElement]) -> Dict[str, List[Dict[str, Any]]]:
        """Get top elements organized by type for display."""
        element_types = {
            "functions": [e for e in elements if e.type == "function"],
            "methods": [e for e in elements if e.type == "method"],
            "classes": [e for e in elements if e.type == "class"],
        }
        
        result = {}
        for element_type, type_elements in element_types.items():
            sorted_elements = sorted(type_elements, key=lambda x: x.complexity, reverse=True)
            result[element_type] = [
                {
                    "name": e.name,
                    "file": e.file_path,
                    "complexity": e.complexity,
                    "parameters": e.parameters,
                    "docstring": e.docstring,
                }
                for e in sorted_elements[:20]
            ]
        
        return result
    
    def _group_analyses_by_language(self, analyses: List[RepositoryAnalysis]) -> Dict[str, List[RepositoryAnalysis]]:
        """Group repository analyses by primary language."""
        groups = {}
        
        for analysis in analyses:
            if analysis.languages:
                primary_lang = max(analysis.languages.items(), key=lambda x: x[1])[0]
                if primary_lang not in groups:
                    groups[primary_lang] = []
                groups[primary_lang].append(analysis)
        
        return groups
    
    def _group_analyses_by_category(self, analyses: List[RepositoryAnalysis]) -> Dict[str, List[RepositoryAnalysis]]:
        """Group repository analyses by functionality category."""
        groups = {
            "high_activity": [],
            "medium_activity": [],
            "low_activity": [],
            "archived": [],
        }
        
        for analysis in analyses:
            if analysis.total_lines > 10000:
                groups["high_activity"].append(analysis)
            elif analysis.total_lines > 1000:
                groups["medium_activity"].append(analysis)
            else:
                groups["low_activity"].append(analysis)
        
        return groups
    
    def _analyze_dependencies(self, analyses: List[RepositoryAnalysis]) -> Dict[str, Any]:
        """Analyze dependencies across all repositories."""
        from collections import Counter
        counts = Counter()
        dependency_graph = {}
        for analysis in analyses:
            dependencies = sorted({
                dependency for file in analysis.file_analyses
                for dependency in file.dependencies
            })
            dependency_graph[analysis.repo_name] = dependencies
            counts.update(dependencies)
        return {
            "common_dependencies": [name for name, _ in counts.most_common(20)],
            "dependency_counts": dict(counts),
            "dependency_graph": dependency_graph,
        }
    
    def _analyze_metrics(self, analyses: List[RepositoryAnalysis]) -> Dict[str, Any]:
        """Analyze metrics across all repositories."""
        if not analyses:
            return {}
        
        total_complexity = sum(a.metrics.get("total_complexity", 0) for a in analyses)
        avg_complexity = total_complexity / len(analyses)
        
        return {
            "average_repository_complexity": avg_complexity,
            "complexity_distribution": {
                "high": len([a for a in analyses if a.metrics.get("total_complexity", 0) > avg_complexity * 1.5]),
                "medium": len([a for a in analyses if avg_complexity * 0.5 < a.metrics.get("total_complexity", 0) <= avg_complexity * 1.5]),
                "low": len([a for a in analyses if a.metrics.get("total_complexity", 0) <= avg_complexity * 0.5]),
            },
            "documentation_coverage": sum(a.metrics.get("documentation_ratio", 0) for a in analyses) / len(analyses),
        }
    
    def _generate_summary_stats(self, analyses: List[RepositoryAnalysis]) -> Dict[str, Any]:
        """Generate summary statistics for the documentation generation."""
        return {
            "total_repositories_analyzed": len(analyses),
            "total_files_documented": sum(a.total_files for a in analyses),
            "total_lines_analyzed": sum(a.total_lines for a in analyses),
            "total_elements_extracted": sum(len(a.all_elements) for a in analyses),
            "generation_timestamp": datetime.now().isoformat(),
        }
