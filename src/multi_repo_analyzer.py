"""
Multi-Repository Analyzer for comprehensive cross-repository analysis.
Provides descriptive, analytical, extractive, and visualization methods for all repositories.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from .code_analyzer import CodeAnalyzer, RepositoryAnalysis, CodeElement
from .utils import Logger, FileUtils
from .config import config

@dataclass
class CrossRepoMetrics:
    """Comprehensive metrics across all repositories."""
    total_repositories: int
    total_files: int
    total_lines_of_code: int
    total_code_elements: int
    language_distribution: Dict[str, int]
    functionality_patterns: Dict[str, List[str]]
    complexity_statistics: Dict[str, float]
    repository_rankings: Dict[str, Any]
    cross_repo_dependencies: Dict[str, List[str]]

class MultiRepositoryAnalyzer:
    """
    Comprehensive analyzer for multiple repositories.
    Provides cross-repository insights, patterns, and visualizations.
    """
    
    def __init__(self, repos_base_path: Path = None):
        """Initialize multi-repository analyzer."""
        self.repos_base_path = repos_base_path or Path(config.base_dir) / config.repos_dir
        self.logger = Logger("multi_repo_analyzer")
        self.analyzer = CodeAnalyzer()
        self.output_dir = Path(config.base_dir) / "multi_repo_analysis"
        self.output_dir.mkdir(exist_ok=True)
        
    def analyze_all_repositories(self, force_reanalyze: bool = False) -> CrossRepoMetrics:
        """
        Analyze all repositories and generate comprehensive cross-repository metrics.
        
        Args:
            force_reanalyze: Whether to force re-analysis of existing results
            
        Returns:
            CrossRepoMetrics: Comprehensive metrics across all repositories
        """
        self.logger.info(f"Starting comprehensive multi-repository analysis...")
        
        # Check for existing analysis
        cache_file = self.output_dir / "cross_repo_analysis.json"
        if cache_file.exists() and not force_reanalyze:
            self.logger.info("Loading existing analysis from cache...")
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                cached_metrics = CrossRepoMetrics(**cached_data)
                
                # Only use cached data if it has a reasonable number of repositories
                if cached_metrics.total_repositories > 10:
                    return cached_metrics
                else:
                    self.logger.info(f"Cached analysis only has {cached_metrics.total_repositories} repositories, running fresh analysis")
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}, proceeding with fresh analysis")
        
        # Get all repository paths
        repo_paths = self._get_all_repository_paths()
        self.logger.info(f"Found {len(repo_paths)} repositories to analyze")
        
        if len(repo_paths) == 0:
            raise ValueError("No repositories found to analyze")
        
        # Analyze each repository
        repository_analyses = []
        failed_analyses = []
        
        for i, repo_path in enumerate(repo_paths, 1):
            try:
                if i % 25 == 0 or i <= 10:  # Log first 10 and every 25th
                    self.logger.info(f"Analyzing repository {i}/{len(repo_paths)}: {repo_path.name}")
                
                # Skip if repo is too small or likely empty
                code_files = self._count_code_files(repo_path)
                if code_files < 1:
                    self.logger.debug(f"Skipping {repo_path.name}: no code files found")
                    continue
                
                analysis = self.analyzer.analyze_repository(repo_path)
                if analysis and analysis.total_files > 0:
                    repository_analyses.append(analysis)
                else:
                    self.logger.debug(f"Skipping {repo_path.name}: no analyzable content")
                    
            except Exception as e:
                self.logger.warning(f"Failed to analyze {repo_path.name}: {e}")
                failed_analyses.append({"repo": repo_path.name, "error": str(e)})
        
        self.logger.info(f"Successfully analyzed {len(repository_analyses)} repositories")
        
        if len(repository_analyses) == 0:
            raise ValueError("No repositories could be successfully analyzed")
        
        # Generate cross-repository metrics
        metrics = self._generate_cross_repo_metrics(repository_analyses)
        
        # Save individual analyses
        self._save_individual_analyses(repository_analyses)
        
        # Save cross-repository metrics
        with open(cache_file, 'w') as f:
            json.dump(asdict(metrics), f, indent=2, default=str)
        
        # Save failed analyses log
        if failed_analyses:
            with open(self.output_dir / "failed_analyses.json", 'w') as f:
                json.dump(failed_analyses, f, indent=2)
        
        self.logger.info(f"Completed analysis of {len(repository_analyses)} repositories")
        return metrics
    
    def _get_all_repository_paths(self) -> List[Path]:
        """Get all repository paths from the repos directory."""
        all_repos_dir = self.repos_base_path / "all_repos"
        
        if not all_repos_dir.exists():
            raise ValueError(f"Repository directory does not exist: {all_repos_dir}")
        
        return [path for path in all_repos_dir.iterdir() if path.is_dir() and not path.name.startswith('.')]
    
    def _count_code_files(self, repo_path: Path) -> int:
        """Count the number of code files in a repository."""
        code_extensions = {'.py', '.js', '.ts', '.java', '.go', '.cpp', '.c', '.h', '.hpp', '.rs', '.rb', '.php', '.scala', '.kt', '.swift', '.dart', '.r', '.R', '.jl', '.pl', '.sh', '.bash', '.zsh', '.fish', '.ps1', '.bat', '.cmd', '.yaml', '.yml', '.json', '.xml', '.toml', '.ini', '.cfg', '.conf', '.properties', '.gradle', '.groovy', '.sbt', '.pom', '.gemspec', '.podspec', '.dockerfile', '.makefile', '.cmake', '.sql', '.html', '.css', '.scss', '.sass', '.less', '.vue', '.jsx', '.tsx', '.svelte', '.elm', '.hs', '.clj', '.cljs', '.ml', '.mli', '.fs', '.fsi', '.fsx', '.vb', '.cs', '.fs', '.ps1', '.psm1', '.psd1', '.lua', '.tcl', '.awk', '.sed', '.vim', '.tex', '.md', '.rst', '.txt'}
        
        try:
            count = 0
            for file_path in repo_path.rglob("*"):
                if (file_path.is_file() and 
                    file_path.suffix.lower() in code_extensions and
                    not self._should_skip_file_path(file_path)):
                    count += 1
                    if count >= 5:  # Early exit if we find enough files
                        break
            return count
        except Exception:
            return 0
    
    def _should_skip_file_path(self, file_path: Path) -> bool:
        """Check if file path should be skipped during analysis."""
        skip_patterns = [
            ".git/", "node_modules/", "__pycache__/", ".pytest_cache/",
            "build/", "dist/", "target/", ".gradle/", "vendor/", "venv/",
            ".venv/", ".env/", ".tox/", ".coverage/", ".mypy_cache/",
            ".DS_Store", "Thumbs.db", ".idea/", ".vscode/", "*.log",
            "*.tmp", "*.temp", "*.bak", "*.swp", "*.swo"
        ]
        
        path_str = str(file_path)
        return any(pattern in path_str for pattern in skip_patterns)
    
    def _generate_cross_repo_metrics(self, analyses: List[RepositoryAnalysis]) -> CrossRepoMetrics:
        """Generate comprehensive cross-repository metrics."""
        total_repositories = len(analyses)
        total_files = sum(a.total_files for a in analyses)
        total_lines = sum(a.total_lines for a in analyses)
        total_elements = sum(len(a.all_elements) for a in analyses)
        
        # Language distribution
        language_dist = defaultdict(int)
        for analysis in analyses:
            for lang, count in analysis.languages.items():
                language_dist[lang] += count
        
        # Functionality patterns
        functionality_patterns = self._extract_functionality_patterns(analyses)
        
        # Complexity statistics
        complexity_stats = self._calculate_complexity_statistics(analyses)
        
        # Repository rankings
        repo_rankings = self._calculate_repository_rankings(analyses)
        
        # Cross-repository dependencies
        cross_deps = self._analyze_cross_dependencies(analyses)
        
        return CrossRepoMetrics(
            total_repositories=total_repositories,
            total_files=total_files,
            total_lines_of_code=total_lines,
            total_code_elements=total_elements,
            language_distribution=dict(language_dist),
            functionality_patterns=functionality_patterns,
            complexity_statistics=complexity_stats,
            repository_rankings=repo_rankings,
            cross_repo_dependencies=cross_deps,
        )
    
    def _extract_functionality_patterns(self, analyses: List[RepositoryAnalysis]) -> Dict[str, List[str]]:
        """Extract common functionality patterns across repositories."""
        patterns = defaultdict(list)
        
        for analysis in analyses:
            for element in analysis.all_elements:
                if element.type == "function" and element.is_public:
                    # Categorize by common patterns
                    name_lower = element.name.lower()
                    
                    if any(word in name_lower for word in ["get", "fetch", "retrieve"]):
                        patterns["data_retrieval"].append(f"{analysis.repo_name}::{element.name}")
                    elif any(word in name_lower for word in ["set", "update", "modify"]):
                        patterns["data_modification"].append(f"{analysis.repo_name}::{element.name}")
                    elif any(word in name_lower for word in ["create", "build", "generate"]):
                        patterns["creation_methods"].append(f"{analysis.repo_name}::{element.name}")
                    elif any(word in name_lower for word in ["validate", "check", "verify"]):
                        patterns["validation_methods"].append(f"{analysis.repo_name}::{element.name}")
                    elif any(word in name_lower for word in ["parse", "process", "transform"]):
                        patterns["processing_methods"].append(f"{analysis.repo_name}::{element.name}")
        
        # Keep only patterns with significant representation
        return {pattern: methods for pattern, methods in patterns.items() if len(methods) >= 1}
    
    def _calculate_complexity_statistics(self, analyses: List[RepositoryAnalysis]) -> Dict[str, float]:
        """Calculate complexity statistics across all repositories."""
        all_complexities = []
        
        for analysis in analyses:
            for element in analysis.all_elements:
                all_complexities.append(element.complexity)
        
        if not all_complexities:
            return {}
        
        return {
            "mean_complexity": float(np.mean(all_complexities)),
            "median_complexity": float(np.median(all_complexities)),
            "std_complexity": float(np.std(all_complexities)),
            "min_complexity": float(np.min(all_complexities)),
            "max_complexity": float(np.max(all_complexities)),
            "percentile_95": float(np.percentile(all_complexities, 95)),
        }
    
    def _calculate_repository_rankings(self, analyses: List[RepositoryAnalysis]) -> Dict[str, Any]:
        """Calculate repository rankings by various metrics."""
        rankings = {
            "by_size": [],
            "by_complexity": [],
            "by_element_count": [],
            "by_language_diversity": [],
        }
        
        for analysis in analyses:
            complexities = [e.complexity for e in analysis.all_elements]
            avg_complexity = np.mean(complexities) if complexities else 0
            
            repo_data = {
                "name": analysis.repo_name,
                "total_lines": analysis.total_lines,
                "total_files": analysis.total_files,
                "element_count": len(analysis.all_elements),
                "avg_complexity": float(avg_complexity),
                "language_count": len(analysis.languages),
                "languages": list(analysis.languages.keys()),
            }
            
            rankings["by_size"].append(repo_data)
            rankings["by_complexity"].append(repo_data)
            rankings["by_element_count"].append(repo_data)
            rankings["by_language_diversity"].append(repo_data)
        
        # Sort rankings
        rankings["by_size"].sort(key=lambda x: x["total_lines"], reverse=True)
        rankings["by_complexity"].sort(key=lambda x: x["avg_complexity"], reverse=True)
        rankings["by_element_count"].sort(key=lambda x: x["element_count"], reverse=True)
        rankings["by_language_diversity"].sort(key=lambda x: x["language_count"], reverse=True)
        
        return rankings
    
    def _analyze_cross_dependencies(self, analyses: List[RepositoryAnalysis]) -> Dict[str, List[str]]:
        """Analyze dependencies between repositories."""
        # This is a simplified implementation - in practice, this would be more sophisticated
        dependencies = defaultdict(list)
        
        repo_names = set(analysis.repo_name for analysis in analyses)
        
        for analysis in analyses:
            for file_analysis in analysis.file_analyses:
                for import_stmt in file_analysis.imports:
                    # Check if import might reference another Palantir repo
                    for repo_name in repo_names:
                        if repo_name.lower().replace('-', '_') in import_stmt.lower():
                            if repo_name != analysis.repo_name:
                                dependencies[analysis.repo_name].append(repo_name)
        
        return dict(dependencies)
    
    def _save_individual_analyses(self, analyses: List[RepositoryAnalysis]):
        """Save individual repository analyses."""
        individual_dir = self.output_dir / "individual_analyses"
        individual_dir.mkdir(exist_ok=True)
        
        for analysis in analyses:
            analysis_file = individual_dir / f"{analysis.repo_name}_analysis.json"
            
            # Convert analysis to dict for JSON serialization
            analysis_dict = {
                "repo_name": analysis.repo_name,
                "total_files": analysis.total_files,
                "total_lines": analysis.total_lines,
                "languages": analysis.languages,
                "element_count": len(analysis.all_elements),
                "functionality_summary": analysis.functionality_summary,
                "metrics": analysis.metrics,
                "elements_by_type": {
                    "functions": len([e for e in analysis.all_elements if e.type == "function"]),
                    "classes": len([e for e in analysis.all_elements if e.type == "class"]),
                    "methods": len([e for e in analysis.all_elements if e.type == "method"]),
                },
            }
            
            with open(analysis_file, 'w') as f:
                json.dump(analysis_dict, f, indent=2, default=str)
    
    def generate_comprehensive_visualizations(self, metrics: CrossRepoMetrics) -> Dict[str, str]:
        """
        Generate comprehensive visualizations for cross-repository analysis.
        
        Args:
            metrics: Cross-repository metrics
            
        Returns:
            Dict mapping visualization names to file paths
        """
        self.logger.info("Generating comprehensive visualizations...")
        
        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        visualizations = {}
        
        # 1. Language Distribution Chart
        viz_path = self._create_language_distribution_chart(metrics.language_distribution, viz_dir)
        visualizations["language_distribution"] = str(viz_path)
        
        # 2. Repository Size Distribution
        viz_path = self._create_repository_size_chart(metrics.repository_rankings, viz_dir)
        visualizations["repository_sizes"] = str(viz_path)
        
        # 3. Complexity Analysis Heatmap
        viz_path = self._create_complexity_heatmap(metrics.complexity_statistics, viz_dir)
        visualizations["complexity_heatmap"] = str(viz_path)
        
        # 4. Functionality Patterns Network
        viz_path = self._create_functionality_network(metrics.functionality_patterns, viz_dir)
        visualizations["functionality_network"] = str(viz_path)
        
        # 5. Cross-Repository Dependencies Graph
        viz_path = self._create_dependency_graph(metrics.cross_repo_dependencies, viz_dir)
        visualizations["dependency_graph"] = str(viz_path)
        
        # 6. Timeline of Repository Activity
        viz_path = self._create_activity_timeline(viz_dir)
        visualizations["activity_timeline"] = str(viz_path)
        
        self.logger.info(f"Generated {len(visualizations)} visualizations")
        return visualizations
    
    def generate_comprehensive_reports(self, metrics: CrossRepoMetrics) -> Dict[str, str]:
        """
        Generate comprehensive analysis reports in multiple formats.
        
        Args:
            metrics: Cross-repository metrics
            
        Returns:
            Dict mapping report names to file paths
        """
        self.logger.info("Generating comprehensive reports...")
        
        reports_dir = self.output_dir / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        reports = {}
        
        # 1. Executive Summary Report
        report_path = self._generate_executive_summary(metrics, reports_dir)
        reports["executive_summary"] = str(report_path)
        
        # 2. Detailed Technical Report
        report_path = self._generate_technical_report(metrics, reports_dir)
        reports["technical_report"] = str(report_path)
        
        # 3. Language-specific Analysis
        report_path = self._generate_language_analysis(metrics, reports_dir)
        reports["language_analysis"] = str(report_path)
        
        # 4. Repository Rankings Report
        report_path = self._generate_repository_rankings(metrics, reports_dir)
        reports["repository_rankings"] = str(report_path)
        
        # 5. CSV Data Exports
        csv_dir = reports_dir / "csv_exports"
        csv_dir.mkdir(exist_ok=True)
        csv_reports = self._generate_csv_exports(metrics, csv_dir)
        reports.update(csv_reports)
        
        self.logger.info(f"Generated {len(reports)} reports")
        return reports
    
    def extract_cross_repo_patterns(self, metrics: CrossRepoMetrics) -> Dict[str, Any]:
        """
        Extract patterns and insights across all repositories.
        
        Args:
            metrics: Cross-repository metrics
            
        Returns:
            Dict containing discovered patterns and insights
        """
        self.logger.info("Extracting cross-repository patterns...")
        
        patterns = {
            "common_naming_patterns": self._extract_naming_patterns(),
            "architectural_patterns": self._extract_architectural_patterns(),
            "code_quality_patterns": self._extract_quality_patterns(metrics),
            "dependency_patterns": self._extract_dependency_patterns(metrics),
            "language_usage_patterns": self._extract_language_patterns(metrics),
            "size_complexity_correlations": self._extract_size_complexity_correlations(metrics),
        }
        
        # Save patterns to file
        patterns_file = self.output_dir / "cross_repo_patterns.json"
        with open(patterns_file, 'w') as f:
            json.dump(patterns, f, indent=2, default=str)
        
        return patterns
    

    
    # Visualization methods
    def _create_language_distribution_chart(self, language_dist: Dict[str, int], viz_dir: Path) -> Path:
        """Create language distribution pie chart."""
        plt.figure(figsize=(12, 8))
        
        # Sort by count and take top 15 languages
        sorted_langs = sorted(language_dist.items(), key=lambda x: x[1], reverse=True)[:15]
        languages, counts = zip(*sorted_langs)
        
        plt.pie(counts, labels=languages, autopct='%1.1f%%', startangle=90)
        plt.title('Programming Language Distribution Across All Repositories', fontsize=16)
        plt.axis('equal')
        
        output_path = viz_dir / "language_distribution.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _create_repository_size_chart(self, repo_rankings: Dict[str, Any], viz_dir: Path) -> Path:
        """Create repository size distribution chart."""
        plt.figure(figsize=(15, 8))
        
        # Get top 20 repositories by size
        top_repos = repo_rankings["by_size"][:20]
        names = [repo["name"][:20] + "..." if len(repo["name"]) > 20 else repo["name"] for repo in top_repos]
        sizes = [repo["total_lines"] for repo in top_repos]
        
        plt.barh(range(len(names)), sizes, color='skyblue')
        plt.yticks(range(len(names)), names)
        plt.xlabel('Lines of Code')
        plt.title('Top 20 Repositories by Size (Lines of Code)', fontsize=16)
        plt.gca().invert_yaxis()
        
        output_path = viz_dir / "repository_sizes.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _create_complexity_heatmap(self, complexity_stats: Dict[str, float], viz_dir: Path) -> Path:
        """Create complexity statistics heatmap."""
        plt.figure(figsize=(10, 6))
        
        # Ensure all values are numeric and handle empty/invalid data
        if not complexity_stats:
            # Create a simple placeholder if no complexity stats available
            plt.text(0.5, 0.5, 'No complexity statistics available', 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
            plt.title('Code Complexity Statistics Across All Repositories', fontsize=16)
        else:
            # Filter and convert to numeric values only
            numeric_stats = {}
            for key, value in complexity_stats.items():
                try:
                    numeric_value = float(value) if value is not None else 0.0
                    numeric_stats[key] = numeric_value
                except (ValueError, TypeError):
                    numeric_stats[key] = 0.0
            
            if numeric_stats:
                # Create data for heatmap
                stats_df = pd.DataFrame([numeric_stats])
                
                sns.heatmap(stats_df, annot=True, fmt='.2f', cmap='YlOrRd', cbar_kws={'label': 'Complexity Score'})
                plt.title('Code Complexity Statistics Across All Repositories', fontsize=16)
                plt.ylabel('')
            else:
                plt.text(0.5, 0.5, 'No valid complexity statistics available', 
                        ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
                plt.title('Code Complexity Statistics Across All Repositories', fontsize=16)
        
        output_path = viz_dir / "complexity_heatmap.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _create_functionality_network(self, functionality_patterns: Dict[str, List[str]], viz_dir: Path) -> Path:
        """Create functionality patterns network diagram."""
        plt.figure(figsize=(14, 10))
        
        # Create a simple visualization of functionality patterns
        y_pos = np.arange(len(functionality_patterns))
        pattern_counts = [len(methods) for methods in functionality_patterns.values()]
        
        plt.barh(y_pos, pattern_counts, color='lightgreen')
        plt.yticks(y_pos, list(functionality_patterns.keys()))
        plt.xlabel('Number of Methods')
        plt.title('Common Functionality Patterns Across Repositories', fontsize=16)
        
        output_path = viz_dir / "functionality_network.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _create_dependency_graph(self, dependencies: Dict[str, List[str]], viz_dir: Path) -> Path:
        """Create cross-repository dependency graph."""
        plt.figure(figsize=(12, 8))
        
        # Simple dependency count visualization
        dep_counts = [(repo, len(deps)) for repo, deps in dependencies.items() if deps]
        dep_counts.sort(key=lambda x: x[1], reverse=True)
        
        if dep_counts:
            repos, counts = zip(*dep_counts[:15])  # Top 15 most connected repos
            
            plt.bar(range(len(repos)), counts, color='orange')
            plt.xticks(range(len(repos)), [repo[:15] + "..." if len(repo) > 15 else repo for repo in repos], rotation=45)
            plt.ylabel('Number of Dependencies')
            plt.title('Cross-Repository Dependencies', fontsize=16)
        else:
            plt.text(0.5, 0.5, 'No cross-repository dependencies detected', 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
            plt.title('Cross-Repository Dependencies', fontsize=16)
        
        output_path = viz_dir / "dependency_graph.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _create_activity_timeline(self, viz_dir: Path) -> Path:
        """Create repository activity timeline."""
        plt.figure(figsize=(12, 6))
        
        # This is a placeholder - in a real implementation, you'd analyze git history
        plt.text(0.5, 0.5, 'Repository Activity Timeline\n(Would require git history analysis)', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
        plt.title('Repository Activity Timeline', fontsize=16)
        
        output_path = viz_dir / "activity_timeline.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    # Report generation methods
    def _generate_executive_summary(self, metrics: CrossRepoMetrics, reports_dir: Path) -> Path:
        """Generate executive summary report."""
        report_path = reports_dir / "executive_summary.md"
        
        with open(report_path, 'w') as f:
            f.write("# Palantir Open Source Ecosystem - Executive Summary\n\n")
            f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Key Metrics\n\n")
            f.write(f"- **Total Repositories:** {int(metrics.total_repositories):,}\n")
            f.write(f"- **Total Files:** {int(metrics.total_files):,}\n")
            f.write(f"- **Total Lines of Code:** {int(metrics.total_lines_of_code):,}\n")
            f.write(f"- **Total Code Elements:** {int(metrics.total_code_elements):,}\n\n")
            
            f.write("## Language Distribution\n\n")
            sorted_langs = sorted(metrics.language_distribution.items(), key=lambda x: x[1], reverse=True)
            for i, (lang, count) in enumerate(sorted_langs[:10], 1):
                count = int(count) if isinstance(count, (int, float, str)) and str(count).replace('.', '').isdigit() else 0
                percentage = (count / sum(int(v) if isinstance(v, (int, float, str)) and str(v).replace('.', '').isdigit() else 0 for v in metrics.language_distribution.values())) * 100
                f.write(f"{i}. **{lang}:** {count:,} files ({percentage:.1f}%)\n")
            
            f.write("\n## Top Repositories by Size\n\n")
            for i, repo in enumerate(metrics.repository_rankings["by_size"][:10], 1):
                # Ensure values are integers for formatting
                total_lines = int(repo['total_lines']) if isinstance(repo['total_lines'], (int, float, str)) and str(repo['total_lines']).replace('.', '').isdigit() else 0
                total_files = int(repo['total_files']) if isinstance(repo['total_files'], (int, float, str)) and str(repo['total_files']).replace('.', '').isdigit() else 0
                f.write(f"{i}. **{repo['name']}:** {total_lines:,} lines, {total_files:,} files\n")
            
            f.write("\n## Complexity Statistics\n\n")
            if metrics.complexity_statistics:
                mean_complexity = float(metrics.complexity_statistics.get('mean_complexity', 0)) if metrics.complexity_statistics.get('mean_complexity') is not None else 0.0
                median_complexity = float(metrics.complexity_statistics.get('median_complexity', 0)) if metrics.complexity_statistics.get('median_complexity') is not None else 0.0
                max_complexity = float(metrics.complexity_statistics.get('max_complexity', 0)) if metrics.complexity_statistics.get('max_complexity') is not None else 0.0
                f.write(f"- **Average Complexity:** {mean_complexity:.2f}\n")
                f.write(f"- **Median Complexity:** {median_complexity:.2f}\n")
                f.write(f"- **Max Complexity:** {max_complexity:.2f}\n")
        
        return report_path
    
    def _generate_technical_report(self, metrics: CrossRepoMetrics, reports_dir: Path) -> Path:
        """Generate detailed technical report."""
        report_path = reports_dir / "technical_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Palantir Open Source Ecosystem - Technical Analysis\n\n")
            f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Repository Analysis Section
            f.write("## Repository Analysis\n\n")
            f.write("### Size Distribution\n\n")
            f.write(f"The ecosystem consists of {metrics.total_repositories:,} repositories with significant variation in size:\n\n")
            
            # Top 5 largest repositories
            top_repos = metrics.repository_rankings["by_size"][:5]
            f.write("**Largest Repositories:**\n")
            for i, repo in enumerate(top_repos, 1):
                total_lines = int(repo['total_lines']) if isinstance(repo['total_lines'], (int, float, str)) and str(repo['total_lines']).replace('.', '').isdigit() else 0
                total_files = int(repo['total_files']) if isinstance(repo['total_files'], (int, float, str)) and str(repo['total_files']).replace('.', '').isdigit() else 0
                f.write(f"{i}. **{repo['name']}**: {total_lines:,} lines across {total_files:,} files\n")
            
            # Language Analysis Section  
            f.write("\n## Language Analysis\n\n")
            f.write("The Palantir ecosystem demonstrates strong diversity in programming languages:\n\n")
            
            sorted_langs = sorted(metrics.language_distribution.items(), key=lambda x: x[1], reverse=True)
            for lang, count in sorted_langs:
                percentage = (count / sum(metrics.language_distribution.values())) * 100
                f.write(f"- **{lang.capitalize()}**: {count:,} files ({percentage:.1f}%)\n")
            
            # Functionality Patterns Section
            f.write("\n## Functionality Patterns\n\n")
            if metrics.functionality_patterns:
                f.write("Common functionality patterns identified across repositories:\n\n")
                for pattern, methods in metrics.functionality_patterns.items():
                    f.write(f"- **{pattern.replace('_', ' ').title()}**: {len(methods):,} instances\n")
            else:
                f.write("No common functionality patterns detected across repositories.\n\n")
            
            # Code Quality Metrics Section
            f.write("\n## Code Quality Metrics\n\n")
            if metrics.complexity_statistics:
                f.write("Code complexity analysis reveals:\n\n")
                stats = metrics.complexity_statistics
                f.write(f"- **Average Complexity**: {stats.get('mean_complexity', 0):.2f}\n")
                f.write(f"- **Median Complexity**: {stats.get('median_complexity', 0):.2f}\n")
                f.write(f"- **Standard Deviation**: {stats.get('std_complexity', 0):.2f}\n")
                f.write(f"- **Maximum Complexity**: {stats.get('max_complexity', 0):.2f}\n")
                f.write(f"- **95th Percentile**: {stats.get('percentile_95', 0):.2f}\n\n")
                
                # Quality assessment
                mean_complexity = stats.get('mean_complexity', 0)
                if mean_complexity < 2:
                    f.write("The low average complexity indicates generally well-structured, maintainable code.\n")
                elif mean_complexity < 5:
                    f.write("The moderate complexity suggests room for refactoring in some areas.\n")
                else:
                    f.write("The high complexity indicates potential maintenance challenges.\n")
            
            # Cross-Repository Dependencies Section
            f.write("\n## Cross-Repository Dependencies\n\n")
            if metrics.cross_repo_dependencies:
                dep_count = sum(len(deps) for deps in metrics.cross_repo_dependencies.values())
                f.write(f"Identified {dep_count:,} cross-repository dependencies among {len(metrics.cross_repo_dependencies):,} repositories.\n\n")
                
                # Most connected repositories
                most_connected = sorted(
                    [(repo, len(deps)) for repo, deps in metrics.cross_repo_dependencies.items()],
                    key=lambda x: x[1], reverse=True
                )[:5]
                
                if most_connected:
                    f.write("**Most Connected Repositories:**\n")
                    for repo, dep_count in most_connected:
                        f.write(f"- **{repo}**: {dep_count} dependencies\n")
            else:
                f.write("No significant cross-repository dependencies detected.\n")
        
        return report_path
    
    def _generate_language_analysis(self, metrics: CrossRepoMetrics, reports_dir: Path) -> Path:
        """Generate language-specific analysis report."""
        report_path = reports_dir / "language_analysis.md"
        
        with open(report_path, 'w') as f:
            f.write("# Language-Specific Analysis\n\n")
            f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Programming Language Ecosystem\n\n")
            f.write(f"The Palantir open source ecosystem spans {len(metrics.language_distribution)} primary programming languages ")
            f.write(f"across {metrics.total_repositories:,} repositories.\n\n")
            
            # Detailed language breakdown
            sorted_langs = sorted(metrics.language_distribution.items(), key=lambda x: x[1], reverse=True)
            total_files = sum(metrics.language_distribution.values())
            
            f.write("## Language Distribution Analysis\n\n")
            for i, (lang, count) in enumerate(sorted_langs, 1):
                percentage = (count / total_files) * 100
                f.write(f"### {i}. {lang.capitalize()}\n")
                f.write(f"- **Files**: {count:,} ({percentage:.1f}% of total)\n")
                
                # Language-specific insights
                if lang == "java":
                    f.write("- **Characteristics**: Enterprise-grade applications, strong typing, extensive ecosystem\n")
                    f.write("- **Use cases**: Backend services, data processing, enterprise applications\n")
                elif lang == "typescript":
                    f.write("- **Characteristics**: Type-safe JavaScript, modern web development\n")
                    f.write("- **Use cases**: Frontend applications, Node.js services, tooling\n")
                elif lang == "javascript":
                    f.write("- **Characteristics**: Dynamic scripting, web development, flexible syntax\n")
                    f.write("- **Use cases**: Frontend development, build tools, configuration\n")
                elif lang == "go":
                    f.write("- **Characteristics**: Systems programming, concurrency, simplicity\n")
                    f.write("- **Use cases**: CLI tools, microservices, infrastructure\n")
                elif lang == "python":
                    f.write("- **Characteristics**: Readable syntax, extensive libraries, versatile\n")
                    f.write("- **Use cases**: Data analysis, automation, scripting, tooling\n")
                
                f.write("\n")
            
            # Language trends and insights
            f.write("## Language Strategy Insights\n\n")
            if sorted_langs[0][0] == "java":
                f.write("- **Java dominance** indicates a strong focus on enterprise-grade, scalable backend systems\n")
            if "typescript" in dict(sorted_langs):
                f.write("- **TypeScript presence** shows commitment to type-safe, maintainable frontend development\n")
            if "go" in dict(sorted_langs):
                f.write("- **Go adoption** reflects emphasis on performant, concurrent systems and tooling\n")
            if "python" in dict(sorted_langs):
                f.write("- **Python usage** suggests focus on data processing, automation, and developer tooling\n")
        
        return report_path
    
    def _generate_repository_rankings(self, metrics: CrossRepoMetrics, reports_dir: Path) -> Path:
        """Generate repository rankings report."""
        report_path = reports_dir / "repository_rankings.md"
        
        with open(report_path, 'w') as f:
            f.write("# Repository Rankings\n\n")
            f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Rankings by size
            f.write("## Repositories by Size (Lines of Code)\n\n")
            f.write("| Rank | Repository | Lines of Code | Files | Elements |\n")
            f.write("|------|------------|---------------|-------|----------|\n")
            
            for i, repo in enumerate(metrics.repository_rankings["by_size"][:20], 1):
                total_lines = int(repo['total_lines']) if isinstance(repo['total_lines'], (int, float, str)) and str(repo['total_lines']).replace('.', '').isdigit() else 0
                total_files = int(repo['total_files']) if isinstance(repo['total_files'], (int, float, str)) and str(repo['total_files']).replace('.', '').isdigit() else 0
                element_count = int(repo['element_count']) if isinstance(repo['element_count'], (int, float, str)) and str(repo['element_count']).replace('.', '').isdigit() else 0
                f.write(f"| {i} | {repo['name']} | {total_lines:,} | {total_files:,} | {element_count:,} |\n")
            
            # Rankings by complexity
            f.write("\n## Repositories by Average Complexity\n\n")
            f.write("| Rank | Repository | Avg Complexity | Elements | Languages |\n")
            f.write("|------|------------|----------------|----------|----------|\n")
            
            # Filter out repos with 0 complexity and sort by complexity
            complex_repos = [repo for repo in metrics.repository_rankings["by_complexity"] 
                           if repo.get('avg_complexity', 0) > 0][:20]
            
            for i, repo in enumerate(complex_repos, 1):
                avg_complexity = float(repo.get('avg_complexity', 0))
                element_count = int(repo['element_count']) if isinstance(repo['element_count'], (int, float, str)) and str(repo['element_count']).replace('.', '').isdigit() else 0
                language_count = int(repo['language_count']) if isinstance(repo['language_count'], (int, float, str)) and str(repo['language_count']).replace('.', '').isdigit() else 0
                f.write(f"| {i} | {repo['name']} | {avg_complexity:.2f} | {element_count:,} | {language_count} |\n")
            
            # Rankings by language diversity
            f.write("\n## Repositories by Language Diversity\n\n")
            f.write("| Rank | Repository | Languages | Primary Languages |\n")
            f.write("|------|------------|-----------|------------------|\n")
            
            for i, repo in enumerate(metrics.repository_rankings["by_language_diversity"][:20], 1):
                language_count = int(repo['language_count']) if isinstance(repo['language_count'], (int, float, str)) and str(repo['language_count']).replace('.', '').isdigit() else 0
                languages = repo.get('languages', [])
                if isinstance(languages, list):
                    lang_str = ", ".join(languages[:3])  # Show first 3 languages
                    if len(languages) > 3:
                        lang_str += "..."
                else:
                    lang_str = str(languages) if languages else ""
                f.write(f"| {i} | {repo['name']} | {language_count} | {lang_str} |\n")
        
        return report_path
    
    def _generate_csv_exports(self, metrics: CrossRepoMetrics, csv_dir: Path) -> Dict[str, str]:
        """Generate CSV exports of analysis data."""
        csv_files = {}
        
        # Repository summary CSV
        repo_summary_path = csv_dir / "repository_summary.csv"
        repo_data = []
        for repo in metrics.repository_rankings["by_size"]:
            # Safely convert values to appropriate types
            languages_list = repo.get("languages", [])
            if isinstance(languages_list, str):
                languages_str = languages_list
            elif isinstance(languages_list, list):
                languages_str = ";".join(str(lang) for lang in languages_list)
            else:
                languages_str = ""
                
            repo_data.append({
                "repository": str(repo.get("name", "")),
                "total_lines": int(repo.get("total_lines", 0)) if str(repo.get("total_lines", "")).replace('.', '').isdigit() else 0,
                "total_files": int(repo.get("total_files", 0)) if str(repo.get("total_files", "")).replace('.', '').isdigit() else 0,
                "element_count": int(repo.get("element_count", 0)) if str(repo.get("element_count", "")).replace('.', '').isdigit() else 0,
                "avg_complexity": float(repo.get("avg_complexity", 0)) if str(repo.get("avg_complexity", "")).replace('.', '', 1).replace('-', '', 1).isdigit() else 0.0,
                "language_count": int(repo.get("language_count", 0)) if str(repo.get("language_count", "")).replace('.', '').isdigit() else 0,
                "languages": languages_str
            })
        
        pd.DataFrame(repo_data).to_csv(repo_summary_path, index=False)
        csv_files["repository_summary"] = str(repo_summary_path)
        
        # Language distribution CSV
        lang_dist_path = csv_dir / "language_distribution.csv"
        lang_data = [{"language": lang, "file_count": count} 
                    for lang, count in metrics.language_distribution.items()]
        pd.DataFrame(lang_data).to_csv(lang_dist_path, index=False)
        csv_files["language_distribution"] = str(lang_dist_path)
        
        return csv_files
    
    # Pattern extraction methods
    def _extract_naming_patterns(self) -> Dict[str, Any]:
        """Extract common naming patterns across repositories."""
        # Implementation for naming pattern analysis
        return {"patterns": [], "analysis_method": "naming_pattern_extraction"}
    
    def _extract_architectural_patterns(self) -> Dict[str, Any]:
        """Extract architectural patterns across repositories."""
        # Implementation for architectural pattern analysis
        return {"patterns": [], "analysis_method": "architectural_pattern_extraction"}
    
    def _extract_quality_patterns(self, metrics: CrossRepoMetrics) -> Dict[str, Any]:
        """Extract code quality patterns."""
        return {"quality_metrics": metrics.complexity_statistics}
    
    def _extract_dependency_patterns(self, metrics: CrossRepoMetrics) -> Dict[str, Any]:
        """Extract dependency patterns."""
        return {"dependencies": metrics.cross_repo_dependencies}
    
    def _extract_language_patterns(self, metrics: CrossRepoMetrics) -> Dict[str, Any]:
        """Extract language usage patterns."""
        return {"language_usage": metrics.language_distribution}
    
    def _extract_size_complexity_correlations(self, metrics: CrossRepoMetrics) -> Dict[str, Any]:
        """Extract correlations between repository size and complexity."""
        # Calculate correlations
        return {"correlations": {}, "analysis_method": "size_complexity_correlation"} 