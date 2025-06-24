#!/usr/bin/env python3
"""
Advanced Statistical Analysis and Visualization Example for Opentir

This example demonstrates comprehensive statistical analysis and visualization
of code repositories including:
- Language distribution analysis
- Code complexity metrics
- Text analysis (words per line, characters per word, etc.)
- ANOVA tests on various metrics
- Advanced visualizations (heatmaps, network graphs, etc.)
"""

import asyncio
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict, Counter
import re
import string
from scipy import stats
from scipy.stats import f_oneway
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from main import OpentirOrchestrator
from multi_repo_analyzer import MultiRepositoryAnalyzer
from code_analyzer import CodeAnalyzer
from utils import Logger

warnings.filterwarnings('ignore')


class AdvancedCodeStatistics:
    """Advanced statistical analysis of code repositories."""
    
    def __init__(self, output_dir: Path = None):
        """Initialize advanced statistics analyzer."""
        self.output_dir = output_dir or Path("advanced_analysis_results")
        self.output_dir.mkdir(exist_ok=True)
        self.logger = Logger("advanced_stats")
        
        # Setup visualization directories
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        (self.output_dir / "statistics").mkdir(exist_ok=True)
        (self.output_dir / "data_exports").mkdir(exist_ok=True)
        
        # Set plot style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def analyze_text_statistics(self, file_content: str) -> Dict[str, float]:
        """
        Analyze text statistics of code content.
        
        Returns:
            Dictionary with text metrics including:
            - Average words per line
            - Average characters per word
            - Word to punctuation ratio
            - Number to punctuation ratio
            - Code complexity indicators
        """
        if not file_content.strip():
            return self._empty_text_stats()
        
        lines = file_content.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        
        if not non_empty_lines:
            return self._empty_text_stats()
        
        # Word analysis
        total_words = 0
        total_chars = 0
        total_punctuation = 0
        total_numbers = 0
        
        for line in non_empty_lines:
            # Extract words (alphanumeric sequences)
            words = re.findall(r'\b\w+\b', line)
            total_words += len(words)
            
            # Count characters in words
            for word in words:
                total_chars += len(word)
            
            # Count punctuation
            total_punctuation += sum(1 for char in line if char in string.punctuation)
            
            # Count numbers
            total_numbers += len(re.findall(r'\d+', line))
        
        # Calculate metrics
        avg_words_per_line = total_words / len(non_empty_lines) if non_empty_lines else 0
        avg_chars_per_word = total_chars / total_words if total_words > 0 else 0
        word_to_punct_ratio = total_words / total_punctuation if total_punctuation > 0 else float('inf')
        number_to_punct_ratio = total_numbers / total_punctuation if total_punctuation > 0 else float('inf')
        
        # Code-specific metrics
        comment_lines = sum(1 for line in non_empty_lines if line.strip().startswith(('#', '//', '/*', '*')))
        comment_ratio = comment_lines / len(non_empty_lines) if non_empty_lines else 0
        
        # Complexity indicators
        bracket_density = sum(line.count('(') + line.count('[') + line.count('{') for line in non_empty_lines) / len(non_empty_lines)
        
        return {
            'avg_words_per_line': avg_words_per_line,
            'avg_chars_per_word': avg_chars_per_word,
            'word_to_punct_ratio': min(word_to_punct_ratio, 1000),  # Cap extreme values
            'number_to_punct_ratio': min(number_to_punct_ratio, 1000),
            'comment_ratio': comment_ratio,
            'bracket_density': bracket_density,
            'total_words': total_words,
            'total_lines': len(non_empty_lines)
        }
    
    def _empty_text_stats(self) -> Dict[str, float]:
        """Return empty text statistics."""
        return {
            'avg_words_per_line': 0.0,
            'avg_chars_per_word': 0.0,
            'word_to_punct_ratio': 0.0,
            'number_to_punct_ratio': 0.0,
            'comment_ratio': 0.0,
            'bracket_density': 0.0,
            'total_words': 0,
            'total_lines': 0
        }
    
    def collect_comprehensive_metrics(self, repos_dir: Path) -> pd.DataFrame:
        """
        Collect comprehensive metrics from all repositories.
        
        Returns:
            DataFrame with detailed metrics for each repository and file.
        """
        self.logger.info("Collecting comprehensive metrics from repositories...")
        
        metrics_data = []
        analyzer = CodeAnalyzer()
        
        repos_path = repos_dir / "all_repos"
        if not repos_path.exists():
            self.logger.error("Repository directory not found")
            return pd.DataFrame()
        
        for repo_dir in repos_path.iterdir():
            if not repo_dir.is_dir() or repo_dir.name.startswith('.'):
                continue
                
            self.logger.info(f"Analyzing repository: {repo_dir.name}")
            
            try:
                # Get repository analysis
                repo_analysis = analyzer.analyze_repository(repo_dir)
                
                # Process each file analysis
                for file_analysis in repo_analysis.file_analyses:
                    # Read file content for text analysis
                    try:
                        with open(file_analysis.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                    except Exception:
                        content = ""
                    
                    # Get text statistics
                    text_stats = self.analyze_text_statistics(content)
                    
                    # Compile metrics
                    metric_row = {
                        'repository': repo_dir.name,
                        'file_path': file_analysis.file_path,
                        'language': file_analysis.language,
                        'lines_of_code': file_analysis.lines_of_code,
                        'complexity_score': file_analysis.complexity_score,
                        'num_elements': len(file_analysis.elements),
                        'num_imports': len(file_analysis.imports),
                        **text_stats
                    }
                    
                    # Add element type counts
                    element_types = Counter(e.type for e in file_analysis.elements)
                    metric_row.update({
                        'functions': element_types.get('function', 0),
                        'classes': element_types.get('class', 0),
                        'methods': element_types.get('method', 0)
                    })
                    
                    metrics_data.append(metric_row)
                    
            except Exception as e:
                self.logger.warning(f"Error analyzing {repo_dir.name}: {e}")
                continue
        
        df = pd.DataFrame(metrics_data)
        
        # Save raw data
        df.to_csv(self.output_dir / "data_exports" / "comprehensive_metrics.csv", index=False)
        
        self.logger.info(f"Collected metrics for {len(df)} files across {df['repository'].nunique()} repositories")
        return df
    
    def perform_statistical_tests(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive statistical tests on the repository data.
        
        Args:
            df: DataFrame with repository metrics
            
        Returns:
            Dictionary containing statistical test results
        """
        self.logger.info("Performing comprehensive statistical tests...")
        
        results = {
            'anova_tests': {},
            'pca_analysis': {},
            'clustering_analysis': {},
            'correlation_analysis': {},
            'normality_tests': {},
            'language_comparison': {},
            'complexity_analysis': {},
            'natural_language_pca': {}
        }
        
        # ANOVA Tests - Analyze differences across categorical variables
        results['anova_tests'] = self._perform_anova_analysis(df)
        
        # PCA Analysis - Dimensionality reduction and pattern discovery
        results['pca_analysis'] = self._perform_pca_analysis(df)
        
        # Natural Language PCA - Analyze text patterns in code
        results['natural_language_pca'] = self._perform_natural_language_pca(df)
        
        # Clustering Analysis - Group repositories by similarity
        results['clustering_analysis'] = self._perform_clustering_analysis(df)
        
        # Correlation Analysis - Identify relationships between variables
        results['correlation_analysis'] = self._perform_correlation_analysis(df)
        
        # Normality Tests - Validate statistical assumptions
        results['normality_tests'] = self._perform_normality_tests(df)
        
        # Language-specific Analysis - Compare programming languages
        results['language_comparison'] = self._perform_language_comparison(df)
        
        # Complexity Analysis - Advanced complexity metrics
        results['complexity_analysis'] = self._perform_complexity_analysis(df)
        
        # Save detailed results
        results_file = self.output_dir / "statistics" / "statistical_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=self._make_json_serializable)
        
        self.logger.info("Statistical tests completed")
        return results
    
    def _perform_anova_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform ANOVA tests across categorical variables."""
        anova_results = {}
        
        # Prepare categorical groupings
        categorical_vars = ['language', 'repository']
        continuous_vars = [
            'lines_of_code', 'complexity_score', 'num_elements',
            'avg_words_per_line', 'avg_chars_per_word', 'comment_ratio',
            'bracket_density', 'functions', 'classes', 'methods'
        ]
        
        for cat_var in categorical_vars:
            if cat_var in df.columns:
                anova_results[cat_var] = {}
                
                # Get unique groups, limiting to prevent excessive computation
                groups = df[cat_var].value_counts()
                if len(groups) > 20:  # Limit to top 20 groups
                    top_groups = groups.head(20).index
                    df_filtered = df[df[cat_var].isin(top_groups)]
                else:
                    df_filtered = df
                
                for cont_var in continuous_vars:
                    if cont_var in df_filtered.columns:
                        try:
                            # Prepare groups for ANOVA
                            group_data = []
                            group_names = []
                            
                            for group_name in df_filtered[cat_var].unique():
                                group_values = df_filtered[df_filtered[cat_var] == group_name][cont_var]
                                # Filter out non-finite values
                                group_values = group_values[np.isfinite(group_values)]
                                
                                if len(group_values) >= 3:  # Minimum sample size
                                    group_data.append(group_values.values)
                                    group_names.append(group_name)
                            
                            if len(group_data) >= 2:  # Need at least 2 groups
                                # Perform ANOVA
                                f_statistic, p_value = f_oneway(*group_data)
                                
                                # Calculate effect size (eta-squared)
                                total_mean = df_filtered[cont_var].mean()
                                ss_between = sum(len(group) * (np.mean(group) - total_mean)**2 
                                               for group in group_data)
                                ss_total = sum((df_filtered[cont_var] - total_mean)**2)
                                eta_squared = ss_between / ss_total if ss_total > 0 else 0
                                
                                anova_results[cat_var][cont_var] = {
                                    'f_statistic': float(f_statistic),
                                    'p_value': float(p_value),
                                    'eta_squared': float(eta_squared),
                                    'significant': p_value < 0.05,
                                    'effect_size': self._interpret_effect_size(eta_squared),
                                    'num_groups': len(group_data),
                                    'group_names': group_names[:10]  # Limit for readability
                                }
                                
                        except Exception as e:
                            self.logger.warning(f"ANOVA failed for {cat_var} vs {cont_var}: {e}")
        
        return anova_results
    
    def _perform_pca_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform Principal Component Analysis for dimensionality reduction."""
        pca_results = {}
        
        # Select numeric columns for PCA
        numeric_cols = [
            'lines_of_code', 'complexity_score', 'num_elements', 'num_imports',
            'avg_words_per_line', 'avg_chars_per_word', 'comment_ratio',
            'bracket_density', 'functions', 'classes', 'methods'
        ]
        
        # Filter available columns
        available_cols = [col for col in numeric_cols if col in df.columns]
        
        if len(available_cols) >= 3:
            try:
                # Prepare data
                pca_data = df[available_cols].copy()
                
                # Handle missing values and infinite values
                pca_data = pca_data.replace([np.inf, -np.inf], np.nan)
                pca_data = pca_data.fillna(pca_data.median())
                
                # Standardize the data
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                pca_data_scaled = scaler.fit_transform(pca_data)
                
                # Perform PCA
                n_components = min(len(available_cols), 10)  # Limit components
                pca = PCA(n_components=n_components)
                pca_transformed = pca.fit_transform(pca_data_scaled)
                
                # Store results
                pca_results = {
                    'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                    'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_).tolist(),
                    'components': pca.components_.tolist(),
                    'feature_names': available_cols,
                    'n_components': n_components,
                    'total_variance_explained': float(np.sum(pca.explained_variance_ratio_))
                }
                
                # Interpret principal components
                pca_results['component_interpretation'] = {}
                for i in range(min(3, n_components)):  # Interpret first 3 components
                    component = pca.components_[i]
                    
                    # Find features with highest absolute loadings
                    feature_loadings = [(available_cols[j], float(component[j])) 
                                      for j in range(len(available_cols))]
                    feature_loadings.sort(key=lambda x: abs(x[1]), reverse=True)
                    
                    pca_results['component_interpretation'][f'PC{i+1}'] = {
                        'variance_explained': float(pca.explained_variance_ratio_[i]),
                        'top_features': feature_loadings[:5],
                        'interpretation': self._interpret_pca_component(feature_loadings[:3])
                    }
                
            except Exception as e:
                self.logger.warning(f"PCA analysis failed: {e}")
                pca_results = {'error': str(e)}
        
        return pca_results
    
    def _perform_natural_language_pca(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform PCA analysis on natural language features in code."""
        nl_pca_results = {}
        
        # Natural language features
        nl_features = [
            'avg_words_per_line', 'avg_chars_per_word', 'word_to_punct_ratio',
            'number_to_punct_ratio', 'comment_ratio', 'total_words'
        ]
        
        # Filter available features
        available_features = [feat for feat in nl_features if feat in df.columns]
        
        if len(available_features) >= 3:
            try:
                # Prepare natural language data
                nl_data = df[available_features].copy()
                
                # Handle missing and infinite values
                nl_data = nl_data.replace([np.inf, -np.inf], np.nan)
                nl_data = nl_data.fillna(nl_data.median())
                
                # Cap extreme values to prevent skewing
                for col in nl_data.columns:
                    q99 = nl_data[col].quantile(0.99)
                    nl_data[col] = nl_data[col].clip(upper=q99)
                
                # Standardize the data
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                nl_data_scaled = scaler.fit_transform(nl_data)
                
                # Perform PCA
                n_components = min(len(available_features), 5)
                pca = PCA(n_components=n_components)
                pca_transformed = pca.fit_transform(nl_data_scaled)
                
                # Store results
                nl_pca_results = {
                    'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                    'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_).tolist(),
                    'components': pca.components_.tolist(),
                    'feature_names': available_features,
                    'n_components': n_components,
                    'total_variance_explained': float(np.sum(pca.explained_variance_ratio_))
                }
                
                # Interpret natural language patterns
                nl_pca_results['language_patterns'] = {}
                for i in range(min(2, n_components)):
                    component = pca.components_[i]
                    
                    feature_loadings = [(available_features[j], float(component[j])) 
                                      for j in range(len(available_features))]
                    feature_loadings.sort(key=lambda x: abs(x[1]), reverse=True)
                    
                    nl_pca_results['language_patterns'][f'Pattern_{i+1}'] = {
                        'variance_explained': float(pca.explained_variance_ratio_[i]),
                        'feature_loadings': feature_loadings,
                        'interpretation': self._interpret_language_pattern(feature_loadings)
                    }
                
            except Exception as e:
                self.logger.warning(f"Natural language PCA failed: {e}")
                nl_pca_results = {'error': str(e)}
        
        return nl_pca_results
    
    def _perform_clustering_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform clustering analysis to group similar repositories."""
        clustering_results = {}
        
        # Select features for clustering
        clustering_features = [
            'lines_of_code', 'complexity_score', 'num_elements',
            'avg_words_per_line', 'comment_ratio', 'functions', 'classes'
        ]
        
        available_features = [feat for feat in clustering_features if feat in df.columns]
        
        if len(available_features) >= 3 and len(df) >= 10:
            try:
                # Prepare clustering data
                cluster_data = df[available_features].copy()
                cluster_data = cluster_data.replace([np.inf, -np.inf], np.nan)
                cluster_data = cluster_data.fillna(cluster_data.median())
                
                # Standardize data
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                cluster_data_scaled = scaler.fit_transform(cluster_data)
                
                # Determine optimal number of clusters using elbow method
                max_clusters = min(10, len(df) // 3)  # Reasonable upper bound
                inertias = []
                k_range = range(2, max_clusters + 1)
                
                for k in k_range:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    kmeans.fit(cluster_data_scaled)
                    inertias.append(kmeans.inertia_)
                
                # Find elbow point
                optimal_k = self._find_elbow_point(list(k_range), inertias)
                
                # Perform final clustering
                kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(cluster_data_scaled)
                
                # Analyze clusters
                clustering_results = {
                    'optimal_clusters': int(optimal_k),
                    'inertias': inertias,
                    'k_range': list(k_range),
                    'cluster_centers': kmeans.cluster_centers_.tolist(),
                    'feature_names': available_features,
                    'cluster_analysis': {}
                }
                
                # Analyze each cluster
                for cluster_id in range(optimal_k):
                    cluster_mask = cluster_labels == cluster_id
                    cluster_repos = df[cluster_mask]
                    
                    clustering_results['cluster_analysis'][f'Cluster_{cluster_id}'] = {
                        'size': int(np.sum(cluster_mask)),
                        'percentage': float(np.sum(cluster_mask) / len(df) * 100),
                        'top_repositories': cluster_repos.nlargest(5, 'lines_of_code')['repository'].tolist() if 'repository' in df.columns else [],
                        'language_distribution': cluster_repos['language'].value_counts().to_dict() if 'language' in df.columns else {},
                        'mean_characteristics': cluster_repos[available_features].mean().to_dict()
                    }
                
            except Exception as e:
                self.logger.warning(f"Clustering analysis failed: {e}")
                clustering_results = {'error': str(e)}
        
        return clustering_results
    
    def _perform_correlation_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform correlation analysis between variables."""
        correlation_results = {}
        
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            try:
                # Calculate correlation matrix
                corr_data = df[numeric_cols].copy()
                corr_data = corr_data.replace([np.inf, -np.inf], np.nan)
                
                # Pearson correlation
                pearson_corr = corr_data.corr(method='pearson')
                
                # Spearman correlation (rank-based, more robust)
                spearman_corr = corr_data.corr(method='spearman')
                
                # Find strong correlations
                strong_correlations = []
                for i in range(len(pearson_corr.columns)):
                    for j in range(i+1, len(pearson_corr.columns)):
                        col1, col2 = pearson_corr.columns[i], pearson_corr.columns[j]
                        pearson_val = pearson_corr.iloc[i, j]
                        spearman_val = spearman_corr.iloc[i, j]
                        
                        if abs(pearson_val) > 0.5:  # Strong correlation threshold
                            strong_correlations.append({
                                'variable1': col1,
                                'variable2': col2,
                                'pearson_correlation': float(pearson_val),
                                'spearman_correlation': float(spearman_val),
                                'strength': 'strong' if abs(pearson_val) > 0.7 else 'moderate'
                            })
                
                correlation_results = {
                    'pearson_matrix': pearson_corr.to_dict(),
                    'spearman_matrix': spearman_corr.to_dict(),
                    'strong_correlations': strong_correlations,
                    'correlation_summary': {
                        'total_variables': len(numeric_cols),
                        'strong_correlations_count': len([c for c in strong_correlations if c['strength'] == 'strong']),
                        'moderate_correlations_count': len([c for c in strong_correlations if c['strength'] == 'moderate'])
                    }
                }
                
            except Exception as e:
                self.logger.warning(f"Correlation analysis failed: {e}")
                correlation_results = {'error': str(e)}
        
        return correlation_results
    
    def _perform_normality_tests(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform normality tests on continuous variables."""
        normality_results = {}
        
        # Select continuous variables
        continuous_vars = [
            'lines_of_code', 'complexity_score', 'num_elements',
            'avg_words_per_line', 'avg_chars_per_word', 'comment_ratio'
        ]
        
        available_vars = [var for var in continuous_vars if var in df.columns]
        
        for var in available_vars:
            try:
                data = df[var].dropna()
                data = data[np.isfinite(data)]  # Remove infinite values
                
                if len(data) >= 8:  # Minimum sample size for tests
                    # Shapiro-Wilk test (sample size <= 5000)
                    if len(data) <= 5000:
                        shapiro_stat, shapiro_p = stats.shapiro(data)
                    else:
                        shapiro_stat, shapiro_p = np.nan, np.nan
                    
                    # Anderson-Darling test
                    anderson_result = stats.anderson(data, dist='norm')
                    
                    # Jarque-Bera test
                    jb_stat, jb_p = stats.jarque_bera(data)
                    
                    normality_results[var] = {
                        'shapiro_wilk': {
                            'statistic': float(shapiro_stat) if not np.isnan(shapiro_stat) else None,
                            'p_value': float(shapiro_p) if not np.isnan(shapiro_p) else None,
                            'normal': bool(shapiro_p > 0.05) if not np.isnan(shapiro_p) else None
                        },
                        'anderson_darling': {
                            'statistic': float(anderson_result.statistic),
                            'critical_values': anderson_result.critical_values.tolist(),
                            'significance_levels': anderson_result.significance_level.tolist()
                        },
                        'jarque_bera': {
                            'statistic': float(jb_stat),
                            'p_value': float(jb_p),
                            'normal': bool(jb_p > 0.05)
                        },
                        'sample_size': len(data),
                        'mean': float(data.mean()),
                        'std': float(data.std()),
                        'skewness': float(stats.skew(data)),
                        'kurtosis': float(stats.kurtosis(data))
                    }
                    
            except Exception as e:
                self.logger.warning(f"Normality test failed for {var}: {e}")
        
        return normality_results
    
    def _perform_language_comparison(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compare metrics across programming languages."""
        language_results = {}
        
        if 'language' in df.columns:
            languages = df['language'].value_counts()
            
            # Focus on languages with sufficient sample size
            min_samples = 5
            major_languages = languages[languages >= min_samples].index.tolist()
            
            comparison_metrics = [
                'lines_of_code', 'complexity_score', 'comment_ratio',
                'avg_words_per_line', 'functions', 'classes'
            ]
            
            available_metrics = [m for m in comparison_metrics if m in df.columns]
            
            for metric in available_metrics:
                try:
                    language_results[metric] = {}
                    
                    # Descriptive statistics by language
                    for lang in major_languages:
                        lang_data = df[df['language'] == lang][metric]
                        lang_data = lang_data[np.isfinite(lang_data)]
                        
                        if len(lang_data) > 0:
                            language_results[metric][lang] = {
                                'count': len(lang_data),
                                'mean': float(lang_data.mean()),
                                'median': float(lang_data.median()),
                                'std': float(lang_data.std()),
                                'min': float(lang_data.min()),
                                'max': float(lang_data.max()),
                                'q25': float(lang_data.quantile(0.25)),
                                'q75': float(lang_data.quantile(0.75))
                            }
                    
                    # Pairwise comparisons between languages
                    if len(major_languages) >= 2:
                        language_results[metric]['pairwise_comparisons'] = {}
                        
                        for i, lang1 in enumerate(major_languages):
                            for lang2 in major_languages[i+1:]:
                                data1 = df[df['language'] == lang1][metric]
                                data2 = df[df['language'] == lang2][metric]
                                
                                data1 = data1[np.isfinite(data1)]
                                data2 = data2[np.isfinite(data2)]
                                
                                if len(data1) >= 3 and len(data2) >= 3:
                                    # Mann-Whitney U test (non-parametric)
                                    u_stat, u_p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                                    
                                    # Effect size (Cohen's d)
                                    pooled_std = np.sqrt(((len(data1)-1)*data1.std()**2 + (len(data2)-1)*data2.std()**2) / (len(data1)+len(data2)-2))
                                    cohens_d = (data1.mean() - data2.mean()) / pooled_std if pooled_std > 0 else 0
                                    
                                    comparison_key = f"{lang1}_vs_{lang2}"
                                    language_results[metric]['pairwise_comparisons'][comparison_key] = {
                                        'mann_whitney_u': float(u_stat),
                                        'p_value': float(u_p),
                                        'significant': bool(u_p < 0.05),
                                        'cohens_d': float(cohens_d),
                                        'effect_size': self._interpret_effect_size_cohens_d(cohens_d)
                                    }
                
                except Exception as e:
                    self.logger.warning(f"Language comparison failed for {metric}: {e}")
        
        return language_results
    
    def _perform_complexity_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform advanced complexity analysis."""
        complexity_results = {}
        
        if 'complexity_score' in df.columns:
            complexity_data = df['complexity_score'].dropna()
            complexity_data = complexity_data[np.isfinite(complexity_data)]
            
            if len(complexity_data) > 0:
                try:
                    # Complexity distribution analysis
                    complexity_results['distribution'] = {
                        'mean': float(complexity_data.mean()),
                        'median': float(complexity_data.median()),
                        'std': float(complexity_data.std()),
                        'min': float(complexity_data.min()),
                        'max': float(complexity_data.max()),
                        'skewness': float(stats.skew(complexity_data)),
                        'kurtosis': float(stats.kurtosis(complexity_data)),
                        'percentiles': {
                            'p10': float(complexity_data.quantile(0.1)),
                            'p25': float(complexity_data.quantile(0.25)),
                            'p50': float(complexity_data.quantile(0.5)),
                            'p75': float(complexity_data.quantile(0.75)),
                            'p90': float(complexity_data.quantile(0.9)),
                            'p95': float(complexity_data.quantile(0.95)),
                            'p99': float(complexity_data.quantile(0.99))
                        }
                    }
                    
                    # Complexity categories
                    complexity_results['categories'] = {
                        'low_complexity': int(np.sum(complexity_data <= 2)),
                        'moderate_complexity': int(np.sum((complexity_data > 2) & (complexity_data <= 10))),
                        'high_complexity': int(np.sum(complexity_data > 10)),
                        'very_high_complexity': int(np.sum(complexity_data > 50))
                    }
                    
                    # Complexity by repository size correlation
                    if 'lines_of_code' in df.columns:
                        size_data = df['lines_of_code'].dropna()
                        common_indices = complexity_data.index.intersection(size_data.index)
                        
                        if len(common_indices) > 5:
                            complexity_subset = complexity_data[common_indices]
                            size_subset = size_data[common_indices]
                            
                            correlation, p_value = stats.pearsonr(complexity_subset, size_subset)
                            
                            complexity_results['size_correlation'] = {
                                'correlation': float(correlation),
                                'p_value': float(p_value),
                                'significant': bool(p_value < 0.05)
                            }
                    
                except Exception as e:
                    self.logger.warning(f"Complexity analysis failed: {e}")
                    complexity_results = {'error': str(e)}
        
        return complexity_results
    
    def _interpret_effect_size(self, eta_squared: float) -> str:
        """Interpret eta-squared effect size."""
        if eta_squared < 0.01:
            return "negligible"
        elif eta_squared < 0.06:
            return "small"
        elif eta_squared < 0.14:
            return "medium"
        else:
            return "large"
    
    def _interpret_effect_size_cohens_d(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _interpret_pca_component(self, top_features: List[Tuple[str, float]]) -> str:
        """Interpret what a PCA component represents."""
        feature_names = [feat[0] for feat in top_features]
        
        if 'complexity_score' in feature_names and 'lines_of_code' in feature_names:
            return "Repository Size and Complexity"
        elif 'comment_ratio' in feature_names and 'avg_words_per_line' in feature_names:
            return "Documentation and Readability"
        elif 'functions' in feature_names and 'classes' in feature_names:
            return "Code Structure and Organization"
        else:
            return f"Mixed pattern involving {', '.join(feature_names[:2])}"
    
    def _interpret_language_pattern(self, feature_loadings: List[Tuple[str, float]]) -> str:
        """Interpret natural language patterns in code."""
        high_loading_features = [feat[0] for feat in feature_loadings if abs(feat[1]) > 0.5]
        
        if 'comment_ratio' in high_loading_features:
            return "Documentation-heavy coding style"
        elif 'avg_words_per_line' in high_loading_features:
            return "Verbose coding patterns"
        elif 'word_to_punct_ratio' in high_loading_features:
            return "Punctuation vs. word usage patterns"
        else:
            return "Mixed natural language characteristics"
    
    def _find_elbow_point(self, k_values: List[int], inertias: List[float]) -> int:
        """Find the elbow point in k-means clustering."""
        if len(k_values) < 3:
            return k_values[0]
        
        # Calculate the rate of change
        deltas = np.diff(inertias)
        delta_deltas = np.diff(deltas)
        
        # Find the point with maximum curvature (elbow)
        elbow_idx = np.argmax(delta_deltas) + 1  # +1 because of double diff
        
        # Ensure it's within bounds
        elbow_idx = max(0, min(elbow_idx, len(k_values) - 1))
        
        return k_values[elbow_idx]
    
    def _make_json_serializable(self, obj):
        """Convert numpy types to JSON serializable types."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def create_advanced_visualizations(self, df: pd.DataFrame, stats_results: Dict[str, Any]) -> Dict[str, str]:
        """
        Create comprehensive visualizations for statistical analysis.
        
        Args:
            df: DataFrame with repository metrics
            stats_results: Results from statistical tests
            
        Returns:
            Dictionary mapping visualization names to file paths
        """
        self.logger.info("Creating advanced visualizations...")
        
        viz_dir = self.output_dir / "visualizations"
        viz_files = {}
        
        # Basic visualizations
        viz_files['language_distribution'] = self._create_language_distribution_viz(df, viz_dir)
        viz_files['size_complexity'] = self._create_size_complexity_viz(df, viz_dir)
        viz_files['text_stats_heatmap'] = self._create_text_stats_heatmap(df, viz_dir)
        viz_files['correlation_matrix'] = self._create_correlation_matrix(df, viz_dir)
        viz_files['repository_radar'] = self._create_repository_radar_chart(df, viz_dir)
        viz_files['complexity_analysis'] = self._create_complexity_analysis(df, viz_dir)
        viz_files['interactive_dashboard'] = self._create_interactive_dashboard(df, viz_dir)
        
        # Advanced statistical visualizations
        viz_files['anova_results'] = self._create_anova_visualization(stats_results, viz_dir)
        viz_files['pca_analysis'] = self._create_pca_visualization(stats_results, viz_dir)
        viz_files['natural_language_pca'] = self._create_natural_language_pca_viz(stats_results, viz_dir)
        viz_files['clustering_analysis'] = self._create_clustering_visualization(stats_results, viz_dir)
        viz_files['language_comparison'] = self._create_language_comparison_viz(stats_results, viz_dir)
        viz_files['statistical_summary'] = self._create_statistical_summary_viz(stats_results, viz_dir)
        
        self.logger.info(f"Created {len(viz_files)} visualizations")
        return viz_files
    
    def _create_anova_visualization(self, stats_results: Dict[str, Any], viz_dir: Path) -> str:
        """Create visualizations for ANOVA results."""
        try:
            if 'anova_tests' not in stats_results:
                return ""
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('ANOVA Analysis Results', fontsize=16, fontweight='bold')
            
            anova_data = stats_results['anova_tests']
            
            # Plot 1: F-statistics heatmap
            if anova_data:
                categories = []
                metrics = []
                f_stats = []
                p_values = []
                
                for cat_var, cat_results in anova_data.items():
                    for metric, result in cat_results.items():
                        categories.append(cat_var)
                        metrics.append(metric)
                        f_stats.append(result.get('f_statistic', 0))
                        p_values.append(result.get('p_value', 1))
                
                if f_stats:
                    # F-statistics plot
                    ax1 = axes[0, 0]
                    scatter = ax1.scatter(range(len(f_stats)), f_stats, 
                                        c=[-np.log10(p) for p in p_values], 
                                        cmap='viridis', s=60, alpha=0.7)
                    ax1.set_xlabel('Test Index')
                    ax1.set_ylabel('F-statistic')
                    ax1.set_title('ANOVA F-statistics')
                    plt.colorbar(scatter, ax=ax1, label='-log10(p-value)')
                    
                    # Significance plot
                    ax2 = axes[0, 1]
                    significant = [1 if p < 0.05 else 0 for p in p_values]
                    ax2.bar(range(len(significant)), significant, alpha=0.7)
                    ax2.set_xlabel('Test Index')
                    ax2.set_ylabel('Significant (1=Yes, 0=No)')
                    ax2.set_title('Statistical Significance')
                    ax2.set_ylim(0, 1.2)
                    
                    # Effect sizes plot
                    ax3 = axes[1, 0]
                    effect_sizes = []
                    for cat_var, cat_results in anova_data.items():
                        for metric, result in cat_results.items():
                            effect_sizes.append(result.get('eta_squared', 0))
                    
                    if effect_sizes:
                        ax3.hist(effect_sizes, bins=20, alpha=0.7, edgecolor='black')
                        ax3.set_xlabel('Eta-squared (Effect Size)')
                        ax3.set_ylabel('Frequency')
                        ax3.set_title('Distribution of Effect Sizes')
                        ax3.axvline(0.01, color='red', linestyle='--', alpha=0.7, label='Small')
                        ax3.axvline(0.06, color='orange', linestyle='--', alpha=0.7, label='Medium')
                        ax3.axvline(0.14, color='green', linestyle='--', alpha=0.7, label='Large')
                        ax3.legend()
                    
                    # Summary statistics
                    ax4 = axes[1, 1]
                    summary_data = {
                        'Total Tests': len(f_stats),
                        'Significant': sum(significant),
                        'Large Effects': sum(1 for es in effect_sizes if es > 0.14),
                        'Medium Effects': sum(1 for es in effect_sizes if 0.06 <= es <= 0.14)
                    }
                    
                    bars = ax4.bar(summary_data.keys(), summary_data.values(), alpha=0.7)
                    ax4.set_title('ANOVA Summary Statistics')
                    ax4.set_ylabel('Count')
                    
                    # Add value labels on bars
                    for bar in bars:
                        height = bar.get_height()
                        ax4.text(bar.get_x() + bar.get_width()/2., height,
                               f'{int(height)}', ha='center', va='bottom')
            
            plt.tight_layout()
            anova_file = viz_dir / "anova_analysis.png"
            plt.savefig(anova_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(anova_file)
            
        except Exception as e:
            self.logger.warning(f"ANOVA visualization failed: {e}")
            return ""
    
    def _create_pca_visualization(self, stats_results: Dict[str, Any], viz_dir: Path) -> str:
        """Create PCA analysis visualizations."""
        try:
            if 'pca_analysis' not in stats_results or 'error' in stats_results['pca_analysis']:
                return ""
            
            pca_data = stats_results['pca_analysis']
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Principal Component Analysis', fontsize=16, fontweight='bold')
            
            # Explained variance plot
            ax1 = axes[0, 0]
            variance_ratio = pca_data['explained_variance_ratio']
            cumulative_variance = pca_data['cumulative_variance_ratio']
            
            x = range(1, len(variance_ratio) + 1)
            ax1.bar(x, variance_ratio, alpha=0.7, label='Individual')
            ax1.plot(x, cumulative_variance, 'ro-', label='Cumulative')
            ax1.set_xlabel('Principal Component')
            ax1.set_ylabel('Explained Variance Ratio')
            ax1.set_title('Explained Variance by Component')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Component loadings heatmap
            ax2 = axes[0, 1]
            if 'components' in pca_data and 'feature_names' in pca_data:
                components = np.array(pca_data['components'])
                feature_names = pca_data['feature_names']
                
                # Show first 3 components
                n_show = min(3, len(components))
                loadings_subset = components[:n_show]
                
                im = ax2.imshow(loadings_subset, cmap='RdBu_r', aspect='auto')
                ax2.set_xticks(range(len(feature_names)))
                ax2.set_xticklabels(feature_names, rotation=45, ha='right')
                ax2.set_yticks(range(n_show))
                ax2.set_yticklabels([f'PC{i+1}' for i in range(n_show)])
                ax2.set_title('Component Loadings')
                plt.colorbar(im, ax=ax2)
            
            # Component interpretation
            ax3 = axes[1, 0]
            if 'component_interpretation' in pca_data:
                interpretations = pca_data['component_interpretation']
                components = list(interpretations.keys())
                variances = [interpretations[comp]['variance_explained'] for comp in components]
                
                bars = ax3.bar(components, variances, alpha=0.7)
                ax3.set_title('Component Variance Explained')
                ax3.set_ylabel('Variance Explained')
                ax3.tick_params(axis='x', rotation=45)
                
                # Add percentage labels
                for bar, var in zip(bars, variances):
                    ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                           f'{var:.1%}', ha='center', va='bottom')
            
            # Feature importance plot
            ax4 = axes[1, 1]
            if 'components' in pca_data and 'feature_names' in pca_data:
                components = np.array(pca_data['components'])
                feature_names = pca_data['feature_names']
                
                # Calculate average absolute loadings across first 3 components
                n_comp = min(3, len(components))
                avg_loadings = np.mean(np.abs(components[:n_comp]), axis=0)
                
                # Sort features by importance
                sorted_indices = np.argsort(avg_loadings)[::-1]
                sorted_features = [feature_names[i] for i in sorted_indices]
                sorted_loadings = avg_loadings[sorted_indices]
                
                bars = ax4.barh(range(len(sorted_features)), sorted_loadings, alpha=0.7)
                ax4.set_yticks(range(len(sorted_features)))
                ax4.set_yticklabels(sorted_features)
                ax4.set_xlabel('Average Absolute Loading')
                ax4.set_title('Feature Importance (First 3 PCs)')
                
                # Add value labels
                for i, (bar, loading) in enumerate(zip(bars, sorted_loadings)):
                    ax4.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2.,
                           f'{loading:.3f}', va='center')
            
            plt.tight_layout()
            pca_file = viz_dir / "pca_analysis.png"
            plt.savefig(pca_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(pca_file)
            
        except Exception as e:
            self.logger.warning(f"PCA visualization failed: {e}")
            return ""
    
    def _create_natural_language_pca_viz(self, stats_results: Dict[str, Any], viz_dir: Path) -> str:
        """Create natural language PCA visualizations."""
        try:
            if 'natural_language_pca' not in stats_results or 'error' in stats_results['natural_language_pca']:
                return ""
            
            nl_pca_data = stats_results['natural_language_pca']
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Natural Language Patterns in Code - PCA Analysis', fontsize=16, fontweight='bold')
            
            # Explained variance
            ax1 = axes[0, 0]
            if 'explained_variance_ratio' in nl_pca_data:
                variance_ratio = nl_pca_data['explained_variance_ratio']
                cumulative_variance = nl_pca_data['cumulative_variance_ratio']
                
                x = range(1, len(variance_ratio) + 1)
                ax1.bar(x, variance_ratio, alpha=0.7, color='lightblue', label='Individual')
                ax1.plot(x, cumulative_variance, 'ro-', label='Cumulative')
                ax1.set_xlabel('Principal Component')
                ax1.set_ylabel('Explained Variance Ratio')
                ax1.set_title('NL Variance Explained')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            
            # Language patterns interpretation
            ax2 = axes[0, 1]
            if 'language_patterns' in nl_pca_data:
                patterns = nl_pca_data['language_patterns']
                pattern_names = list(patterns.keys())
                variances = [patterns[name]['variance_explained'] for name in pattern_names]
                
                bars = ax2.bar(pattern_names, variances, alpha=0.7, color='lightgreen')
                ax2.set_title('Language Pattern Importance')
                ax2.set_ylabel('Variance Explained')
                
                for bar, var in zip(bars, variances):
                    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                           f'{var:.1%}', ha='center', va='bottom')
            
            # Feature loadings for language patterns
            ax3 = axes[1, 0]
            if 'components' in nl_pca_data and 'feature_names' in nl_pca_data:
                components = np.array(nl_pca_data['components'])
                feature_names = nl_pca_data['feature_names']
                
                # Show loadings for first 2 components
                n_show = min(2, len(components))
                im = ax3.imshow(components[:n_show], cmap='RdBu_r', aspect='auto')
                ax3.set_xticks(range(len(feature_names)))
                ax3.set_xticklabels(feature_names, rotation=45, ha='right')
                ax3.set_yticks(range(n_show))
                ax3.set_yticklabels([f'Pattern {i+1}' for i in range(n_show)])
                ax3.set_title('Language Pattern Loadings')
                plt.colorbar(im, ax=ax3)
            
            # Pattern interpretations text
            ax4 = axes[1, 1]
            ax4.axis('off')
            if 'language_patterns' in nl_pca_data:
                text_lines = ['Natural Language Pattern Interpretations:\\n']
                for pattern_name, pattern_data in nl_pca_data['language_patterns'].items():
                    interpretation = pattern_data.get('interpretation', 'Unknown pattern')
                    variance = pattern_data.get('variance_explained', 0)
                    text_lines.append(f'{pattern_name}: {interpretation}')
                    text_lines.append(f'  Variance: {variance:.1%}\\n')
                
                ax4.text(0.05, 0.95, '\\n'.join(text_lines), transform=ax4.transAxes,
                        verticalalignment='top', fontsize=10, 
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            
            plt.tight_layout()
            nl_pca_file = viz_dir / "natural_language_pca.png"
            plt.savefig(nl_pca_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(nl_pca_file)
            
        except Exception as e:
            self.logger.warning(f"Natural language PCA visualization failed: {e}")
            return ""
    
    def _create_clustering_visualization(self, stats_results: Dict[str, Any], viz_dir: Path) -> str:
        """Create clustering analysis visualizations."""
        try:
            if 'clustering_analysis' not in stats_results or 'error' in stats_results['clustering_analysis']:
                return ""
            
            cluster_data = stats_results['clustering_analysis']
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Repository Clustering Analysis', fontsize=16, fontweight='bold')
            
            # Elbow curve for optimal k
            ax1 = axes[0, 0]
            if 'k_range' in cluster_data and 'inertias' in cluster_data:
                k_range = cluster_data['k_range']
                inertias = cluster_data['inertias']
                optimal_k = cluster_data.get('optimal_clusters', k_range[0])
                
                ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
                ax1.axvline(optimal_k, color='red', linestyle='--', alpha=0.7, 
                           label=f'Optimal k={optimal_k}')
                ax1.set_xlabel('Number of Clusters (k)')
                ax1.set_ylabel('Inertia')
                ax1.set_title('Elbow Method for Optimal k')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            
            # Cluster sizes
            ax2 = axes[0, 1]
            if 'cluster_analysis' in cluster_data:
                cluster_analysis = cluster_data['cluster_analysis']
                cluster_names = list(cluster_analysis.keys())
                cluster_sizes = [cluster_analysis[name]['size'] for name in cluster_names]
                cluster_percentages = [cluster_analysis[name]['percentage'] for name in cluster_names]
                
                bars = ax2.bar(cluster_names, cluster_sizes, alpha=0.7)
                ax2.set_title('Cluster Sizes')
                ax2.set_ylabel('Number of Repositories')
                ax2.tick_params(axis='x', rotation=45)
                
                # Add percentage labels
                for bar, pct in zip(bars, cluster_percentages):
                    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                           f'{pct:.1f}%', ha='center', va='bottom')
            
            # Cluster characteristics heatmap
            ax3 = axes[1, 0]
            if 'cluster_analysis' in cluster_data and 'feature_names' in cluster_data:
                cluster_analysis = cluster_data['cluster_analysis']
                feature_names = cluster_data['feature_names']
                
                # Build characteristics matrix
                characteristics_matrix = []
                cluster_labels = []
                
                for cluster_name, cluster_info in cluster_analysis.items():
                    if 'mean_characteristics' in cluster_info:
                        characteristics = cluster_info['mean_characteristics']
                        row = [characteristics.get(feat, 0) for feat in feature_names]
                        characteristics_matrix.append(row)
                        cluster_labels.append(cluster_name)
                
                if characteristics_matrix:
                    # Normalize for better visualization
                    characteristics_matrix = np.array(characteristics_matrix)
                    # Z-score normalization across features
                    normalized_matrix = (characteristics_matrix - np.mean(characteristics_matrix, axis=0)) / (np.std(characteristics_matrix, axis=0) + 1e-8)
                    
                    im = ax3.imshow(normalized_matrix, cmap='RdYlBu_r', aspect='auto')
                    ax3.set_xticks(range(len(feature_names)))
                    ax3.set_xticklabels(feature_names, rotation=45, ha='right')
                    ax3.set_yticks(range(len(cluster_labels)))
                    ax3.set_yticklabels(cluster_labels)
                    ax3.set_title('Cluster Characteristics (Normalized)')
                    plt.colorbar(im, ax=ax3)
            
            # Language distribution by cluster
            ax4 = axes[1, 1]
            if 'cluster_analysis' in cluster_data:
                cluster_analysis = cluster_data['cluster_analysis']
                
                # Collect language data
                all_languages = set()
                for cluster_info in cluster_analysis.values():
                    if 'language_distribution' in cluster_info:
                        all_languages.update(cluster_info['language_distribution'].keys())
                
                if all_languages:
                    all_languages = list(all_languages)[:5]  # Top 5 languages
                    cluster_names = list(cluster_analysis.keys())
                    
                    # Create stacked bar chart
                    bottom = np.zeros(len(cluster_names))
                    colors = plt.cm.Set3(np.linspace(0, 1, len(all_languages)))
                    
                    for i, lang in enumerate(all_languages):
                        values = []
                        for cluster_name in cluster_names:
                            lang_dist = cluster_analysis[cluster_name].get('language_distribution', {})
                            values.append(lang_dist.get(lang, 0))
                        
                        ax4.bar(cluster_names, values, bottom=bottom, 
                               label=lang, color=colors[i], alpha=0.8)
                        bottom += values
                    
                    ax4.set_title('Language Distribution by Cluster')
                    ax4.set_ylabel('Number of Files')
                    ax4.tick_params(axis='x', rotation=45)
                    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            cluster_file = viz_dir / "clustering_analysis.png"
            plt.savefig(cluster_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(cluster_file)
            
        except Exception as e:
            self.logger.warning(f"Clustering visualization failed: {e}")
            return ""
    
    def _create_language_comparison_viz(self, stats_results: Dict[str, Any], viz_dir: Path) -> str:
        """Create language comparison visualizations."""
        try:
            if 'language_comparison' not in stats_results:
                return ""
            
            lang_data = stats_results['language_comparison']
            
            # Create a larger figure with subplots
            n_metrics = len(lang_data)
            if n_metrics == 0:
                return ""
            
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle('Programming Language Comparison Analysis', fontsize=16, fontweight='bold')
            axes = axes.flatten()
            
            plot_idx = 0
            
            for metric_name, metric_data in list(lang_data.items())[:6]:  # Show first 6 metrics
                if plot_idx >= 6:
                    break
                    
                ax = axes[plot_idx]
                
                # Extract language statistics
                languages = []
                means = []
                stds = []
                
                for lang, stats in metric_data.items():
                    if isinstance(stats, dict) and 'mean' in stats:
                        languages.append(lang)
                        means.append(stats['mean'])
                        stds.append(stats.get('std', 0))
                
                if languages and means:
                    # Create box plot style visualization
                    positions = range(len(languages))
                    bars = ax.bar(positions, means, yerr=stds, alpha=0.7, capsize=5)
                    
                    ax.set_xticks(positions)
                    ax.set_xticklabels(languages, rotation=45, ha='right')
                    ax.set_ylabel(metric_name.replace('_', ' ').title())
                    ax.set_title(f'{metric_name.replace("_", " ").title()} by Language')
                    ax.grid(True, alpha=0.3)
                    
                    # Add value labels on bars
                    for bar, mean in zip(bars, means):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{mean:.2f}', ha='center', va='bottom', fontsize=8)
                
                plot_idx += 1
            
            # Hide unused subplots
            for i in range(plot_idx, 6):
                axes[i].axis('off')
            
            plt.tight_layout()
            lang_comp_file = viz_dir / "language_comparison.png"
            plt.savefig(lang_comp_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(lang_comp_file)
            
        except Exception as e:
            self.logger.warning(f"Language comparison visualization failed: {e}")
            return ""
    
    def _create_statistical_summary_viz(self, stats_results: Dict[str, Any], viz_dir: Path) -> str:
        """Create a comprehensive statistical summary visualization."""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle('Statistical Analysis Summary Dashboard', fontsize=16, fontweight='bold')
            
            # Summary statistics
            ax1 = axes[0, 0]
            summary_stats = {
                'ANOVA Tests': len(stats_results.get('anova_tests', {})),
                'PCA Components': len(stats_results.get('pca_analysis', {}).get('explained_variance_ratio', [])),
                'Clusters Found': stats_results.get('clustering_analysis', {}).get('optimal_clusters', 0),
                'Languages Compared': len(stats_results.get('language_comparison', {}))
            }
            
            bars = ax1.bar(summary_stats.keys(), summary_stats.values(), alpha=0.7)
            ax1.set_title('Analysis Summary')
            ax1.set_ylabel('Count')
            ax1.tick_params(axis='x', rotation=45)
            
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom')
            
            # Correlation strength distribution
            ax2 = axes[0, 1]
            if 'correlation_analysis' in stats_results and 'strong_correlations' in stats_results['correlation_analysis']:
                correlations = stats_results['correlation_analysis']['strong_correlations']
                corr_values = [abs(c['pearson_correlation']) for c in correlations]
                
                if corr_values:
                    ax2.hist(corr_values, bins=15, alpha=0.7, edgecolor='black')
                    ax2.set_xlabel('Correlation Strength')
                    ax2.set_ylabel('Frequency')
                    ax2.set_title('Distribution of Strong Correlations')
                    ax2.axvline(0.7, color='red', linestyle='--', alpha=0.7, label='Strong threshold')
                    ax2.legend()
            
            # Normality test results
            ax3 = axes[0, 2]
            if 'normality_tests' in stats_results:
                normality_data = stats_results['normality_tests']
                variables = list(normality_data.keys())
                normal_counts = []
                
                for var in variables:
                    jb_result = normality_data[var].get('jarque_bera', {})
                    is_normal = jb_result.get('normal', False)
                    normal_counts.append(1 if is_normal else 0)
                
                if variables:
                    bars = ax3.bar(variables, normal_counts, alpha=0.7)
                    ax3.set_title('Normality Test Results')
                    ax3.set_ylabel('Normal Distribution (1=Yes, 0=No)')
                    ax3.tick_params(axis='x', rotation=45)
            
            # Effect sizes from ANOVA
            ax4 = axes[1, 0]
            if 'anova_tests' in stats_results:
                effect_sizes = []
                for cat_var, cat_results in stats_results['anova_tests'].items():
                    for metric, result in cat_results.items():
                        effect_sizes.append(result.get('eta_squared', 0))
                
                if effect_sizes:
                    ax4.hist(effect_sizes, bins=15, alpha=0.7, edgecolor='black')
                    ax4.set_xlabel('Effect Size (Eta-squared)')
                    ax4.set_ylabel('Frequency')
                    ax4.set_title('ANOVA Effect Sizes Distribution')
                    ax4.axvline(0.01, color='yellow', linestyle='--', label='Small')
                    ax4.axvline(0.06, color='orange', linestyle='--', label='Medium')
                    ax4.axvline(0.14, color='red', linestyle='--', label='Large')
                    ax4.legend()
            
            # PCA variance explained
            ax5 = axes[1, 1]
            if 'pca_analysis' in stats_results and 'explained_variance_ratio' in stats_results['pca_analysis']:
                variance_ratios = stats_results['pca_analysis']['explained_variance_ratio']
                cumsum_variance = np.cumsum(variance_ratios)
                
                x = range(1, len(variance_ratios) + 1)
                ax5.bar(x, variance_ratios, alpha=0.7, label='Individual')
                ax5.plot(x, cumsum_variance, 'ro-', label='Cumulative')
                ax5.set_xlabel('Principal Component')
                ax5.set_ylabel('Variance Explained')
                ax5.set_title('PCA Variance Explained')
                ax5.legend()
            
            # Statistical test p-values
            ax6 = axes[1, 2]
            p_values = []
            
            # Collect p-values from various tests
            if 'anova_tests' in stats_results:
                for cat_var, cat_results in stats_results['anova_tests'].items():
                    for metric, result in cat_results.items():
                        p_values.append(result.get('p_value', 1))
            
            if 'language_comparison' in stats_results:
                for metric, lang_data in stats_results['language_comparison'].items():
                    if 'pairwise_comparisons' in lang_data:
                        for comp, comp_data in lang_data['pairwise_comparisons'].items():
                            p_values.append(comp_data.get('p_value', 1))
            
            if p_values:
                ax6.hist(p_values, bins=20, alpha=0.7, edgecolor='black')
                ax6.set_xlabel('P-value')
                ax6.set_ylabel('Frequency')
                ax6.set_title('Distribution of P-values')
                ax6.axvline(0.05, color='red', linestyle='--', alpha=0.7, label='α = 0.05')
                ax6.legend()
            
            plt.tight_layout()
            summary_file = viz_dir / "statistical_summary.png"
            plt.savefig(summary_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(summary_file)
            
        except Exception as e:
            self.logger.warning(f"Statistical summary visualization failed: {e}")
            return ""


async def run_advanced_analysis():
    """Run the complete advanced statistical analysis."""
    print("🚀 Starting Advanced Statistical Analysis of Palantir OSS Ecosystem")
    print("=" * 80)
    
    # Initialize components
    analyzer = AdvancedCodeStatistics()
    repos_dir = Path("repos")
    
    # Check if repositories exist
    if not (repos_dir / "all_repos").exists():
        print("❌ No repositories found. Please run the main analysis first:")
        print("   python main.py")
        return
    
    try:
        # Step 1: Collect comprehensive metrics
        print("\n📊 Step 1: Collecting comprehensive metrics...")
        df = analyzer.collect_comprehensive_metrics(repos_dir)
        
        if df.empty:
            print("❌ No data collected. Please check repository structure.")
            return
        
        print(f"✅ Collected metrics for {len(df)} files")
        
        # Step 2: Perform statistical tests
        print("\n📈 Step 2: Performing statistical tests...")
        stats_results = analyzer.perform_statistical_tests(df)
        print("✅ Statistical analysis completed")
        
        # Step 3: Create visualizations
        print("\n🎨 Step 3: Creating advanced visualizations...")
        viz_files = analyzer.create_advanced_visualizations(df, stats_results)
        print(f"✅ Generated {len(viz_files)} visualizations")
        
        # Step 4: Generate comprehensive report
        print("\n📋 Step 4: Generating comprehensive report...")
        report_path = analyzer.generate_comprehensive_report(df, stats_results, viz_files)
        print(f"✅ Report saved to: {report_path}")
        
        # Summary
        print("\n🎉 Advanced Analysis Complete!")
        print(f"📁 Results directory: {analyzer.output_dir}")
        print(f"📊 Files analyzed: {len(df):,}")
        print(f"🏢 Repositories: {df['repository'].nunique()}")
        print(f"💻 Languages: {df['language'].nunique()}")
        
        print("\n📈 Key Statistics:")
        print(f"   • Average complexity: {df['complexity_score'].mean():.2f}")
        print(f"   • Average words per line: {df['avg_words_per_line'].mean():.2f}")
        print(f"   • Average comment ratio: {df['comment_ratio'].mean():.1%}")
        
        print(f"\n🎯 Next Steps:")
        print(f"   • View visualizations: open {analyzer.output_dir}/visualizations/")
        print(f"   • Read report: open {report_path}")
        print(f"   • Explore data: {analyzer.output_dir}/data_exports/comprehensive_metrics.csv")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        analyzer.logger.error(f"Analysis failed: {e}")


if __name__ == "__main__":
    asyncio.run(run_advanced_analysis()) 