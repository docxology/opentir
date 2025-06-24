#!/usr/bin/env python3
"""
Comprehensive Statistical Analysis of OpenTIR Analysis Data
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any
import warnings
from scipy import stats
from scipy.stats import f_oneway
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import silhouette_score

warnings.filterwarnings('ignore')

class ComprehensiveAnalyzer:
    def __init__(self):
        self.analysis_dir = Path("multi_repo_analysis")
        self.output_dir = Path("comprehensive_analysis_results")
        self.output_dir.mkdir(exist_ok=True)
        
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        (self.output_dir / "data_exports").mkdir(exist_ok=True)
        
        plt.style.use('seaborn-v0_8')
        
    def load_analysis_data(self) -> pd.DataFrame:
        print("🔍 Loading analysis data...")
        
        individual_analyses_dir = self.analysis_dir / "individual_analyses"
        if not individual_analyses_dir.exists():
            raise FileNotFoundError("Analysis data not found.")
        
        all_data = []
        
        for analysis_file in individual_analyses_dir.glob("*_analysis.json"):
            try:
                with open(analysis_file, 'r') as f:
                    data = json.load(f)
                
                repo_data = {
                    'repository': data.get('repo_name', ''),
                    'total_files': data.get('total_files', 0),
                    'total_lines': data.get('total_lines', 0),
                    'element_count': data.get('element_count', 0),
                    'complexity_score': data.get('complexity_score', 1.0)
                }
                
                languages = data.get('languages', {})
                repo_data['primary_language'] = max(languages.keys(), key=languages.get) if languages else 'unknown'
                
                func_summary = data.get('functionality_summary', {})
                by_type = func_summary.get('by_type', {})
                repo_data['functions'] = by_type.get('function', 0)
                repo_data['classes'] = by_type.get('class', 0)
                
                repo_data['lines_per_file'] = repo_data['total_lines'] / repo_data['total_files'] if repo_data['total_files'] > 0 else 0
                repo_data['elements_per_file'] = repo_data['element_count'] / repo_data['total_files'] if repo_data['total_files'] > 0 else 0
                repo_data['function_class_ratio'] = repo_data['functions'] / max(repo_data['classes'], 1)
                
                if repo_data['total_lines'] > 100000:
                    repo_data['size_category'] = 'large'
                elif repo_data['total_lines'] > 10000:
                    repo_data['size_category'] = 'medium'
                else:
                    repo_data['size_category'] = 'small'
                
                all_data.append(repo_data)
                
            except Exception as e:
                print(f"⚠️ Error processing {analysis_file}: {e}")
                continue
        
        df = pd.DataFrame(all_data)
        print(f"✅ Loaded data for {len(df)} repositories")
        return df
    
    def perform_anova_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        print("📊 Performing ANOVA analysis...")
        
        results = {}
        language_groups = df.groupby('primary_language')
        metrics = ['total_lines', 'complexity_score', 'lines_per_file', 'elements_per_file']
        
        for metric in metrics:
            if metric not in df.columns:
                continue
                
            groups = [group[metric].dropna() for name, group in language_groups if len(group) >= 3]
            
            if len(groups) >= 2 and all(len(group) > 0 for group in groups):
                try:
                    f_stat, p_value = f_oneway(*groups)
                    
                    results[metric] = {
                        'f_statistic': f_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'group_means': {name: group[metric].mean() for name, group in language_groups if len(group) >= 3}
                    }
                except Exception as e:
                    print(f"⚠️ ANOVA error for {metric}: {e}")
        
        return results
    
    def perform_pca_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        print("🔬 Performing PCA analysis...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        pca_data = df[numeric_cols].fillna(0)
        pca_data = pca_data.loc[:, pca_data.var() > 0]
        
        if pca_data.shape[1] < 2:
            return {'error': 'Insufficient numeric data for PCA'}
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(pca_data)
        
        pca = PCA()
        pca_result = pca.fit_transform(scaled_data)
        
        return {
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'feature_names': pca_data.columns.tolist(),
            'pca_scores': pca_result.tolist()
        }
    
    def perform_clustering_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        print("🎯 Performing clustering analysis...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cluster_data = df[numeric_cols].fillna(0)
        cluster_data = cluster_data.loc[:, cluster_data.var() > 0]
        
        if cluster_data.shape[1] < 2:
            return {'error': 'Insufficient data for clustering'}
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_data)
        
        k_range = range(2, min(11, len(df) // 2))
        silhouette_scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(scaled_data)
            score = silhouette_score(scaled_data, kmeans.labels_)
            silhouette_scores.append(score)
        
        optimal_k = k_range[np.argmax(silhouette_scores)]
        
        final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = final_kmeans.fit_predict(scaled_data)
        
        return {
            'optimal_k': optimal_k,
            'silhouette_scores': silhouette_scores,
            'cluster_labels': cluster_labels.tolist()
        }
    
    def create_visualizations(self, df: pd.DataFrame, results: Dict[str, Any]):
        print("🎨 Creating visualizations...")
        
        viz_dir = self.output_dir / "visualizations"
        
        # Language distribution
        plt.figure(figsize=(12, 8))
        lang_counts = df['primary_language'].value_counts()
        plt.pie(lang_counts.values, labels=lang_counts.index, autopct='%1.1f%%')
        plt.title('Repository Distribution by Programming Language')
        plt.savefig(viz_dir / 'language_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Size vs Complexity
        plt.figure(figsize=(10, 6))
        plt.scatter(df['total_lines'], df['complexity_score'], alpha=0.6)
        plt.xlabel('Total Lines of Code')
        plt.ylabel('Complexity Score')
        plt.title('Repository Size vs Complexity')
        plt.xscale('log')
        plt.savefig(viz_dir / 'size_vs_complexity.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # ANOVA results
        if 'anova' in results and results['anova']:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('ANOVA Analysis Results')
            
            metrics = list(results['anova'].keys())[:4]
            for i, metric in enumerate(metrics):
                ax = axes[i//2, i%2]
                result = results['anova'][metric]
                
                group_means = result['group_means']
                languages = list(group_means.keys())
                means = list(group_means.values())
                
                bars = ax.bar(languages, means)
                ax.set_title(f'{metric}\nF={result["f_statistic"]:.2f}, p={result["p_value"]:.4f}')
                ax.set_ylabel('Mean Value')
                ax.tick_params(axis='x', rotation=45)
                
                color = 'red' if result['significant'] else 'blue'
                for bar in bars:
                    bar.set_color(color)
            
            plt.tight_layout()
            plt.savefig(viz_dir / 'anova_results.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print("✅ Visualizations created")
    
    def run_complete_analysis(self):
        print("🚀 Starting Comprehensive Statistical Analysis")
        print("=" * 60)
        
        try:
            df = self.load_analysis_data()
            
            all_results = {}
            all_results['anova'] = self.perform_anova_analysis(df)
            all_results['pca'] = self.perform_pca_analysis(df)
            all_results['clustering'] = self.perform_clustering_analysis(df)
            
            self.create_visualizations(df, all_results)
            
            # Save results
            results_path = self.output_dir / "data_exports" / "comprehensive_results.json"
            with open(results_path, 'w') as f:
                json.dump(all_results, f, indent=2, default=lambda x: str(x))
            
            # Print summary
            print("=" * 60)
            print("🎉 Analysis Complete!")
            print(f"📁 Results: {self.output_dir}")
            print(f"📊 Dataset: {len(df)} repositories")
            print(f"🔤 Languages: {df['primary_language'].nunique()}")
            print(f"📈 Total LOC: {df['total_lines'].sum():,}")
            print(f"⚙️  Total Elements: {df['element_count'].sum():,}")
            
            # ANOVA summary
            if all_results['anova']:
                significant_tests = sum(1 for r in all_results['anova'].values() if r['significant'])
                print(f"📊 ANOVA: {significant_tests}/{len(all_results['anova'])} significant differences")
            
            # PCA summary
            if 'error' not in all_results['pca']:
                pc1_var = all_results['pca']['explained_variance_ratio'][0]
                print(f"🔬 PCA: PC1 explains {pc1_var:.1%} of variance")
            
            # Clustering summary
            if 'error' not in all_results['clustering']:
                optimal_k = all_results['clustering']['optimal_k']
                print(f"🎯 Clustering: {optimal_k} optimal clusters identified")
            
            return all_results
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return None

if __name__ == "__main__":
    analyzer = ComprehensiveAnalyzer()
    results = analyzer.run_complete_analysis()
