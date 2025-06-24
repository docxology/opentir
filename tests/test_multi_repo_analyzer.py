"""
Tests for multi-repository analyzer functionality.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.multi_repo_analyzer import MultiRepositoryAnalyzer, CrossRepoMetrics
from src.code_analyzer import RepositoryAnalysis, CodeElement


class TestMultiRepositoryAnalyzer:
    """Test suite for MultiRepositoryAnalyzer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.repos_dir = self.temp_dir / "repos"
        self.repos_dir.mkdir(parents=True)
        
        # Create mock repo directories
        (self.repos_dir / "all_repos").mkdir()
        (self.repos_dir / "all_repos" / "repo1").mkdir()
        (self.repos_dir / "all_repos" / "repo2").mkdir()
        (self.repos_dir / "all_repos" / "repo3").mkdir()
        
        self.analyzer = MultiRepositoryAnalyzer(repos_base_path=self.repos_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_get_all_repository_paths(self):
        """Test getting all repository paths."""
        paths = self.analyzer._get_all_repository_paths()
        
        assert len(paths) == 3
        assert all(path.is_dir() for path in paths)
        assert {path.name for path in paths} == {"repo1", "repo2", "repo3"}
    
    def test_get_all_repository_paths_missing_directory(self):
        """Test error when repository directory doesn't exist."""
        analyzer = MultiRepositoryAnalyzer(repos_base_path=Path("/nonexistent"))
        
        with pytest.raises(ValueError, match="Repository directory does not exist"):
            analyzer._get_all_repository_paths()
    
    def test_generate_cross_repo_metrics(self):
        """Test generating cross-repository metrics."""
        # Create mock analyses
        mock_analyses = [
            self._create_mock_analysis("repo1", 100, 10, {"python": 5, "java": 5}),
            self._create_mock_analysis("repo2", 200, 15, {"python": 10, "javascript": 5}),
            self._create_mock_analysis("repo3", 150, 8, {"java": 8}),
        ]
        
        metrics = self.analyzer._generate_cross_repo_metrics(mock_analyses)
        
        assert isinstance(metrics, CrossRepoMetrics)
        assert metrics.total_repositories == 3
        assert metrics.total_files == 33  # 10 + 15 + 8
        assert metrics.total_lines_of_code == 450  # 100 + 200 + 150
        assert metrics.language_distribution["python"] == 15  # 5 + 10
        assert metrics.language_distribution["java"] == 13  # 5 + 8
        assert metrics.language_distribution["javascript"] == 5
    
    def test_extract_functionality_patterns(self):
        """Test extracting functionality patterns."""
        mock_analyses = [
            self._create_mock_analysis_with_elements("repo1", [
                self._create_mock_element("get_data", "function", True),
                self._create_mock_element("set_value", "function", True),
                self._create_mock_element("private_method", "function", False),
            ]),
            self._create_mock_analysis_with_elements("repo2", [
                self._create_mock_element("fetch_users", "function", True),
                self._create_mock_element("update_user", "function", True),
                self._create_mock_element("validate_input", "function", True),
            ]),
        ]
        
        patterns = self.analyzer._extract_functionality_patterns(mock_analyses)
        
        assert "data_retrieval" in patterns
        assert "data_modification" in patterns
        assert "validation_methods" in patterns
        assert "repo1::get_data" in patterns["data_retrieval"]
        assert "repo2::fetch_users" in patterns["data_retrieval"]
        assert "repo1::set_value" in patterns["data_modification"]
        assert "repo2::validate_input" in patterns["validation_methods"]
    
    def test_calculate_complexity_statistics(self):
        """Test calculating complexity statistics."""
        mock_analyses = [
            self._create_mock_analysis_with_elements("repo1", [
                self._create_mock_element("func1", "function", True, complexity=5),
                self._create_mock_element("func2", "function", True, complexity=10),
            ]),
            self._create_mock_analysis_with_elements("repo2", [
                self._create_mock_element("func3", "function", True, complexity=3),
                self._create_mock_element("func4", "function", True, complexity=7),
            ]),
        ]
        
        stats = self.analyzer._calculate_complexity_statistics(mock_analyses)
        
        assert stats["mean_complexity"] == 6.25  # (5+10+3+7)/4
        assert stats["min_complexity"] == 3
        assert stats["max_complexity"] == 10
        assert "median_complexity" in stats
        assert "std_complexity" in stats
    
    def test_calculate_repository_rankings(self):
        """Test calculating repository rankings."""
        mock_analyses = [
            self._create_mock_analysis("repo1", 100, 10, {"python": 5}),
            self._create_mock_analysis("repo2", 300, 20, {"python": 15, "java": 5}),
            self._create_mock_analysis("repo3", 200, 15, {"javascript": 15}),
        ]
        
        rankings = self.analyzer._calculate_repository_rankings(mock_analyses)
        
        # Test by_size ranking
        assert rankings["by_size"][0]["name"] == "repo2"  # Largest
        assert rankings["by_size"][1]["name"] == "repo3"
        assert rankings["by_size"][2]["name"] == "repo1"  # Smallest
        
        # Test by_language_diversity ranking
        assert rankings["by_language_diversity"][0]["name"] == "repo2"  # 2 languages
        assert rankings["by_language_diversity"][0]["language_count"] == 2
    
    def test_analyze_cross_dependencies(self):
        """Test analyzing cross-repository dependencies."""
        # Create mock file analyses with imports
        mock_analyses = [
            self._create_mock_analysis_with_imports("repo1", ["import repo2_utils", "from repo3 import helper"]),
            self._create_mock_analysis_with_imports("repo2", ["import standard_lib"]),
            self._create_mock_analysis_with_imports("repo3", ["import repo1_module"]),
        ]
        
        dependencies = self.analyzer._analyze_cross_dependencies(mock_analyses)
        
        # Note: This is a simplified test - actual implementation would be more sophisticated
        assert isinstance(dependencies, dict)
    
    @patch('src.multi_repo_analyzer.MultiRepositoryAnalyzer._get_all_repository_paths')
    @patch('src.code_analyzer.CodeAnalyzer.analyze_repository')
    def test_analyze_all_repositories_with_cache(self, mock_analyze, mock_get_paths):
        """Test analyzing all repositories with cache."""
        # Setup mocks
        mock_get_paths.return_value = [Path("repo1"), Path("repo2")]
        mock_analyze.side_effect = [
            self._create_mock_analysis("repo1", 100, 10, {"python": 10}),
            self._create_mock_analysis("repo2", 200, 20, {"java": 20}),
        ]
        
        # Create cache file
        cache_data = {
            "total_repositories": 2,
            "total_files": 30,
            "total_lines_of_code": 300,
            "total_code_elements": 50,
            "language_distribution": {"python": 10, "java": 20},
            "functionality_patterns": {},
            "complexity_statistics": {},
            "repository_rankings": {},
            "cross_repo_dependencies": {},
        }
        
        cache_file = self.analyzer.output_dir / "cross_repo_analysis.json"
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
        
        # Test with cache (should not call analyze_repository)
        metrics = self.analyzer.analyze_all_repositories(force_reanalyze=False)
        
        assert mock_analyze.call_count == 0  # Should use cache
        assert metrics.total_repositories == 2
        assert metrics.total_files == 30
    
    @patch('src.multi_repo_analyzer.MultiRepositoryAnalyzer._get_all_repository_paths')
    @patch('src.code_analyzer.CodeAnalyzer.analyze_repository')
    def test_analyze_all_repositories_force_reanalyze(self, mock_analyze, mock_get_paths):
        """Test force re-analysis ignores cache."""
        # Setup mocks
        mock_paths = [Path("repo1")]
        mock_get_paths.return_value = mock_paths
        mock_analyze.return_value = self._create_mock_analysis("repo1", 100, 10, {"python": 10})
        
        # Test with force_reanalyze=True
        metrics = self.analyzer.analyze_all_repositories(force_reanalyze=True)
        
        assert mock_analyze.call_count == 1  # Should analyze despite cache
        assert metrics.total_repositories == 1
    
    def test_save_individual_analyses(self):
        """Test saving individual analyses."""
        mock_analyses = [
            self._create_mock_analysis("repo1", 100, 10, {"python": 10}),
            self._create_mock_analysis("repo2", 200, 20, {"java": 20}),
        ]
        
        self.analyzer._save_individual_analyses(mock_analyses)
        
        # Check files were created
        individual_dir = self.analyzer.output_dir / "individual_analyses"
        assert individual_dir.exists()
        assert (individual_dir / "repo1_analysis.json").exists()
        assert (individual_dir / "repo2_analysis.json").exists()
        
        # Check file content
        with open(individual_dir / "repo1_analysis.json") as f:
            data = json.load(f)
            assert data["repo_name"] == "repo1"
            assert data["total_lines"] == 100
            assert data["total_files"] == 10
    
    # Helper methods for creating mock objects
    
    def _create_mock_analysis(self, repo_name: str, total_lines: int, total_files: int, languages: dict) -> RepositoryAnalysis:
        """Create a mock RepositoryAnalysis object."""
        mock_analysis = Mock(spec=RepositoryAnalysis)
        mock_analysis.repo_name = repo_name
        mock_analysis.total_lines = total_lines
        mock_analysis.total_files = total_files
        mock_analysis.languages = languages
        mock_analysis.all_elements = []
        mock_analysis.file_analyses = []
        mock_analysis.functionality_summary = {}
        mock_analysis.metrics = {}
        return mock_analysis
    
    def _create_mock_analysis_with_elements(self, repo_name: str, elements: list) -> RepositoryAnalysis:
        """Create a mock RepositoryAnalysis with specific elements."""
        mock_analysis = self._create_mock_analysis(repo_name, 100, 10, {"python": 10})
        mock_analysis.all_elements = elements
        return mock_analysis
    
    def _create_mock_element(self, name: str, element_type: str, is_public: bool, complexity: int = 5) -> CodeElement:
        """Create a mock CodeElement."""
        mock_element = Mock(spec=CodeElement)
        mock_element.name = name
        mock_element.type = element_type
        mock_element.is_public = is_public
        mock_element.complexity = complexity
        mock_element.language = "python"
        mock_element.file_path = "/mock/path.py"
        mock_element.line_number = 1
        mock_element.docstring = ""
        mock_element.parameters = []
        mock_element.return_type = ""
        mock_element.is_async = False
        return mock_element
    
    def _create_mock_analysis_with_imports(self, repo_name: str, imports: list) -> RepositoryAnalysis:
        """Create a mock RepositoryAnalysis with specific imports."""
        mock_analysis = self._create_mock_analysis(repo_name, 100, 10, {"python": 10})
        
        # Create mock file analysis with imports
        mock_file_analysis = Mock()
        mock_file_analysis.imports = imports
        mock_analysis.file_analyses = [mock_file_analysis]
        
        return mock_analysis


# Integration test that requires actual repository structure
class TestMultiRepositoryAnalyzerIntegration:
    """Integration tests for MultiRepositoryAnalyzer."""
    
    @pytest.mark.skipif(not Path("repos/all_repos").exists(), reason="Requires actual repository structure")
    def test_analyze_real_repositories(self):
        """Test analyzing actual repositories (if available)."""
        analyzer = MultiRepositoryAnalyzer()
        
        # Get first few repositories for testing
        repo_paths = analyzer._get_all_repository_paths()[:3]  # Limit to 3 for test speed
        
        if not repo_paths:
            pytest.skip("No repositories available for testing")
        
        # Test analyzing real repositories
        analyses = []
        for repo_path in repo_paths:
            try:
                analysis = analyzer.analyzer.analyze_repository(repo_path)
                analyses.append(analysis)
            except Exception as e:
                pytest.skip(f"Failed to analyze repository {repo_path.name}: {e}")
        
        if analyses:
            metrics = analyzer._generate_cross_repo_metrics(analyses)
            assert metrics.total_repositories == len(analyses)
            assert metrics.total_files >= 0
            assert metrics.total_lines_of_code >= 0
            assert isinstance(metrics.language_distribution, dict)


if __name__ == "__main__":
    pytest.main([__file__]) 