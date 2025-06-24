"""
Tests for documentation generator functionality.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.docs_generator import DocumentationGenerator
from src.code_analyzer import RepositoryAnalysis, CodeElement


class TestDocumentationGenerator:
    """Test suite for DocumentationGenerator class."""
    
    @pytest.fixture
    def temp_docs_dir(self):
        """Create temporary docs directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            docs_dir = Path(temp_dir) / "docs"
            docs_dir.mkdir()
            yield docs_dir
    
    @pytest.fixture
    def docs_generator(self, temp_docs_dir):
        """Create documentation generator for testing."""
        return DocumentationGenerator(temp_docs_dir)
    
    @pytest.fixture
    def mock_analyses(self):
        """Create mock repository analyses for testing."""
        analyses = []
        
        # Create mock analysis 1
        mock_analysis1 = Mock(spec=RepositoryAnalysis)
        mock_analysis1.repo_name = "test-repo-1"
        mock_analysis1.total_files = 10
        mock_analysis1.total_lines = 1000
        mock_analysis1.languages = {"python": 8, "javascript": 2}
        mock_analysis1.functionality_summary = {
            "total_elements": 50,
            "public_methods": [{"name": "get_data", "file": "main.py", "parameters": [], "docstring": "Get data"}],
            "classes": [{"name": "TestClass", "file": "main.py", "docstring": "Test class"}],
            "async_functions": [],
            "functionality_categories": {"utilities": ["helper_func"], "api": ["api_call"]}
        }
        mock_analysis1.metrics = {
            "average_file_complexity": 5.5,
            "documentation_ratio": 0.8,
            "total_complexity": 55
        }
        mock_analysis1.all_elements = [
            self._create_mock_element("get_data", "function", True, 3),
            self._create_mock_element("TestClass", "class", True, 5)
        ]
        analyses.append(mock_analysis1)
        
        # Create mock analysis 2
        mock_analysis2 = Mock(spec=RepositoryAnalysis)
        mock_analysis2.repo_name = "test-repo-2"
        mock_analysis2.total_files = 15
        mock_analysis2.total_lines = 2000
        mock_analysis2.languages = {"java": 12, "python": 3}
        mock_analysis2.functionality_summary = {
            "total_elements": 75,
            "public_methods": [{"name": "process_data", "file": "processor.java", "parameters": ["data"], "docstring": "Process data"}],
            "classes": [{"name": "DataProcessor", "file": "processor.java", "docstring": "Data processor"}],
            "async_functions": [],
            "functionality_categories": {"data_processing": ["process_data"], "utilities": ["util_func"]}
        }
        mock_analysis2.metrics = {
            "average_file_complexity": 7.2,
            "documentation_ratio": 0.6,
            "total_complexity": 108
        }
        mock_analysis2.all_elements = [
            self._create_mock_element("process_data", "function", True, 7),
            self._create_mock_element("DataProcessor", "class", True, 10)
        ]
        analyses.append(mock_analysis2)
        
        return analyses
    
    @pytest.fixture
    def mock_functionality_matrix(self):
        """Create mock functionality matrix for testing."""
        return {
            "repositories": {
                "test-repo-1": {
                    "basic_info": {"total_files": 10, "total_lines": 1000, "languages": {"python": 8}},
                    "functionality": {"total_elements": 50, "public_methods": [], "classes": []},
                    "metrics": {"total_complexity": 55, "documentation_ratio": 0.8}
                },
                "test-repo-2": {
                    "basic_info": {"total_files": 15, "total_lines": 2000, "languages": {"java": 12}},
                    "functionality": {"total_elements": 75, "public_methods": [], "classes": []},
                    "metrics": {"total_complexity": 108, "documentation_ratio": 0.6}
                }
            },
            "global_summary": {
                "total_repositories": 2,
                "languages": {"python": 11, "java": 12, "javascript": 2},
                "top_methods": {"get_data": 1, "process_data": 1, "helper_func": 2},
                "functionality_categories": {"utilities": ["helper_func"], "api": ["api_call"]}
            }
        }
    
    @pytest.fixture
    def mock_org_info(self):
        """Create mock organization info for testing."""
        return {
            "name": "Test Organization",
            "login": "test-org",
            "description": "Test organization for testing",
            "public_repos": 50,
            "html_url": "https://github.com/test-org"
        }
    
    def _create_mock_element(self, name: str, element_type: str, is_public: bool, complexity: int) -> CodeElement:
        """Create mock CodeElement for testing."""
        mock_element = Mock(spec=CodeElement)
        mock_element.name = name
        mock_element.type = element_type
        mock_element.is_public = is_public
        mock_element.complexity = complexity
        mock_element.language = "python"
        mock_element.file_path = "/mock/path.py"
        mock_element.line_number = 1
        mock_element.docstring = f"Mock {name}"
        mock_element.parameters = []
        mock_element.return_type = ""
        mock_element.is_async = False
        return mock_element
    
    def test_initialization(self, temp_docs_dir):
        """Test DocumentationGenerator initialization."""
        docs_gen = DocumentationGenerator(temp_docs_dir)
        
        assert docs_gen.docs_dir == temp_docs_dir
        assert docs_gen.logger is not None
        assert docs_gen.jinja_env is not None
    
    def test_setup_docs_structure(self, docs_generator):
        """Test documentation structure setup."""
        docs_generator._setup_docs_structure()
        
        expected_dirs = [
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
        
        for dir_path in expected_dirs:
            assert (docs_generator.docs_dir / dir_path).exists()
    
    def test_generate_main_index(self, docs_generator, mock_org_info, mock_analyses):
        """Test main index generation."""
        result = docs_generator._generate_main_index(mock_org_info, mock_analyses)
        
        # Check that file was created
        index_file = docs_generator.docs_dir / "index.md"
        assert index_file.exists()
        assert result == str(index_file)
        
        # Check content
        content = index_file.read_text()
        assert "Test Organization" in content
        assert "2" in content  # total repositories
        assert "25" in content  # total files (10 + 15)
    
    def test_generate_functionality_matrix_docs(self, docs_generator, mock_functionality_matrix):
        """Test functionality matrix documentation generation."""
        result = docs_generator._generate_functionality_matrix_docs(mock_functionality_matrix)
        
        # Check that file was created
        matrix_file = docs_generator.docs_dir / "analysis" / "functionality_matrix.md"
        assert matrix_file.exists()
        assert result == str(matrix_file)
        
        # Check content
        content = matrix_file.read_text()
        assert "Comprehensive Functionality Matrix" in content
        assert "test-repo-1" in content
        assert "test-repo-2" in content
    
    def test_create_functionality_comparison_table(self, docs_generator, mock_functionality_matrix):
        """Test functionality comparison table creation."""
        repositories = mock_functionality_matrix["repositories"]
        table_data = docs_generator._create_functionality_comparison_table(repositories)
        
        assert len(table_data) == 2
        assert table_data[0]["repository"] in ["test-repo-1", "test-repo-2"]
        assert "total_files" in table_data[0]
        assert "total_elements" in table_data[0]
        assert "complexity" in table_data[0]
    
    def test_analyze_method_frequency(self, docs_generator):
        """Test method frequency analysis."""
        global_summary = {
            "top_methods": {"get_data": 5, "process_data": 3, "helper_func": 8, "util_func": 1}
        }
        
        result = docs_generator._analyze_method_frequency(global_summary)
        
        assert "most_common_methods" in result
        assert "method_distribution" in result
        assert "common_patterns" in result
        
        # Check method distribution
        distribution = result["method_distribution"]
        assert distribution["unique_methods"] == 4
        assert distribution["methods_used_once"] == 1
        assert distribution["methods_used_5_plus"] == 2
    
    def test_identify_common_patterns(self, docs_generator):
        """Test common pattern identification."""
        methods = {
            "get_data": 5, "get_info": 3, "set_value": 2, "set_config": 1,
            "helper_util": 2, "test_helper": 1, "process_factory": 1
        }
        
        patterns = docs_generator._identify_common_patterns(methods)
        
        # Should identify get_, set_ patterns
        pattern_names = [p["pattern"] for p in patterns]
        assert "get_" in pattern_names
        assert "set_" in pattern_names
    
    @patch('pandas.DataFrame.to_csv')
    def test_export_functionality_data(self, mock_to_csv, docs_generator, mock_functionality_matrix):
        """Test functionality data export."""
        docs_generator._export_functionality_data(mock_functionality_matrix)
        
        # Check that CSV export was called
        assert mock_to_csv.called
        
        # Check that JSON file was created
        json_file = docs_generator.docs_dir / "assets" / "data" / "full_functionality_matrix.json"
        assert json_file.exists()
    
    def test_generate_api_reference(self, docs_generator, mock_analyses):
        """Test API reference generation."""
        result = docs_generator._generate_api_reference(mock_analyses)
        
        assert "methods" in result
        assert "classes" in result
        assert "modules" in result
        
        # Check that files were created
        methods_file = docs_generator.docs_dir / "api_reference" / "methods" / "index.md"
        classes_file = docs_generator.docs_dir / "api_reference" / "classes" / "index.md"
        modules_file = docs_generator.docs_dir / "api_reference" / "modules" / "index.md"
        
        assert methods_file.exists()
        assert classes_file.exists()
        assert modules_file.exists()
    
    def test_generate_mkdocs_config(self, docs_generator):
        """Test MkDocs configuration generation."""
        result = docs_generator._generate_mkdocs_config()
        
        config_file = docs_generator.docs_dir / "mkdocs.yml"
        assert config_file.exists()
        assert result == str(config_file)
        
        # Check content
        content = config_file.read_text()
        assert "site_name:" in content
        assert "theme:" in content
        assert "nav:" in content
    
    def test_get_top_elements_for_display(self, docs_generator):
        """Test top elements extraction for display."""
        elements = [
            self._create_mock_element("func1", "function", True, 5),
            self._create_mock_element("func2", "function", True, 8),
            self._create_mock_element("Class1", "class", True, 10),
            self._create_mock_element("method1", "method", True, 3),
        ]
        
        result = docs_generator._get_top_elements_for_display(elements)
        
        assert "functions" in result
        assert "classes" in result
        assert "methods" in result
        
        # Should be sorted by complexity
        assert len(result["functions"]) == 2
        assert result["functions"][0]["name"] == "func2"  # Higher complexity first
    
    def test_group_analyses_by_language(self, docs_generator, mock_analyses):
        """Test grouping analyses by language."""
        result = docs_generator._group_analyses_by_language(mock_analyses)
        
        # Should group by primary language (highest file count)
        assert "python" in result
        assert "java" in result
        assert len(result["python"]) == 1
        assert len(result["java"]) == 1
    
    def test_group_analyses_by_category(self, docs_generator, mock_analyses):
        """Test grouping analyses by category."""
        result = docs_generator._group_analyses_by_category(mock_analyses)
        
        assert "high_activity" in result
        assert "medium_activity" in result
        assert "low_activity" in result
        
        # Based on total_lines: repo1=1000 (medium), repo2=2000 (medium)
        assert len(result["medium_activity"]) == 2
    
    def test_generate_complete_documentation(self, docs_generator, mock_analyses, mock_functionality_matrix, mock_org_info):
        """Test complete documentation generation."""
        result = docs_generator.generate_complete_documentation(
            mock_analyses, mock_functionality_matrix, mock_org_info
        )
        
        assert "success" in str(result).lower() or len(result) > 0
        assert "main_index" in result
        assert "repository_docs" in result
        assert "functionality_matrix" in result
        assert "api_reference" in result
        assert "generation_summary" in result
    
    def test_generate_summary_stats(self, docs_generator, mock_analyses):
        """Test summary statistics generation."""
        result = docs_generator._generate_summary_stats(mock_analyses)
        
        assert "total_repositories_analyzed" in result
        assert "total_files_documented" in result
        assert "total_lines_analyzed" in result
        assert "total_elements_extracted" in result
        assert "generation_timestamp" in result
        
        assert result["total_repositories_analyzed"] == 2
        assert result["total_files_documented"] == 25  # 10 + 15
        assert result["total_lines_analyzed"] == 3000  # 1000 + 2000


class TestTemplateCreation:
    """Test template creation functionality."""
    
    def test_create_default_templates(self):
        """Test default template creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            template_dir = Path(temp_dir) / "templates"
            docs_gen = DocumentationGenerator()
            docs_gen._create_default_templates(template_dir)
            
            # Check that templates were created
            expected_templates = [
                "main_index.md",
                "repository_detail.md", 
                "functionality_matrix.md",
                "mkdocs.yml"
            ]
            
            for template in expected_templates:
                template_file = template_dir / template
                assert template_file.exists()
                assert template_file.stat().st_size > 0  # Not empty


if __name__ == "__main__":
    pytest.main([__file__]) 