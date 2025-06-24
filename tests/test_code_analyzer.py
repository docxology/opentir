"""
Tests for code analyzer functionality.
"""

import pytest
from pathlib import Path
import tempfile
import os
from src.code_analyzer import CodeAnalyzer, PythonParser, CodeElement, RepositoryAnalysis

class TestCodeAnalyzer:
    """Test suite for CodeAnalyzer class."""
    
    @pytest.fixture
    def code_analyzer(self):
        """Create a code analyzer for testing."""
        return CodeAnalyzer()
    
    @pytest.fixture
    def temp_repo(self):
        """Create a temporary repository structure for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir) / "test_repo"
            repo_path.mkdir()
            
            # Create test Python file
            python_file = repo_path / "test_module.py"
            python_content = '''
"""Test module for analysis."""

import os
import sys

class TestClass:
    """A test class for demonstration."""
    
    def __init__(self, name):
        self.name = name
    
    def get_name(self):
        """Get the name."""
        return self.name
    
    def _private_method(self):
        """A private method."""
        pass

def public_function(x, y):
    """A public function."""
    if x > y:
        return x
    else:
        return y

async def async_function():
    """An async function."""
    return "async result"

def _private_function():
    """A private function."""
    pass
'''
            python_file.write_text(python_content)
            
            # Create test JavaScript file
            js_file = repo_path / "test_script.js"
            js_content = '''
class TestJSClass {
    constructor(name) {
        this.name = name;
    }
    
    getName() {
        return this.name;
    }
}

function regularFunction(a, b) {
    return a + b;
}

const arrowFunction = async (x) => {
    return x * 2;
};

export { TestJSClass, regularFunction };
'''
            js_file.write_text(js_content)
            
            yield repo_path
    
    def test_initialization(self, code_analyzer):
        """Test code analyzer initialization."""
        assert code_analyzer.logger is not None
        assert "python" in code_analyzer.parsers
        assert "javascript" in code_analyzer.parsers
        assert "typescript" in code_analyzer.parsers
    
    def test_analyze_repository(self, code_analyzer, temp_repo):
        """Test analyzing a complete repository."""
        analysis = code_analyzer.analyze_repository(temp_repo)
        
        assert isinstance(analysis, RepositoryAnalysis)
        assert analysis.repo_name == temp_repo.name
        assert analysis.total_files > 0
        assert analysis.total_lines > 0
        assert len(analysis.all_elements) > 0
        assert "python" in analysis.languages
        assert "javascript" in analysis.languages
    
    def test_get_analyzable_files(self, code_analyzer, temp_repo):
        """Test getting list of analyzable files."""
        files = code_analyzer._get_analyzable_files(temp_repo)
        
        assert len(files) == 2  # Python and JavaScript files
        file_extensions = [f.suffix for f in files]
        assert ".py" in file_extensions
        assert ".js" in file_extensions
    
    def test_should_skip_file(self, code_analyzer):
        """Test file skipping logic."""
        # Should skip git files
        assert code_analyzer._should_skip_file(Path(".git/config"))
        
        # Should skip node_modules
        assert code_analyzer._should_skip_file(Path("node_modules/package/index.js"))
        
        # Should skip __pycache__
        assert code_analyzer._should_skip_file(Path("__pycache__/module.pyc"))
        
        # Should not skip regular files
        assert not code_analyzer._should_skip_file(Path("src/module.py"))
    
    def test_analyze_file(self, code_analyzer, temp_repo):
        """Test analyzing a single file."""
        python_file = temp_repo / "test_module.py"
        analysis = code_analyzer._analyze_file(python_file, "python")
        
        assert analysis is not None
        assert analysis.language == "python"
        assert analysis.lines_of_code > 0
        assert len(analysis.elements) > 0
        assert len(analysis.imports) > 0
    
    def test_generate_functionality_matrix(self, code_analyzer, temp_repo):
        """Test generating functionality matrix."""
        analysis = code_analyzer.analyze_repository(temp_repo)
        matrix = code_analyzer.generate_functionality_matrix([analysis])
        
        assert "repositories" in matrix
        assert "global_summary" in matrix
        assert temp_repo.name in matrix["repositories"]
        
        global_summary = matrix["global_summary"]
        assert "total_repositories" in global_summary
        assert "languages" in global_summary
        assert "functionality_categories" in global_summary

class TestPythonParser:
    """Test suite for PythonParser class."""
    
    @pytest.fixture
    def python_parser(self):
        """Create a Python parser for testing."""
        return PythonParser()
    
    def test_parse_python_code(self, python_parser):
        """Test parsing Python code."""
        code = '''
def test_function(x, y):
    """Test function."""
    return x + y

class TestClass:
    """Test class."""
    
    def method(self):
        """Test method."""
        pass

async def async_function():
    """Async function."""
    return "result"
'''
        
        elements = python_parser.parse(code, "test.py")
        
        # Should find function, class, method, and async function
        assert len(elements) >= 4
        
        # Check function
        functions = [e for e in elements if e.type == "function"]
        assert len(functions) >= 2  # test_function and async_function
        
        # Check class
        classes = [e for e in elements if e.type == "class"]
        assert len(classes) == 1
        assert classes[0].name == "TestClass"
        
        # Check async function
        async_functions = [e for e in elements if e.is_async]
        assert len(async_functions) == 1
        assert async_functions[0].name == "async_function"
    
    def test_extract_imports(self, python_parser):
        """Test extracting import statements."""
        code = '''
import os
import sys
from pathlib import Path
from typing import Dict, List
'''
        
        imports = python_parser.extract_imports(code)
        
        assert "os" in imports
        assert "sys" in imports
        assert "pathlib" in imports
        assert "typing" in imports
    
    def test_calculate_complexity(self, python_parser):
        """Test complexity calculation."""
        # Simple function - complexity should be 1
        simple_code = '''
def simple_function():
    return True
'''
        elements = python_parser.parse(simple_code, "test.py")
        assert elements[0].complexity == 1
        
        # Complex function with if/else and loop
        complex_code = '''
def complex_function(x):
    if x > 0:
        for i in range(x):
            if i % 2 == 0:
                print(i)
    else:
        return False
    return True
'''
        elements = python_parser.parse(complex_code, "test.py")
        assert elements[0].complexity > 1
    
    def test_create_function_element(self, python_parser):
        """Test creating function element from AST node."""
        code = '''
def test_function(x, y, z=None):
    """Test function with parameters."""
    return x + y
'''
        
        elements = python_parser.parse(code, "test.py")
        func = elements[0]
        
        assert func.name == "test_function"
        assert func.type == "function"
        assert func.language == "python"
        assert func.docstring == "Test function with parameters."
        assert "x" in func.parameters
        assert "y" in func.parameters
        assert "z" in func.parameters
        assert func.is_public is True
        assert func.is_async is False
    
    def test_create_class_element(self, python_parser):
        """Test creating class element from AST node."""
        code = '''
class TestClass:
    """A test class."""
    
    def method1(self):
        pass
    
    def method2(self):
        pass
'''
        
        elements = python_parser.parse(code, "test.py")
        class_elements = [e for e in elements if e.type == "class"]
        
        assert len(class_elements) == 1
        cls = class_elements[0]
        
        assert cls.name == "TestClass"
        assert cls.type == "class"
        assert cls.docstring == "A test class."
        assert cls.is_public is True
        assert cls.complexity > 1  # Should account for methods
    
    def test_private_vs_public(self, python_parser):
        """Test detection of private vs public elements."""
        code = '''
def public_function():
    pass

def _private_function():
    pass

class PublicClass:
    pass

class _PrivateClass:
    pass
'''
        
        elements = python_parser.parse(code, "test.py")
        
        public_elements = [e for e in elements if e.is_public]
        private_elements = [e for e in elements if not e.is_public]
        
        assert len(public_elements) == 2  # public_function, PublicClass
        assert len(private_elements) == 2  # _private_function, _PrivateClass

class TestCodeElement:
    """Test suite for CodeElement class."""
    
    def test_code_element_creation(self):
        """Test creating a CodeElement instance."""
        element = CodeElement(
            name="test_function",
            type="function",
            language="python",
            file_path="/test/file.py",
            line_number=10,
            docstring="Test function",
            parameters=["x", "y"],
            return_type="int",
            complexity=2,
            is_public=True,
            is_async=False
        )
        
        assert element.name == "test_function"
        assert element.type == "function"
        assert element.language == "python"
        assert element.file_path == "/test/file.py"
        assert element.line_number == 10
        assert element.docstring == "Test function"
        assert element.parameters == ["x", "y"]
        assert element.return_type == "int"
        assert element.complexity == 2
        assert element.is_public is True
        assert element.is_async is False

class TestFunctionalityCategories:
    """Test functionality categorization."""
    
    @pytest.fixture
    def code_analyzer(self):
        return CodeAnalyzer()
    
    def test_categorize_functionality(self, code_analyzer):
        """Test functionality categorization."""
        # Test data processing
        assert code_analyzer._categorize_functionality("process_data", "process input data") == "data_processing"
        
        # Test API functions
        assert code_analyzer._categorize_functionality("api_handler", "handle HTTP requests") == "api"
        
        # Test database functions
        assert code_analyzer._categorize_functionality("query_db", "query database") == "database"
        
        # Test file operations
        assert code_analyzer._categorize_functionality("read_file", "read file content") == "file_operations"
        
        # Test utilities
        assert code_analyzer._categorize_functionality("format_string", "utility function") == "utilities"
        
        # Test general category (fallback)
        assert code_analyzer._categorize_functionality("unknown_func", "some function") == "general" 