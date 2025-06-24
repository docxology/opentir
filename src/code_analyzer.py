"""
Code analyzer for extracting methods, functions, classes, and analyzing functionality.
Supports multiple programming languages and generates comprehensive analysis reports.
"""

import ast
import os
import re
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

from .config import config, SUPPORTED_FILE_EXTENSIONS
from .utils import Logger, FileUtils

@dataclass
class CodeElement:
    """Represents a code element (function, class, method, etc.)."""
    name: str
    type: str  # function, class, method, variable
    language: str
    file_path: str
    line_number: int
    docstring: str
    parameters: List[str]
    return_type: str
    complexity: int
    is_public: bool
    is_async: bool

@dataclass
class FileAnalysis:
    """Represents analysis results for a single file."""
    file_path: str
    language: str
    lines_of_code: int
    elements: List[CodeElement]
    imports: List[str]
    dependencies: List[str]
    complexity_score: int

@dataclass
class RepositoryAnalysis:
    """Represents comprehensive analysis results for a repository."""
    repo_name: str
    total_files: int
    total_lines: int
    languages: Dict[str, int]  # language -> file count
    file_analyses: List[FileAnalysis]
    all_elements: List[CodeElement]
    functionality_summary: Dict[str, Any]
    metrics: Dict[str, Any]

class CodeAnalyzer:
    """
    Comprehensive code analyzer for extracting and analyzing code elements.
    Supports multiple programming languages with extensible parsing.
    """
    
    def __init__(self):
        """Initialize code analyzer with language parsers."""
        self.logger = Logger("code_analyzer")
        self.parsers = {
            "python": PythonParser(),
            "javascript": JavaScriptParser(),
            "typescript": TypeScriptParser(),
            "java": JavaParser(),
            "go": GoParser(),
        }
        
    def analyze_repository(self, repo_path: Path) -> RepositoryAnalysis:
        """
        Analyze entire repository and extract all code elements.
        Returns comprehensive analysis including functionality mapping.
        """
        self.logger.info(f"Analyzing repository: {repo_path.name}")
        
        if not repo_path.exists():
            raise ValueError(f"Repository path does not exist: {repo_path}")
        
        # Initialize analysis results
        file_analyses = []
        all_elements = []
        languages = defaultdict(int)
        total_files = 0
        total_lines = 0
        
        # Analyze all supported files
        for file_path in self._get_analyzable_files(repo_path):
            try:
                language = FileUtils.get_language_from_extension(file_path)
                if not language or language not in self.parsers:
                    continue
                
                file_analysis = self._analyze_file(file_path, language)
                if file_analysis:
                    file_analyses.append(file_analysis)
                    all_elements.extend(file_analysis.elements)
                    languages[language] += 1
                    total_files += 1
                    total_lines += file_analysis.lines_of_code
                    
            except Exception as e:
                self.logger.debug(f"Error analyzing {file_path}: {e}")
                continue
        
        # Generate functionality summary
        functionality_summary = self._generate_functionality_summary(all_elements)
        
        # Calculate metrics
        metrics = self._calculate_repository_metrics(file_analyses, all_elements)
        
        return RepositoryAnalysis(
            repo_name=repo_path.name,
            total_files=total_files,
            total_lines=total_lines,
            languages=dict(languages),
            file_analyses=file_analyses,
            all_elements=all_elements,
            functionality_summary=functionality_summary,
            metrics=metrics,
        )
    
    def _get_analyzable_files(self, repo_path: Path) -> List[Path]:
        """Get list of files that can be analyzed."""
        analyzable_files = []
        
        for file_path in repo_path.rglob("*"):
            if (file_path.is_file() and 
                not self._should_skip_file(file_path) and
                FileUtils.get_language_from_extension(file_path)):
                analyzable_files.append(file_path)
        
        return analyzable_files
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped during analysis."""
        skip_patterns = [
            ".git/", "node_modules/", "__pycache__/", ".pytest_cache/",
            "build/", "dist/", "target/", ".gradle/", "vendor/"
        ]
        
        path_str = str(file_path)
        return any(pattern in path_str for pattern in skip_patterns)
    
    def _analyze_file(self, file_path: Path, language: str) -> Optional[FileAnalysis]:
        """Analyze a single file and extract code elements."""
        try:
            if not FileUtils.is_text_file(file_path):
                return None
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            parser = self.parsers.get(language)
            if not parser:
                return None
            
            elements = parser.parse(content, str(file_path))
            imports = parser.extract_imports(content)
            dependencies = parser.extract_dependencies(content)
            lines_of_code = len([line for line in content.split('\n') if line.strip()])
            complexity_score = self._calculate_file_complexity(elements)
            
            return FileAnalysis(
                file_path=str(file_path),
                language=language,
                lines_of_code=lines_of_code,
                elements=elements,
                imports=imports,
                dependencies=dependencies,
                complexity_score=complexity_score,
            )
            
        except Exception as e:
            self.logger.debug(f"Error analyzing file {file_path}: {e}")
            return None
    
    def _calculate_file_complexity(self, elements: List[CodeElement]) -> int:
        """Calculate complexity score for a file."""
        return sum(element.complexity for element in elements)
    
    def _generate_functionality_summary(self, elements: List[CodeElement]) -> Dict[str, Any]:
        """Generate comprehensive functionality summary from code elements."""
        summary = {
            "total_elements": len(elements),
            "by_type": defaultdict(int),
            "by_language": defaultdict(int),
            "public_methods": [],
            "classes": [],
            "async_functions": [],
            "functionality_categories": defaultdict(list),
        }
        
        for element in elements:
            summary["by_type"][element.type] += 1
            summary["by_language"][element.language] += 1
            
            if element.is_public and element.type in ["function", "method"]:
                summary["public_methods"].append({
                    "name": element.name,
                    "file": element.file_path,
                    "parameters": element.parameters,
                    "docstring": element.docstring,
                })
            
            if element.type == "class":
                summary["classes"].append({
                    "name": element.name,
                    "file": element.file_path,
                    "docstring": element.docstring,
                })
            
            if element.is_async:
                summary["async_functions"].append({
                    "name": element.name,
                    "file": element.file_path,
                })
            
            # Categorize by functionality
            category = self._categorize_functionality(element.name, element.docstring)
            summary["functionality_categories"][category].append(element.name)
        
        # Convert defaultdicts to regular dicts for JSON serialization
        summary["by_type"] = dict(summary["by_type"])
        summary["by_language"] = dict(summary["by_language"])
        summary["functionality_categories"] = dict(summary["functionality_categories"])
        
        return summary
    
    def _categorize_functionality(self, name: str, docstring: str) -> str:
        """Categorize functionality based on name and docstring."""
        text = f"{name} {docstring}".lower()
        
        # First check name-based categorization
        name = name.lower()
        if name.startswith(('util_', 'utils_', 'helper_', 'format_', 'validate_', 'check_')):
            return "utilities"
        if name.startswith(('db_', 'database_', 'query_')):
            return "database"
        if name.startswith(('api_', 'http_', 'rest_')):
            return "api"
        if name.startswith(('file_', 'read_', 'write_', 'save_', 'load_')):
            return "file_operations"
        
        # Then check content-based categorization
        categories = {
            "data_processing": ["process data", "transform data", "parse data", "convert data", "filter data"],
            "api": ["api endpoint", "http request", "rest api", "api response", "http client"],
            "database": ["database query", "sql query", "database connection", "database transaction"],
            "file_operations": ["file operation", "file system", "file io", "file handling"],
            "utilities": ["utility function", "helper function", "format string", "validation", "utility method"],
            "authentication": ["authentication", "authorization", "login process", "token validation"],
            "testing": ["unit test", "test case", "mock object", "test fixture"],
            "configuration": ["configuration", "settings", "config option", "parameter config"],
            "ui": ["user interface", "ui component", "render view", "display element"],
            "logging": ["log message", "debug log", "error log", "logging system"],
        }
        
        for category, phrases in categories.items():
            if any(phrase in text for phrase in phrases):
                return category
        
        # If no specific category is found, use keyword-based fallback
        keywords = {
            "data_processing": ["process", "transform", "parse", "convert", "filter"],
            "api": ["api", "endpoint", "request", "response", "http"],
            "database": ["database", "query", "select", "insert", "update"],
            "file_operations": ["file", "read", "write", "save", "load"],
            "utilities": ["util", "helper", "format", "validate", "check"],
            "authentication": ["auth", "login", "token", "credential"],
            "testing": ["test", "mock", "assert", "verify"],
            "configuration": ["config", "setting", "option", "param"],
            "ui": ["ui", "interface", "component", "render"],
            "logging": ["log", "debug", "error", "warn"],
        }
        
        for category, kwords in keywords.items():
            if any(kw in name for kw in kwords):
                return category
        
        return "general"
    
    def _calculate_repository_metrics(self, file_analyses: List[FileAnalysis], elements: List[CodeElement]) -> Dict[str, Any]:
        """Calculate comprehensive metrics for repository."""
        if not file_analyses:
            return {}
        
        total_complexity = sum(analysis.complexity_score for analysis in file_analyses)
        avg_complexity = total_complexity / len(file_analyses) if file_analyses else 0
        
        return {
            "average_file_complexity": avg_complexity,
            "total_complexity": total_complexity,
            "average_elements_per_file": len(elements) / len(file_analyses) if file_analyses else 0,
            "public_method_ratio": len([e for e in elements if e.is_public]) / len(elements) if elements else 0,
            "async_function_count": len([e for e in elements if e.is_async]),
            "documentation_ratio": len([e for e in elements if e.docstring]) / len(elements) if elements else 0,
        }
    
    def generate_functionality_matrix(self, analyses: List[RepositoryAnalysis]) -> Dict[str, Any]:
        """
        Generate comprehensive functionality matrix across all repositories.
        Creates a vast table of functionality for documentation.
        """
        self.logger.info("Generating functionality matrix...")
        
        matrix = {
            "repositories": {},
            "global_summary": {
                "total_repositories": len(analyses),
                "languages": defaultdict(int),
                "functionality_categories": defaultdict(set),
                "common_patterns": defaultdict(int),
                "top_methods": defaultdict(int),
            }
        }
        
        for analysis in analyses:
            repo_matrix = {
                "basic_info": {
                    "total_files": analysis.total_files,
                    "total_lines": analysis.total_lines,
                    "languages": analysis.languages,
                },
                "functionality": analysis.functionality_summary,
                "metrics": analysis.metrics,
                "top_elements": self._get_top_elements(analysis.all_elements),
            }
            
            matrix["repositories"][analysis.repo_name] = repo_matrix
            
            # Update global summary
            for lang, count in analysis.languages.items():
                matrix["global_summary"]["languages"][lang] += count
            
            for category, elements in analysis.functionality_summary.get("functionality_categories", {}).items():
                matrix["global_summary"]["functionality_categories"][category].update(elements)
            
            for element in analysis.all_elements:
                if element.is_public:
                    matrix["global_summary"]["top_methods"][element.name] += 1
        
        # Convert sets to lists for JSON serialization
        for category in matrix["global_summary"]["functionality_categories"]:
            matrix["global_summary"]["functionality_categories"][category] = list(
                matrix["global_summary"]["functionality_categories"][category]
            )
        
        matrix["global_summary"]["languages"] = dict(matrix["global_summary"]["languages"])
        matrix["global_summary"]["functionality_categories"] = dict(matrix["global_summary"]["functionality_categories"])
        matrix["global_summary"]["top_methods"] = dict(sorted(
            matrix["global_summary"]["top_methods"].items(), 
            key=lambda x: x[1], reverse=True
        )[:100])  # Top 100 most common methods
        
        return matrix
    
    def _get_top_elements(self, elements: List[CodeElement], limit: int = 20) -> Dict[str, List[Dict[str, Any]]]:
        """Get top elements by type for a repository."""
        top_elements = {
            "functions": [],
            "classes": [],
            "methods": [],
        }
        
        for element_type in top_elements.keys():
            filtered_elements = [e for e in elements if e.type == element_type.rstrip('s')]
            sorted_elements = sorted(filtered_elements, key=lambda x: x.complexity, reverse=True)
            
            top_elements[element_type] = [
                {
                    "name": e.name,
                    "file": e.file_path,
                    "complexity": e.complexity,
                    "parameters": e.parameters,
                    "docstring": e.docstring[:200] + "..." if len(e.docstring) > 200 else e.docstring,
                }
                for e in sorted_elements[:limit]
            ]
        
        return top_elements

# Language-specific parsers
class LanguageParser:
    """Base class for language-specific parsers."""
    
    def parse(self, content: str, file_path: str) -> List[CodeElement]:
        """Parse code content and extract elements."""
        raise NotImplementedError
    
    def extract_imports(self, content: str) -> List[str]:
        """Extract import statements."""
        raise NotImplementedError
    
    def extract_dependencies(self, content: str) -> List[str]:
        """Extract dependencies."""
        raise NotImplementedError

class PythonParser(LanguageParser):
    """Parser for Python code."""
    
    def parse(self, content: str, file_path: str) -> List[CodeElement]:
        """Parse Python code using AST."""
        elements = []
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    elements.append(self._create_function_element(node, file_path))
                elif isinstance(node, ast.AsyncFunctionDef):
                    elements.append(self._create_async_function_element(node, file_path))
                elif isinstance(node, ast.ClassDef):
                    elements.append(self._create_class_element(node, file_path))
        
        except SyntaxError:
            pass  # Skip files with syntax errors
        
        return elements
    
    def _create_function_element(self, node: ast.FunctionDef, file_path: str) -> CodeElement:
        """Create CodeElement from function node."""
        return CodeElement(
            name=node.name,
            type="function",
            language="python",
            file_path=file_path,
            line_number=node.lineno,
            docstring=ast.get_docstring(node) or "",
            parameters=[arg.arg for arg in node.args.args],
            return_type="",  # Could be enhanced with type annotations
            complexity=self._calculate_complexity(node),
            is_public=not node.name.startswith('_'),
            is_async=False,
        )
    
    def _create_async_function_element(self, node: ast.AsyncFunctionDef, file_path: str) -> CodeElement:
        """Create CodeElement from async function node."""
        return CodeElement(
            name=node.name,
            type="function",
            language="python",
            file_path=file_path,
            line_number=node.lineno,
            docstring=ast.get_docstring(node) or "",
            parameters=[arg.arg for arg in node.args.args],
            return_type="",
            complexity=self._calculate_complexity(node),
            is_public=not node.name.startswith('_'),
            is_async=True,
        )
    
    def _create_class_element(self, node: ast.ClassDef, file_path: str) -> CodeElement:
        """Create CodeElement from class node."""
        return CodeElement(
            name=node.name,
            type="class",
            language="python",
            file_path=file_path,
            line_number=node.lineno,
            docstring=ast.get_docstring(node) or "",
            parameters=[],
            return_type="",
            complexity=len(node.body),
            is_public=not node.name.startswith('_'),
            is_async=False,
        )
    
    def _calculate_complexity(self, node) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler, ast.With)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def extract_imports(self, content: str) -> List[str]:
        """Extract Python import statements."""
        imports = []
        
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
        except SyntaxError:
            pass
        
        return imports
    
    def extract_dependencies(self, content: str) -> List[str]:
        """Extract Python dependencies."""
        return self.extract_imports(content)

# Simplified parsers for other languages (can be enhanced)
class JavaScriptParser(LanguageParser):
    """Basic parser for JavaScript code."""
    
    def parse(self, content: str, file_path: str) -> List[CodeElement]:
        """Parse JavaScript using regex patterns."""
        elements = []
        
        # Simple regex patterns for function detection
        function_pattern = r'(?:function\s+|const\s+|let\s+|var\s+)(\w+)\s*(?:=\s*)?(?:async\s+)?(?:function\s*)?\([^)]*\)'
        class_pattern = r'class\s+(\w+)'
        
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Functions
            func_match = re.search(function_pattern, line)
            if func_match:
                elements.append(CodeElement(
                    name=func_match.group(1),
                    type="function",
                    language="javascript",
                    file_path=file_path,
                    line_number=i,
                    docstring="",
                    parameters=[],
                    return_type="",
                    complexity=1,
                    is_public=True,
                    is_async="async" in line,
                ))
            
            # Classes
            class_match = re.search(class_pattern, line)
            if class_match:
                elements.append(CodeElement(
                    name=class_match.group(1),
                    type="class",
                    language="javascript",
                    file_path=file_path,
                    line_number=i,
                    docstring="",
                    parameters=[],
                    return_type="",
                    complexity=1,
                    is_public=True,
                    is_async=False,
                ))
        
        return elements
    
    def extract_imports(self, content: str) -> List[str]:
        """Extract JavaScript imports."""
        imports = []
        import_pattern = r'(?:import|require)\s*\(?[\'"]([^\'"]+)[\'"]'
        
        for match in re.finditer(import_pattern, content):
            imports.append(match.group(1))
        
        return imports
    
    def extract_dependencies(self, content: str) -> List[str]:
        """Extract JavaScript dependencies."""
        return self.extract_imports(content)

class TypeScriptParser(JavaScriptParser):
    """Parser for TypeScript code (extends JavaScript parser)."""
    
    def parse(self, content: str, file_path: str) -> List[CodeElement]:
        """Parse TypeScript code."""
        elements = super().parse(content, file_path)
        
        # Update language for all elements
        for element in elements:
            element.language = "typescript"
        
        return elements

class JavaParser(LanguageParser):
    """Basic parser for Java code."""
    
    def parse(self, content: str, file_path: str) -> List[CodeElement]:
        """Parse Java using regex patterns."""
        elements = []
        
        # Simple patterns for Java
        method_pattern = r'(?:public|private|protected)?\s*(?:static\s+)?(?:\w+\s+)+(\w+)\s*\([^)]*\)'
        class_pattern = r'(?:public\s+)?class\s+(\w+)'
        
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            method_match = re.search(method_pattern, line)
            if method_match and not line.strip().startswith('//'):
                elements.append(CodeElement(
                    name=method_match.group(1),
                    type="method",
                    language="java",
                    file_path=file_path,
                    line_number=i,
                    docstring="",
                    parameters=[],
                    return_type="",
                    complexity=1,
                    is_public="public" in line,
                    is_async=False,
                ))
            
            class_match = re.search(class_pattern, line)
            if class_match:
                elements.append(CodeElement(
                    name=class_match.group(1),
                    type="class",
                    language="java",
                    file_path=file_path,
                    line_number=i,
                    docstring="",
                    parameters=[],
                    return_type="",
                    complexity=1,
                    is_public="public" in line,
                    is_async=False,
                ))
        
        return elements
    
    def extract_imports(self, content: str) -> List[str]:
        """Extract Java imports."""
        imports = []
        import_pattern = r'import\s+([^;]+);'
        
        for match in re.finditer(import_pattern, content):
            imports.append(match.group(1).strip())
        
        return imports
    
    def extract_dependencies(self, content: str) -> List[str]:
        """Extract Java dependencies."""
        return self.extract_imports(content)

class GoParser(LanguageParser):
    """Basic parser for Go code."""
    
    def parse(self, content: str, file_path: str) -> List[CodeElement]:
        """Parse Go using regex patterns."""
        elements = []
        
        func_pattern = r'func\s+(?:\([^)]*\)\s+)?(\w+)\s*\([^)]*\)'
        type_pattern = r'type\s+(\w+)\s+(?:struct|interface)'
        
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            func_match = re.search(func_pattern, line)
            if func_match:
                elements.append(CodeElement(
                    name=func_match.group(1),
                    type="function",
                    language="go",
                    file_path=file_path,
                    line_number=i,
                    docstring="",
                    parameters=[],
                    return_type="",
                    complexity=1,
                    is_public=func_match.group(1)[0].isupper(),
                    is_async=False,
                ))
            
            type_match = re.search(type_pattern, line)
            if type_match:
                elements.append(CodeElement(
                    name=type_match.group(1),
                    type="type",
                    language="go",
                    file_path=file_path,
                    line_number=i,
                    docstring="",
                    parameters=[],
                    return_type="",
                    complexity=1,
                    is_public=type_match.group(1)[0].isupper(),
                    is_async=False,
                ))
        
        return elements
    
    def extract_imports(self, content: str) -> List[str]:
        """Extract Go imports."""
        imports = []
        import_pattern = r'import\s+(?:\(([^)]+)\)|"([^"]+)")'
        
        for match in re.finditer(import_pattern, content, re.MULTILINE | re.DOTALL):
            if match.group(1):  # Multiple imports
                for line in match.group(1).split('\n'):
                    line = line.strip()
                    if line and '"' in line:
                        imports.append(line.split('"')[1])
            elif match.group(2):  # Single import
                imports.append(match.group(2))
        
        return imports
    
    def extract_dependencies(self, content: str) -> List[str]:
        """Extract Go dependencies."""
        return self.extract_imports(content) 