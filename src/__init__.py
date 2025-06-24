"""
Opentir - A comprehensive toolkit for working with Palantir's open source ecosystem.

This package provides tools for:
- Fetching and managing Palantir repositories
- Generating comprehensive documentation
- Analyzing code functionality and methods
- Creating organized project structures
"""

__version__ = "1.0.0"
__author__ = "Opentir Contributors"
__license__ = "Apache 2.0"

# Core modules
from .github_client import GitHubClient
from .repo_manager import RepositoryManager
from .docs_generator import DocumentationGenerator
from .code_analyzer import CodeAnalyzer
from .utils import Logger, ConfigManager

# Main classes for easy import
__all__ = [
    "GitHubClient",
    "RepositoryManager", 
    "DocumentationGenerator",
    "CodeAnalyzer",
    "Logger",
    "ConfigManager",
] 