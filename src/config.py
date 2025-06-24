"""
Configuration module for Opentir package.
Manages settings, constants, and configuration loading.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import Field
from pydantic_settings import BaseSettings
import json

class OpentirConfig(BaseSettings):
    """Main configuration class for Opentir package."""
    
    # GitHub API settings
    github_token: Optional[str] = Field(None, env="GITHUB_TOKEN")
    github_org: str = "palantir"
    github_api_url: str = "https://api.github.com"
    
    # Rate limiting settings
    requests_per_hour: int = 5000
    max_concurrent_requests: int = 10
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    # Directory structure settings
    base_dir: Path = Path.cwd()
    repos_dir: str = "repos"
    docs_dir: str = "docs"
    cache_dir: str = ".cache"
    logs_dir: str = "logs"
    
    # Documentation settings
    docs_theme: str = "material"
    docs_site_name: str = "Opentir - Palantir OSS Documentation"
    docs_site_description: str = "Comprehensive documentation for Palantir's open source ecosystem"
    
    # Code analysis settings
    supported_languages: list = [
        "python", "javascript", "typescript", "java", "go", 
        "rust", "scala", "shell", "groovy", "css"
    ]
    analysis_depth: str = "comprehensive"  # basic, standard, comprehensive
    include_tests: bool = True
    include_examples: bool = True
    
    # Logging settings
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Global configuration instance
config = OpentirConfig()

# Constants
PALANTIR_ORG = "palantir"
REPO_BATCH_SIZE = 50
MAX_FILE_SIZE_MB = 10
SUPPORTED_FILE_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript", 
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".scala": "scala",
    ".sc": "scala",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".groovy": "groovy",
    ".gradle": "groovy",
    ".css": "css",
    ".scss": "css",
    ".less": "css",
}

# Documentation templates and structure
DOCS_TEMPLATE_STRUCTURE = {
    "index.md": "main_index",
    "repositories/": {
        "index.md": "repos_index",
        "by_language/": "language_index",
        "by_category/": "category_index",
        "popular/": "popular_repos",
    },
    "api_reference/": {
        "index.md": "api_index",
        "methods/": "methods_docs",
        "classes/": "classes_docs",
        "modules/": "modules_docs",
    },
    "guides/": {
        "getting_started.md": "getting_started",
        "contributing.md": "contributing",
        "examples.md": "examples",
    },
    "analysis/": {
        "functionality_matrix.md": "functionality_matrix",
        "dependency_graph.md": "dependency_graph",
        "statistics.md": "statistics",
    }
}

def get_config_dict() -> Dict[str, Any]:
    """Get configuration as dictionary for easy access."""
    return config.dict()

def update_config(**kwargs) -> None:
    """Update configuration values."""
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

def ensure_directories() -> None:
    """Ensure all required directories exist."""
    directories = [
        config.base_dir / config.repos_dir,
        config.base_dir / config.docs_dir,
        config.base_dir / config.cache_dir,
        config.base_dir / config.logs_dir,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def load_config_from_file(config_path: str) -> None:
    """Load configuration from JSON file."""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_data = json.load(f)
            update_config(**config_data)

def save_config_to_file(config_path: str) -> None:
    """Save current configuration to JSON file."""
    config_data = get_config_dict()
    # Convert Path objects to strings for JSON serialization
    for key, value in config_data.items():
        if isinstance(value, Path):
            config_data[key] = str(value)
    
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2) 