"""
Utility functions and classes for Opentir package.
Provides logging, configuration management, and common helper functions.
"""

import os
import json
import logging
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from dataclasses import dataclass
import hashlib

from .config import config, ensure_directories

class Logger:
    """Enhanced logging utility with Rich formatting and file output."""
    
    def __init__(self, name: str = "opentir"):
        self.name = name
        self.console = Console()
        self._setup_logger()
    
    def _setup_logger(self) -> None:
        """Set up logging with both console and file handlers."""
        # Create logger
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(getattr(logging, config.log_level.upper()))
        
        # Clear existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            handler.close()
        
        # Console handler with Rich formatting
        console_handler = RichHandler(
            console=self.console,
            show_time=True,
            show_path=False,
            rich_tracebacks=True
        )
        console_handler.setLevel(logging.INFO)
        
        # Formatters
        console_formatter = logging.Formatter("%(message)s")
        
        console_handler.setFormatter(console_formatter)
        
        # Add handlers
        self.logger.addHandler(console_handler)

    def _log(self, level: int, message: str, **kwargs) -> None:
        if not any(isinstance(handler, logging.FileHandler) for handler in self.logger.handlers):
            log_directory = config.base_dir / config.logs_dir
            log_directory.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(log_directory / f"{self.name}.log", encoding="utf-8")
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(logging.Formatter(config.log_format))
            self.logger.addHandler(handler)
        self.logger.log(level, message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self._log(logging.INFO, message, **kwargs)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message.""" 
        self._log(logging.DEBUG, message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self._log(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        self._log(logging.CRITICAL, message, **kwargs)
    
    def progress(self, description: str = "Processing..."):
        """Create a progress bar context manager."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=self.console
        )

class ConfigManager:
    """Configuration management utility."""
    
    def __init__(self, config_file: str = "opentir_config.json"):
        self.config_file = config_file
        self.logger = Logger("config_manager")
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading config: {e}")
                return {}
        return {}
    
    def save_config(self, config_data: Dict[str, Any]) -> None:
        """Save configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2, default=str)
            self.logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            self.logger.error(f"Error saving config: {e}")
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a specific setting value."""
        config_data = self.load_config()
        return config_data.get(key, default)
    
    def set_setting(self, key: str, value: Any) -> None:
        """Set a specific setting value."""
        config_data = self.load_config()
        config_data[key] = value
        self.save_config(config_data)

@dataclass
class RepositoryInfo:
    """Data class for repository information."""
    name: str
    full_name: str
    description: str
    url: str
    clone_url: str
    language: str
    stars: int
    forks: int
    size: int
    topics: List[str]
    created_at: datetime
    updated_at: datetime
    archived: bool
    private: bool

class FileUtils:
    """File system utility functions."""
    
    @staticmethod
    def ensure_directory(path: Union[str, Path]) -> Path:
        """Ensure directory exists and return Path object."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    def get_file_hash(file_path: Union[str, Path]) -> str:
        """Get MD5 hash of file content."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    @staticmethod
    def get_file_size_mb(file_path: Union[str, Path]) -> float:
        """Get file size in MB."""
        return os.path.getsize(file_path) / (1024 * 1024)
    
    @staticmethod
    def is_text_file(file_path: Union[str, Path]) -> bool:
        """Check if file is a text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if '\x00' in f.read(1024):
                    return False
            return True
        except (UnicodeDecodeError, IOError):
            return False
    
    @staticmethod
    def get_language_from_extension(file_path: Union[str, Path]) -> Optional[str]:
        """Get programming language from file extension."""
        from .config import SUPPORTED_FILE_EXTENSIONS
        suffix = Path(file_path).suffix.lower()
        return SUPPORTED_FILE_EXTENSIONS.get(suffix)

class AsyncUtils:
    """Async utility functions."""
    
    @staticmethod
    async def run_with_semaphore(semaphore: asyncio.Semaphore, coro):
        """Run coroutine with semaphore for rate limiting."""
        async with semaphore:
            return await coro
    
    @staticmethod
    async def gather_with_limit(limit: int, *coroutines):
        """Gather coroutines with concurrency limit."""
        semaphore = asyncio.Semaphore(limit)
        return await asyncio.gather(*[
            AsyncUtils.run_with_semaphore(semaphore, coro)
            for coro in coroutines
        ])
    
    @staticmethod
    async def retry_async(func, max_retries: int = 3, delay: float = 1.0, *args, **kwargs):
        """Retry async function with exponential backoff."""
        if max_retries < 0:
            raise ValueError("max_retries must be nonnegative")
        for attempt in range(max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries:
                    raise e
                await asyncio.sleep(delay * (2 ** attempt))

class DataUtils:
    """Data processing utility functions."""
    
    @staticmethod
    def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(DataUtils.flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    @staticmethod
    def group_by_key(items: List[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
        """Group list of dictionaries by specified key."""
        groups = {}
        for item in items:
            group_key = item.get(key, 'unknown')
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(item)
        return groups
    
    @staticmethod
    def filter_by_criteria(items: List[Dict[str, Any]], criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter list of dictionaries by criteria."""
        filtered = []
        for item in items:
            match = True
            for key, value in criteria.items():
                if key not in item or item[key] != value:
                    match = False
                    break
            if match:
                filtered.append(item)
        return filtered

def format_bytes(bytes_value: int) -> str:
    """Format bytes into human readable string."""
    if bytes_value < 1024:
        return f"{bytes_value} B"
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"

def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable string."""
    minutes, remainder = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {remainder}s"
    if minutes:
        return f"{minutes}m {remainder}s"
    return f"{remainder}s"

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for cross-platform compatibility."""
    invalid_chars = '<>:"/\\|?*\' '
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename[:255]  # Limit filename length

# Global logger instance
logger = Logger()
