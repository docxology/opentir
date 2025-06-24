"""
Tests for utilities functionality.
"""

import pytest
import asyncio
import tempfile
import hashlib
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from src.utils import (
    Logger, ConfigManager, RepositoryInfo, FileUtils, AsyncUtils, DataUtils,
    format_bytes, format_duration, sanitize_filename
)


class TestLogger:
    """Test suite for Logger class."""
    
    def test_logger_initialization(self):
        """Test logger initialization."""
        logger = Logger("test_logger")
        
        assert logger.name == "test_logger"
        assert logger.logger is not None
        assert logger.logger.name == "test_logger"
    
    def test_logger_methods(self):
        """Test logger methods."""
        logger = Logger("test")
        
        # Test that methods exist and can be called
        logger.info("Test info message")
        logger.debug("Test debug message")
        logger.warning("Test warning message")
        logger.error("Test error message")
        logger.critical("Test critical message")
    
    def test_logger_progress(self):
        """Test logger progress method."""
        logger = Logger("test")
        
        # Test progress method returns a context manager
        progress = logger.progress("Test progress")
        assert hasattr(progress, '__enter__')
        assert hasattr(progress, '__exit__')


class TestConfigManager:
    """Test suite for ConfigManager class."""
    
    @pytest.fixture
    def temp_config_file(self):
        """Create temporary config file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {
                "github_token": "test_token",
                "base_dir": "/test/dir",
                "log_level": "DEBUG"
            }
            import json
            json.dump(config_data, f)
            yield f.name
        
        # Cleanup
        Path(f.name).unlink(missing_ok=True)
    
    def test_config_manager_initialization(self):
        """Test ConfigManager initialization."""
        config_manager = ConfigManager("test_config.json")
        
        assert config_manager.config_file == "test_config.json"
    
    def test_load_config_existing_file(self, temp_config_file):
        """Test loading existing config file."""
        config_manager = ConfigManager(temp_config_file)
        config_data = config_manager.load_config()
        
        assert config_data["github_token"] == "test_token"
        assert config_data["base_dir"] == "/test/dir"
        assert config_data["log_level"] == "DEBUG"
    
    def test_load_config_nonexistent_file(self):
        """Test loading non-existent config file."""
        config_manager = ConfigManager("nonexistent.json")
        config_data = config_manager.load_config()
        
        assert config_data == {}
    
    def test_save_config(self):
        """Test saving config data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_file = f.name
        
        try:
            config_manager = ConfigManager(config_file)
            test_data = {"test_key": "test_value", "number": 42}
            
            config_manager.save_config(test_data)
            
            # Verify file was saved
            assert Path(config_file).exists()
            
            # Verify content
            loaded_data = config_manager.load_config()
            assert loaded_data["test_key"] == "test_value"
            assert loaded_data["number"] == 42
        
        finally:
            Path(config_file).unlink(missing_ok=True)
    
    def test_get_setting(self, temp_config_file):
        """Test getting individual settings."""
        config_manager = ConfigManager(temp_config_file)
        
        # Load config first
        config_manager.load_config()
        
        # Test existing setting
        assert config_manager.get_setting("github_token") == "test_token"
        
        # Test non-existent setting with default
        assert config_manager.get_setting("nonexistent", "default_value") == "default_value"
    
    def test_set_setting(self, temp_config_file):
        """Test setting individual settings."""
        config_manager = ConfigManager(temp_config_file)
        
        # Load existing config
        config_manager.load_config()
        
        # Set new setting
        config_manager.set_setting("new_setting", "new_value")
        
        # Verify it was set
        assert config_manager.get_setting("new_setting") == "new_value"


class TestRepositoryInfo:
    """Test suite for RepositoryInfo dataclass."""
    
    def test_repository_info_creation(self):
        """Test RepositoryInfo creation."""
        repo_info = RepositoryInfo(
            name="test-repo",
            full_name="org/test-repo",
            description="Test repository",
            url="https://github.com/org/test-repo",
            clone_url="https://github.com/org/test-repo.git",
            language="Python",
            stars=100,
            forks=20,
            size=1000,
            topics=["python", "testing"],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            archived=False,
            private=False
        )
        
        assert repo_info.name == "test-repo"
        assert repo_info.full_name == "org/test-repo"
        assert repo_info.language == "Python"
        assert repo_info.stars == 100
        assert repo_info.archived is False


class TestFileUtils:
    """Test suite for FileUtils class."""
    
    @pytest.fixture
    def temp_directory(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def test_ensure_directory(self, temp_directory):
        """Test directory creation."""
        test_dir = temp_directory / "test_subdir" / "nested"
        
        result = FileUtils.ensure_directory(test_dir)
        
        assert result.exists()
        assert result.is_dir()
        assert result == test_dir
    
    def test_ensure_directory_existing(self, temp_directory):
        """Test ensuring existing directory."""
        existing_dir = temp_directory / "existing"
        existing_dir.mkdir()
        
        result = FileUtils.ensure_directory(existing_dir)
        
        assert result.exists()
        assert result == existing_dir
    
    def test_get_file_hash(self, temp_directory):
        """Test file hash calculation."""
        test_file = temp_directory / "test.txt"
        test_content = "test content for hashing"
        test_file.write_text(test_content)
        
        file_hash = FileUtils.get_file_hash(test_file)
        
        # Verify it's a valid MD5 hash
        assert len(file_hash) == 32
        assert all(c in '0123456789abcdef' for c in file_hash)
        
        # Verify consistency
        expected_hash = hashlib.md5(test_content.encode()).hexdigest()
        assert file_hash == expected_hash
    
    def test_get_file_hash_nonexistent(self):
        """Test hash calculation for non-existent file."""
        file_hash = FileUtils.get_file_hash("nonexistent.txt")
        assert file_hash == ""
    
    def test_get_file_size_mb(self, temp_directory):
        """Test file size calculation."""
        test_file = temp_directory / "test.txt"
        test_content = "x" * 1024 * 1024  # 1MB of data
        test_file.write_text(test_content)
        
        size_mb = FileUtils.get_file_size_mb(test_file)
        
        assert isinstance(size_mb, float)
        assert 0.9 < size_mb < 1.1  # Should be approximately 1MB
    
    def test_is_text_file(self, temp_directory):
        """Test text file detection."""
        # Create text file
        text_file = temp_directory / "test.txt"
        text_file.write_text("This is a text file")
        
        # Create binary file
        binary_file = temp_directory / "test.bin"
        binary_file.write_bytes(b'\x00\x01\x02\x03\x04\x05')
        
        assert FileUtils.is_text_file(text_file) is True
        assert FileUtils.is_text_file(binary_file) is False
    
    def test_get_language_from_extension(self):
        """Test language detection from file extension."""
        test_cases = [
            ("test.py", "python"),
            ("script.js", "javascript"),
            ("component.tsx", "typescript"),
            ("Main.java", "java"),
            ("main.go", "go"),
            ("unknown.xyz", None),
        ]
        
        for filename, expected_language in test_cases:
            result = FileUtils.get_language_from_extension(Path(filename))
            assert result == expected_language


class TestAsyncUtils:
    """Test suite for AsyncUtils class."""
    
    @pytest.mark.asyncio
    async def test_run_with_semaphore(self):
        """Test running coroutine with semaphore."""
        semaphore = asyncio.Semaphore(1)
        
        async def test_coro():
            await asyncio.sleep(0.01)
            return "test_result"
        
        result = await AsyncUtils.run_with_semaphore(semaphore, test_coro())
        
        assert result == "test_result"
    
    @pytest.mark.asyncio
    async def test_gather_with_limit(self):
        """Test gathering coroutines with concurrency limit."""
        async def test_coro(value):
            await asyncio.sleep(0.01)
            return value * 2
        
        coroutines = [test_coro(i) for i in range(5)]
        results = await AsyncUtils.gather_with_limit(2, *coroutines)
        
        assert results == [0, 2, 4, 6, 8]
    
    @pytest.mark.asyncio
    async def test_retry_async_success(self):
        """Test async retry with successful function."""
        call_count = 0
        
        async def success_func():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = await AsyncUtils.retry_async(success_func, max_retries=3)
        
        assert result == "success"
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_retry_async_eventual_success(self):
        """Test async retry with eventual success."""
        call_count = 0
        
        async def eventual_success_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return "success"
        
        result = await AsyncUtils.retry_async(eventual_success_func, max_retries=3, delay=0.01)
        
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_retry_async_failure(self):
        """Test async retry with persistent failure."""
        call_count = 0
        
        async def failing_func():
            nonlocal call_count
            call_count += 1
            raise Exception("Persistent failure")
        
        with pytest.raises(Exception) as exc_info:
            await AsyncUtils.retry_async(failing_func, max_retries=2, delay=0.01)
        
        assert "Persistent failure" in str(exc_info.value)
        assert call_count == 3  # Initial call + 2 retries


class TestDataUtils:
    """Test suite for DataUtils class."""
    
    def test_flatten_dict(self):
        """Test dictionary flattening."""
        nested_dict = {
            "a": 1,
            "b": {
                "c": 2,
                "d": {
                    "e": 3
                }
            },
            "f": 4
        }
        
        flattened = DataUtils.flatten_dict(nested_dict)
        
        expected = {
            "a": 1,
            "b.c": 2,
            "b.d.e": 3,
            "f": 4
        }
        
        assert flattened == expected
    
    def test_flatten_dict_custom_separator(self):
        """Test dictionary flattening with custom separator."""
        nested_dict = {
            "a": {"b": {"c": 1}}
        }
        
        flattened = DataUtils.flatten_dict(nested_dict, sep="_")
        
        assert flattened == {"a_b_c": 1}
    
    def test_group_by_key(self):
        """Test grouping items by key."""
        items = [
            {"category": "A", "value": 1},
            {"category": "B", "value": 2},
            {"category": "A", "value": 3},
            {"category": "C", "value": 4},
            {"category": "B", "value": 5}
        ]
        
        grouped = DataUtils.group_by_key(items, "category")
        
        assert len(grouped["A"]) == 2
        assert len(grouped["B"]) == 2
        assert len(grouped["C"]) == 1
        assert grouped["A"][0]["value"] == 1
        assert grouped["A"][1]["value"] == 3
    
    def test_filter_by_criteria(self):
        """Test filtering items by criteria."""
        items = [
            {"name": "item1", "category": "A", "value": 10},
            {"name": "item2", "category": "B", "value": 20},
            {"name": "item3", "category": "A", "value": 30},
            {"name": "item4", "category": "C", "value": 15}
        ]
        
        # Test single criterion
        filtered = DataUtils.filter_by_criteria(items, {"category": "A"})
        assert len(filtered) == 2
        assert all(item["category"] == "A" for item in filtered)
        
        # Test multiple criteria
        filtered = DataUtils.filter_by_criteria(items, {"category": "A", "value": 30})
        assert len(filtered) == 1
        assert filtered[0]["name"] == "item3"
        
        # Test no matches
        filtered = DataUtils.filter_by_criteria(items, {"category": "X"})
        assert len(filtered) == 0


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_format_bytes(self):
        """Test byte formatting."""
        test_cases = [
            (0, "0 B"),
            (1024, "1.0 KB"),
            (1024 * 1024, "1.0 MB"),
            (1024 * 1024 * 1024, "1.0 GB"),
            (1536, "1.5 KB"),  # 1.5 * 1024
        ]
        
        for bytes_value, expected in test_cases:
            result = format_bytes(bytes_value)
            assert result == expected
    
    def test_format_duration(self):
        """Test duration formatting."""
        test_cases = [
            (0, "0s"),
            (30, "30s"),
            (60, "1m 0s"),
            (90, "1m 30s"),
            (3600, "1h 0m 0s"),
            (3661, "1h 1m 1s"),
            (7325, "2h 2m 5s"),
        ]
        
        for seconds, expected in test_cases:
            result = format_duration(seconds)
            assert result == expected
    
    def test_sanitize_filename(self):
        """Test filename sanitization."""
        test_cases = [
            ("normal_file.txt", "normal_file.txt"),
            ("file with spaces.txt", "file_with_spaces.txt"),
            ("file/with\\slashes.txt", "file_with_slashes.txt"),
            ("file:with*special?chars.txt", "file_with_special_chars.txt"),
            ("file<with>pipes|.txt", "file_with_pipes_.txt"),
            ("file\"with'quotes.txt", "file_with_quotes.txt"),
        ]
        
        for input_filename, expected in test_cases:
            result = sanitize_filename(input_filename)
            assert result == expected


if __name__ == "__main__":
    pytest.main([__file__]) 