"""
Tests for repository manager functionality.
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from src.repo_manager import RepositoryManager
from src.github_client import GitHubClient
from src.utils import RepositoryInfo


class TestRepositoryManager:
    """Test suite for RepositoryManager class."""
    
    @pytest.fixture
    def temp_base_dir(self):
        """Create temporary base directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def repo_manager(self, temp_base_dir):
        """Create repository manager for testing."""
        return RepositoryManager(temp_base_dir)
    
    @pytest.fixture
    def mock_repo_info(self):
        """Create mock repository info for testing."""
        from datetime import datetime
        return RepositoryInfo(
            name="test-repo",
            full_name="palantir/test-repo",
            description="Test repository",
            url="https://github.com/palantir/test-repo",
            clone_url="https://github.com/palantir/test-repo.git",
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
    
    @pytest.fixture
    def mock_github_client(self):
        """Create mock GitHub client."""
        mock_client = Mock(spec=GitHubClient)
        mock_client.get_all_repositories.return_value = []
        return mock_client
    
    def test_initialization(self, temp_base_dir):
        """Test RepositoryManager initialization."""
        repo_manager = RepositoryManager(temp_base_dir)
        
        assert repo_manager.base_dir == temp_base_dir
        assert repo_manager.repos_dir == temp_base_dir / "repos"
        assert repo_manager.logger is not None
        assert repo_manager.github_client is not None
    
    def test_setup_repository_structure(self, repo_manager):
        """Test repository structure setup."""
        repo_manager.setup_repository_structure()
        
        expected_dirs = [
            "all_repos",
            "by_language",
            "by_category", 
            "popular",
            "archived"
        ]
        
        for dir_name in expected_dirs:
            assert (repo_manager.repos_dir / dir_name).exists()
    
    def test_get_fast_local_repository_status(self, repo_manager):
        """Test fast local repository status check."""
        # Setup some test directories
        repo_manager.setup_repository_structure()
        test_repo_dir = repo_manager.repos_dir / "all_repos" / "test-repo"
        test_repo_dir.mkdir(parents=True)
        
        status = repo_manager._get_fast_local_repository_status()
        
        assert "repos_dir_exists" in status
        assert "all_repos_dir_exists" in status
        assert "local_repo_count" in status
        assert "local_repo_names" in status
        
        assert status["repos_dir_exists"] is True
        assert status["all_repos_dir_exists"] is True
        assert status["local_repo_count"] == 1
        assert "test-repo" in status["local_repo_names"]
    
    def test_get_repo_size_mb(self, repo_manager):
        """Test repository size calculation."""
        # Create a test file
        test_repo_dir = repo_manager.repos_dir / "test-repo"
        test_repo_dir.mkdir(parents=True)
        test_file = test_repo_dir / "test.txt"
        test_file.write_text("test content" * 1000)  # Create some content
        
        size_mb = repo_manager._get_repo_size_mb(test_repo_dir)
        
        assert isinstance(size_mb, float)
        assert size_mb > 0
    
    def test_detect_primary_language(self, repo_manager):
        """Test primary language detection."""
        # Create test repository with Python files
        test_repo_dir = repo_manager.repos_dir / "test-repo"
        test_repo_dir.mkdir(parents=True)
        
        # Create Python files
        (test_repo_dir / "main.py").write_text("print('hello')")
        (test_repo_dir / "utils.py").write_text("def utility(): pass")
        
        # Create one JavaScript file
        (test_repo_dir / "script.js").write_text("console.log('test')")
        
        language = repo_manager._detect_primary_language(test_repo_dir)
        
        assert language == "python"  # Should detect Python as primary
    
    @patch('git.Repo.clone_from')
    def test_clone_repository_sync_success(self, mock_clone, repo_manager, mock_repo_info):
        """Test successful repository cloning."""
        mock_repo = Mock()
        mock_clone.return_value = mock_repo
        
        result = repo_manager._clone_repository_sync(mock_repo_info, force_update=False)
        
        assert result["success"] is True
        assert result["repo_name"] == "test-repo"
        assert "clone_path" in result
        assert mock_clone.called
    
    @patch('git.Repo.clone_from')
    def test_clone_repository_sync_failure(self, mock_clone, repo_manager, mock_repo_info):
        """Test repository cloning failure."""
        mock_clone.side_effect = Exception("Clone failed")
        
        result = repo_manager._clone_repository_sync(mock_repo_info, force_update=False)
        
        assert result["success"] is False
        assert result["repo_name"] == "test-repo"
        assert "error" in result
    
    @patch('git.Repo.clone_from')
    def test_clone_repository_force_update(self, mock_clone, repo_manager, mock_repo_info):
        """Test repository cloning with force update."""
        # Create existing repository
        existing_repo_dir = repo_manager.repos_dir / "all_repos" / "test-repo"
        existing_repo_dir.mkdir(parents=True)
        (existing_repo_dir / "existing_file.txt").write_text("existing content")
        
        mock_repo = Mock()
        mock_clone.return_value = mock_repo
        
        result = repo_manager._clone_repository_sync(mock_repo_info, force_update=True)
        
        assert result["success"] is True
        assert mock_clone.called
    
    @pytest.mark.asyncio
    async def test_clone_single_repository(self, repo_manager, mock_repo_info):
        """Test single repository cloning with semaphore."""
        semaphore = asyncio.Semaphore(1)
        
        with patch.object(repo_manager, '_clone_repository_sync') as mock_clone_sync:
            mock_clone_sync.return_value = {"success": True, "repo_name": "test-repo"}
            
            result = await repo_manager._clone_single_repository(semaphore, mock_repo_info, False)
            
            assert result["success"] is True
            assert mock_clone_sync.called
    
    def test_analyze_repository_status(self, repo_manager):
        """Test repository status analysis."""
        available_repos = [
            Mock(name="repo1"),
            Mock(name="repo2"),
            Mock(name="repo3")
        ]
        
        local_status = {
            "local_repo_names": ["repo1", "repo3"],
            "local_repo_count": 2
        }
        
        analysis = repo_manager._analyze_repository_status(available_repos, local_status)
        
        assert "total_available_repos" in analysis
        assert "repos_to_clone" in analysis
        assert "repos_already_local" in analysis
        assert "new_repos_needed" in analysis
        
        assert analysis["total_available_repos"] == 3
        assert analysis["repos_already_local"] == 2
        assert analysis["new_repos_needed"] == 1
    
    @pytest.mark.asyncio
    async def test_organize_repositories(self, repo_manager):
        """Test repository organization."""
        # Setup successful repos data
        successful_repos = [
            {
                "repo_name": "python-repo",
                "clone_path": str(repo_manager.repos_dir / "all_repos" / "python-repo"),
                "primary_language": "python",
                "stars": 1500,
                "archived": False
            },
            {
                "repo_name": "java-repo", 
                "clone_path": str(repo_manager.repos_dir / "all_repos" / "java-repo"),
                "primary_language": "java",
                "stars": 500,
                "archived": False
            },
            {
                "repo_name": "archived-repo",
                "clone_path": str(repo_manager.repos_dir / "all_repos" / "archived-repo"),
                "primary_language": "python",
                "stars": 100,
                "archived": True
            }
        ]
        
        # Create the actual directories for linking
        for repo in successful_repos:
            Path(repo["clone_path"]).mkdir(parents=True, exist_ok=True)
        
        await repo_manager._organize_repositories(successful_repos)
        
        # Check language organization
        python_dir = repo_manager.repos_dir / "by_language" / "python"
        java_dir = repo_manager.repos_dir / "by_language" / "java"
        
        assert python_dir.exists()
        assert java_dir.exists()
        
        # Check popular organization
        popular_dir = repo_manager.repos_dir / "popular"
        assert popular_dir.exists()
        
        # Check archived organization
        archived_dir = repo_manager.repos_dir / "archived"
        assert archived_dir.exists()
    
    def test_get_local_repository_stats(self, repo_manager):
        """Test local repository statistics."""
        # Setup test repositories
        repo_manager.setup_repository_structure()
        
        # Create test repos
        all_repos_dir = repo_manager.repos_dir / "all_repos"
        (all_repos_dir / "repo1").mkdir()
        (all_repos_dir / "repo2").mkdir()
        (all_repos_dir / "repo3").mkdir()
        
        # Create language organization
        python_dir = repo_manager.repos_dir / "by_language" / "python"
        python_dir.mkdir(parents=True)
        
        stats = repo_manager.get_local_repository_stats()
        
        assert "total_repositories" in stats
        assert "organization_structure" in stats
        assert "total_size_mb" in stats
        assert "languages" in stats
        
        assert stats["total_repositories"] == 3
        assert stats["organization_structure"]["by_language"] is True
    
    @pytest.mark.asyncio
    @patch('src.repo_manager.RepositoryManager._clone_single_repository')
    async def test_clone_all_repositories_success(self, mock_clone_single, repo_manager):
        """Test successful cloning of all repositories."""
        # Mock GitHub client
        mock_repo_info = Mock()
        mock_repo_info.name = "test-repo"
        
        repo_manager.github_client = Mock()
        repo_manager.github_client.get_all_repositories.return_value = [mock_repo_info]
        
        # Mock successful clone results
        mock_clone_single.return_value = {
            "success": True,
            "repo_name": "test-repo",
            "clone_path": "/test/path",
            "size_mb": 5.0,
            "primary_language": "python",
            "stars": 100,
            "archived": False
        }
        
        result = await repo_manager.clone_all_repositories()
        
        assert result["success"] is True
        assert result["total_repositories"] == 1
        assert result["successful_clones"] == 1
        assert result["failed_clones"] == 0
        assert "total_size_mb" in result
    
    @pytest.mark.asyncio
    async def test_clone_all_repositories_github_error(self, repo_manager):
        """Test handling GitHub API errors."""
        # Mock GitHub client to raise exception
        repo_manager.github_client = Mock()
        repo_manager.github_client.get_all_repositories.side_effect = Exception("GitHub API error")
        
        result = await repo_manager.clone_all_repositories()
        
        assert result["success"] is False
        assert "error" in result
    
    def test_cleanup_repositories_keep_popular(self, repo_manager):
        """Test repository cleanup while keeping popular ones."""
        # Setup test repositories
        repo_manager.setup_repository_structure()
        all_repos_dir = repo_manager.repos_dir / "all_repos"
        
        # Create test repos
        popular_repo = all_repos_dir / "popular-repo"
        unpopular_repo = all_repos_dir / "unpopular-repo"
        popular_repo.mkdir()
        unpopular_repo.mkdir()
        
        # Create files to simulate repo content
        (popular_repo / "test.txt").write_text("content")
        (unpopular_repo / "test.txt").write_text("content")
        
        # Mock the popular directory with a link to popular repo
        popular_dir = repo_manager.repos_dir / "popular"
        popular_dir.mkdir(exist_ok=True)
        
        result = repo_manager.cleanup_repositories(keep_popular=True)
        
        assert "removed_repositories" in result
        assert "total_size_freed_mb" in result
        assert isinstance(result["removed_repositories"], int)
    
    @pytest.mark.asyncio
    async def test_update_all_repositories(self, repo_manager):
        """Test updating all repositories."""
        # Setup existing repositories
        repo_manager.setup_repository_structure()
        all_repos_dir = repo_manager.repos_dir / "all_repos"
        test_repo = all_repos_dir / "test-repo"
        test_repo.mkdir()
        
        with patch('git.Repo') as mock_git_repo:
            mock_repo = Mock()
            mock_origin = Mock()
            mock_repo.remotes.origin = mock_origin
            mock_git_repo.return_value = mock_repo
            
            result = await repo_manager.update_all_repositories()
            
            assert "total_repositories" in result
            assert "successful_updates" in result
            assert "failed_updates" in result


class TestRepositoryManagerIntegration:
    """Integration tests for RepositoryManager."""
    
    @pytest.fixture
    def integration_repo_manager(self):
        """Create repo manager for integration tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield RepositoryManager(Path(temp_dir))
    
    def test_full_repository_workflow(self, integration_repo_manager):
        """Test complete repository management workflow."""
        repo_manager = integration_repo_manager
        
        # Step 1: Setup structure
        repo_manager.setup_repository_structure()
        
        # Step 2: Get initial status
        initial_status = repo_manager._get_fast_local_repository_status()
        assert initial_status["local_repo_count"] == 0
        
        # Step 3: Get stats
        stats = repo_manager.get_local_repository_stats()
        assert stats["total_repositories"] == 0
        
        # Verify structure was created correctly
        assert repo_manager.repos_dir.exists()
        assert (repo_manager.repos_dir / "all_repos").exists()
        assert (repo_manager.repos_dir / "by_language").exists()


if __name__ == "__main__":
    pytest.main([__file__]) 