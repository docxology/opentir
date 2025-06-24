"""
Tests for GitHub client functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from github.GithubException import GithubException
from src.github_client import GitHubClient
from src.utils import RepositoryInfo

class TestGitHubClient:
    """Test suite for GitHubClient class."""
    
    @pytest.fixture
    def mock_github(self):
        """Create a mock GitHub instance."""
        with patch('src.github_client.Github') as mock:
            yield mock
    
    @pytest.fixture
    def github_client(self, mock_github):
        """Create a GitHub client for testing."""
        # Setup mock rate limit
        mock_rate_limit = Mock()
        mock_rate_limit.core.remaining = 1000
        mock_github.return_value.get_rate_limit.return_value = mock_rate_limit
        
        return GitHubClient(token="test_token")
    
    @pytest.fixture
    def mock_repo(self):
        """Create a mock GitHub repository object."""
        repo = Mock()
        repo.name = "test-repo"
        repo.full_name = "palantir/test-repo"
        repo.description = "Test repository"
        repo.html_url = "https://github.com/palantir/test-repo"
        repo.clone_url = "https://github.com/palantir/test-repo.git"
        repo.language = "Python"
        repo.stargazers_count = 100
        repo.forks_count = 20
        repo.size = 1000
        repo.get_topics.return_value = ["python", "testing"]
        repo.created_at = "2020-01-01"
        repo.updated_at = "2023-01-01"
        repo.archived = False
        repo.private = False
        return repo
    
    def test_initialization(self, github_client):
        """Test GitHub client initialization."""
        assert github_client.token == "test_token"
        assert github_client.logger is not None
        assert github_client.github is not None
    
    def test_initialization_without_token(self):
        """Test GitHub client initialization without token."""
        client = GitHubClient()
        assert client.token is None
        assert client.github is not None
    
    def test_get_organization_info(self, mock_github, github_client):
        """Test fetching organization information."""
        # Setup mock
        mock_org = Mock()
        mock_org.login = "palantir"
        mock_org.name = "Palantir Technologies"
        mock_org.description = "Test description"
        mock_org.company = "Palantir"
        mock_org.location = "Test location"
        mock_org.email = "test@palantir.com"
        mock_org.blog = "https://blog.palantir.com"
        mock_org.twitter_username = "palantir"
        mock_org.public_repos = 250
        mock_org.public_gists = 10
        mock_org.followers = 1000
        mock_org.following = 0
        mock_org.html_url = "https://github.com/palantir"
        mock_org.avatar_url = "https://avatars.githubusercontent.com/u/palantir"
        mock_org.type = "Organization"
        mock_org.created_at = "2020-01-01"
        mock_org.updated_at = "2023-01-01"
        
        mock_github.return_value.get_organization.return_value = mock_org
        
        # Test successful case
        org_info = github_client.get_organization_info()
        assert org_info["login"] == "palantir"
        assert org_info["name"] == "Palantir Technologies"
        assert org_info["public_repos"] == 250
        
        # Test authentication error
        mock_github.return_value.get_organization.side_effect = GithubException(401, {"message": "Bad credentials"})
        org_info = github_client.get_organization_info()
        assert org_info["error"] == "authentication_failed"
        
        # Test rate limit error
        mock_github.return_value.get_organization.side_effect = GithubException(403, {"message": "API rate limit exceeded"})
        org_info = github_client.get_organization_info()
        assert org_info["error"] == "rate_limit_exceeded"
    
    def test_get_all_repositories(self, mock_github, github_client, mock_repo):
        """Test fetching all repositories."""
        # Setup mock
        mock_org = Mock()
        mock_repos = Mock()
        mock_repos.totalCount = 1
        mock_repos.__iter__ = Mock(return_value=iter([mock_repo]))
        mock_org.get_repos.return_value = mock_repos
        mock_github.return_value.get_organization.return_value = mock_org
        
        # Test successful case
        repositories = github_client.get_all_repositories()
        assert len(repositories) == 1
        assert isinstance(repositories[0], RepositoryInfo)
        assert repositories[0].name == "test-repo"
        assert repositories[0].language == "Python"
        assert repositories[0].stars == 100
        
        # Test authentication error
        mock_github.return_value.get_organization.side_effect = GithubException(401, {"message": "Bad credentials"})
        repositories = github_client.get_all_repositories()
        assert len(repositories) == 0
        
        # Test rate limit error
        mock_github.return_value.get_organization.side_effect = GithubException(403, {"message": "API rate limit exceeded"})
        repositories = github_client.get_all_repositories()
        assert len(repositories) == 0
    
    def test_extract_repository_info(self, github_client, mock_repo):
        """Test extracting repository information."""
        repo_info = github_client._extract_repository_info(mock_repo)
        
        assert isinstance(repo_info, RepositoryInfo)
        assert repo_info.name == "test-repo"
        assert repo_info.full_name == "palantir/test-repo"
        assert repo_info.language == "Python"
        assert repo_info.stars == 100
        assert repo_info.topics == ["python", "testing"]
    
    @pytest.mark.asyncio
    async def test_async_context_manager(self, github_client):
        """Test async context manager functionality."""
        async with github_client as client:
            assert client.session is not None
        
        # Session should be closed after context
        assert github_client.session is None or github_client.session.closed
    
    @pytest.mark.asyncio
    async def test_get_repository_content_async(self, github_client):
        """Test async repository content fetching."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {"type": "file", "name": "test.py"}
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_get.return_value.__aexit__ = AsyncMock(return_value=None)
            
            async with github_client:
                content = await github_client.get_repository_content_async("test-repo")
                assert content["type"] == "file"
                assert content["name"] == "test.py"
    
    @pytest.mark.asyncio
    async def test_get_repository_languages_async(self, github_client):
        """Test async repository languages fetching."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {"Python": 1000, "JavaScript": 500}
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_get.return_value.__aexit__ = AsyncMock(return_value=None)
            
            async with github_client:
                languages = await github_client.get_repository_languages_async("test-repo")
                assert languages["Python"] == 1000
                assert languages["JavaScript"] == 500
    
    def test_rate_limit_checking(self, github_client):
        """Test rate limit checking functionality."""
        with patch.object(github_client.github, 'get_rate_limit') as mock_rate_limit:
            mock_core = Mock()
            mock_core.remaining = 50  # Above threshold
            mock_rate_limit.return_value.core = mock_core
            
            # Should not sleep
            github_client._check_rate_limit()
            
            # Test low rate limit
            mock_core.remaining = 5  # Below threshold
            mock_core.reset.timestamp.return_value = 1234567890
            
            with patch('time.sleep') as mock_sleep, patch('time.time', return_value=1234567880):
                github_client._check_rate_limit()
                mock_sleep.assert_called_once()
    
    def test_get_repository_detailed_info(self, mock_github, github_client):
        """Test getting detailed repository information."""
        # Setup mocks
        mock_repo = Mock()
        mock_repo.name = "test-repo"
        mock_repo.description = "Test repository"
        mock_repo.html_url = "https://github.com/palantir/test-repo"
        mock_repo.clone_url = "https://github.com/palantir/test-repo.git"
        mock_repo.language = "Python"
        mock_repo.stargazers_count = 100
        mock_repo.forks_count = 20
        mock_repo.size = 1000
        mock_repo.get_topics.return_value = ["python", "testing"]
        mock_repo.created_at = "2020-01-01"
        mock_repo.updated_at = "2023-01-01"
        mock_repo.archived = False
        mock_repo.private = False
        mock_repo.get_readme.return_value.decoded_content = b"# Test Repo"
        mock_repo.license = None
        mock_repo.get_contents.side_effect = Exception("File not found")
        
        mock_github.return_value.get_repo.return_value = mock_repo
        
        # Test successful case
        detailed_info = github_client.get_repository_detailed_info("test-repo")
        assert "basic_info" in detailed_info
        assert "readme" in detailed_info
        assert detailed_info["readme"] == "# Test Repo"
        
        # Test authentication error
        mock_github.return_value.get_repo.side_effect = GithubException(401, {"message": "Bad credentials"})
        detailed_info = github_client.get_repository_detailed_info("test-repo")
        assert "error" in detailed_info
        assert detailed_info["error"] == "authentication_failed"
    
    def test_parse_dependencies_python(self, github_client):
        """Test parsing Python dependencies."""
        requirements_content = """
        requests>=2.25.0
        pandas==1.3.0
        # Comment line
        numpy
        """
        
        dependencies = github_client._parse_dependencies("requirements.txt", requirements_content)
        assert "requests" in dependencies
        assert "pandas" in dependencies
        assert "numpy" in dependencies
        assert len(dependencies) == 3
    
    @pytest.mark.asyncio
    async def test_fetch_all_repositories_with_details(self, github_client):
        """Test comprehensive repository fetching."""
        # Mock basic repository fetch
        with patch.object(github_client, 'get_all_repositories') as mock_get_repos:
            mock_repo_info = RepositoryInfo(
                name="test-repo",
                full_name="palantir/test-repo",
                description="Test",
                url="https://github.com/palantir/test-repo",
                clone_url="https://github.com/palantir/test-repo.git",
                language="Python",
                stars=100,
                forks=20,
                size=1000,
                topics=["test"],
                created_at="2020-01-01",
                updated_at="2023-01-01",
                archived=False,
                private=False
            )
            mock_get_repos.return_value = [mock_repo_info]
            
            # Mock async methods
            async def mock_fetch_details(semaphore, repo_name):
                return {
                    "languages": {"Python": 1000},
                    "contributors": [],
                    "releases": [],
                    "content": {}
                }
            
            with patch.object(github_client, '_fetch_repository_comprehensive_details', side_effect=mock_fetch_details):
                results = await github_client.fetch_all_repositories_with_details()
                assert len(results) == 1
                assert results[0]["basic_info"] == mock_repo_info
                assert "detailed_info" in results[0] 