"""
Tests for main orchestrator functionality.
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime

from src.main import OpentirOrchestrator, build_complete_ecosystem, get_workspace_status
from src.code_analyzer import RepositoryAnalysis
from src.utils import RepositoryInfo


class TestOpentirOrchestrator:
    """Test suite for OpentirOrchestrator class."""
    
    @pytest.fixture
    def temp_base_dir(self):
        """Create temporary base directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def orchestrator(self, temp_base_dir):
        """Create orchestrator for testing."""
        return OpentirOrchestrator(base_dir=temp_base_dir, github_token="test_token")
    
    @pytest.fixture
    def mock_analysis(self):
        """Create mock repository analysis."""
        mock_analysis = Mock(spec=RepositoryAnalysis)
        mock_analysis.repo_name = "test-repo"
        mock_analysis.total_files = 10
        mock_analysis.total_lines = 1000
        mock_analysis.languages = {"python": 8, "javascript": 2}
        mock_analysis.all_elements = [Mock(), Mock(), Mock()]  # 3 elements
        mock_analysis.functionality_summary = {"total_elements": 3}
        mock_analysis.metrics = {"complexity": 50}
        return mock_analysis
    
    def test_initialization(self, temp_base_dir):
        """Test OpentirOrchestrator initialization."""
        orchestrator = OpentirOrchestrator(base_dir=temp_base_dir, github_token="test_token")
        
        assert orchestrator.base_dir == temp_base_dir
        assert orchestrator.github_token == "test_token"
        assert orchestrator.logger is not None
        assert orchestrator.github_client is not None
        assert orchestrator.repo_manager is not None
        assert orchestrator.code_analyzer is not None
        assert orchestrator.docs_generator is not None
        assert "started_at" in orchestrator.execution_state
    
    @pytest.mark.asyncio
    async def test_execute_step_success(self, orchestrator):
        """Test successful step execution."""
        def test_function():
            return {"result": "success"}
        
        result = await orchestrator._execute_step("test_step", test_function)
        
        assert result == {"result": "success"}
        assert "test_step" in orchestrator.execution_state["steps_completed"]
    
    @pytest.mark.asyncio
    async def test_execute_step_failure(self, orchestrator):
        """Test step execution failure."""
        def failing_function():
            raise Exception("Test error")
        
        with pytest.raises(Exception):
            await orchestrator._execute_step("failing_step", failing_function)
        
        assert "failing_step" not in orchestrator.execution_state["steps_completed"]
        assert len(orchestrator.execution_state["errors"]) > 0
    
    @pytest.mark.asyncio
    async def test_fetch_organization_info(self, orchestrator):
        """Test organization info fetching."""
        # Mock GitHub client
        orchestrator.github_client.get_organization_info = Mock(return_value={
            "name": "Test Org",
            "login": "test-org"
        })
        
        result = await orchestrator._fetch_organization_info()
        
        assert result["name"] == "Test Org"
        assert result["login"] == "test-org"
    
    @pytest.mark.asyncio
    async def test_fetch_repository_list(self, orchestrator):
        """Test repository list fetching."""
        # Mock repository info
        mock_repo = Mock(spec=RepositoryInfo)
        mock_repo.language = "Python"
        mock_repo.stars = 100
        mock_repo.archived = False
        
        orchestrator.github_client.get_all_repositories = Mock(return_value=[mock_repo])
        
        result = await orchestrator._fetch_repository_list()
        
        assert "repositories" in result
        assert "total_count" in result
        assert "by_language" in result
        assert "by_activity" in result
        assert result["total_count"] == 1
    
    def test_group_repos_by_language(self, orchestrator):
        """Test repository grouping by language."""
        mock_repos = [
            Mock(language="Python"),
            Mock(language="Python"),
            Mock(language="Java"),
            Mock(language=None),  # Unknown language
        ]
        
        result = orchestrator._group_repos_by_language(mock_repos)
        
        assert result["Python"] == 2
        assert result["Java"] == 1
        assert result["Unknown"] == 1
    
    def test_categorize_repos_by_activity(self, orchestrator):
        """Test repository categorization by activity."""
        mock_repos = [
            Mock(archived=True, stars=100),      # archived
            Mock(archived=False, stars=1500),    # high
            Mock(archived=False, stars=500),     # medium
            Mock(archived=False, stars=50),      # low
        ]
        
        result = orchestrator._categorize_repos_by_activity(mock_repos)
        
        assert result["archived"] == 1
        assert result["high"] == 1
        assert result["medium"] == 1
        assert result["low"] == 1
    
    @pytest.mark.asyncio
    async def test_analyze_all_repositories(self, orchestrator, mock_analysis):
        """Test repository analysis."""
        # Setup mock repos directory
        repos_dir = orchestrator.base_dir / "repos" / "all_repos"
        repos_dir.mkdir(parents=True)
        test_repo_dir = repos_dir / "test-repo"
        test_repo_dir.mkdir()
        
        # Mock code analyzer
        orchestrator.code_analyzer.analyze_repository = Mock(return_value=mock_analysis)
        orchestrator.code_analyzer.generate_functionality_matrix = Mock(return_value={"test": "matrix"})
        
        result = await orchestrator._analyze_all_repositories()
        
        assert result["success"] is True
        assert result["total_repositories_analyzed"] == 1
        assert result["total_elements_extracted"] == 3
        assert "functionality_matrix" in result
        
        # Check that analysis files were created
        analysis_dir = orchestrator.base_dir / "analysis_results"
        assert analysis_dir.exists()
        assert (analysis_dir / "repository_analyses.json").exists()
        assert (analysis_dir / "functionality_matrix.json").exists()
    
    def test_serialize_analysis(self, orchestrator, mock_analysis):
        """Test analysis serialization."""
        result = orchestrator._serialize_analysis(mock_analysis)
        
        assert result["repo_name"] == "test-repo"
        assert result["total_files"] == 10
        assert result["total_lines"] == 1000
        assert result["element_count"] == 3
        assert "functionality_summary" in result
        assert "metrics" in result
    
    @pytest.mark.asyncio
    async def test_generate_comprehensive_docs(self, orchestrator):
        """Test documentation generation."""
        analysis_results = {
            "success": True,
            "analyses": [Mock()],
            "functionality_matrix": {"test": "matrix"}
        }
        org_info = {"name": "Test Org"}
        
        # Mock docs generator
        orchestrator.docs_generator.generate_complete_documentation = Mock(return_value={
            "main_index": "index.md",
            "api_reference": "api.md"
        })
        
        result = await orchestrator._generate_comprehensive_docs(analysis_results, org_info)
        
        assert result["success"] is True
        assert "documentation_components" in result
        assert "docs_location" in result
    
    def test_generate_workflow_summary(self, orchestrator):
        """Test workflow summary generation."""
        steps = {
            "clone_repos": {
                "total_repositories": 10,
                "successful_clones": 8
            },
            "analyze_code": {
                "total_repositories_analyzed": 8,
                "total_elements_extracted": 1500
            },
            "generate_docs": {
                "success": True
            }
        }
        
        summary = orchestrator._generate_workflow_summary(steps)
        
        assert summary["steps_completed"] == 3
        assert summary["total_repositories"] == 10
        assert summary["successful_clones"] == 8
        assert summary["repositories_analyzed"] == 8
        assert summary["total_elements_extracted"] == 1500
        assert summary["documentation_generated"] is True
    
    def test_get_workspace_status(self, orchestrator):
        """Test workspace status retrieval."""
        # Create some test structure
        repos_dir = orchestrator.base_dir / "repos" / "all_repos"
        repos_dir.mkdir(parents=True)
        (repos_dir / "test-repo").mkdir()
        
        docs_dir = orchestrator.base_dir / "docs"
        docs_dir.mkdir()
        (docs_dir / "index.md").write_text("# Test")
        (docs_dir / "mkdocs.yml").write_text("site_name: Test")
        
        status = orchestrator.get_workspace_status()
        
        assert status["workspace_initialized"] is True
        assert "repositories" in status
        assert "documentation" in status
        assert "configuration" in status
        assert status["repositories"]["count"] == 1
        assert status["documentation"]["status"] == "generated"
    
    def test_get_repository_status(self, orchestrator):
        """Test repository status check."""
        # Create test repository structure
        repos_dir = orchestrator.base_dir / "repos"
        all_repos_dir = repos_dir / "all_repos"
        all_repos_dir.mkdir(parents=True)
        
        # Create test repos
        (all_repos_dir / "repo1").mkdir()
        (all_repos_dir / "repo2").mkdir()
        
        # Create organization structure
        (repos_dir / "by_language").mkdir()
        (repos_dir / "by_category").mkdir()
        
        status = orchestrator._get_repository_status()
        
        assert status["status"] == "cloned"
        assert status["count"] == 2
        assert status["organization"]["by_language"] is True
        assert status["organization"]["by_category"] is True
    
    def test_get_analysis_status_completed(self, orchestrator):
        """Test analysis status when completed."""
        # Create analysis results
        analysis_dir = orchestrator.base_dir / "analysis_results"
        analysis_dir.mkdir()
        
        analyses_data = [{"repo_name": "test", "element_count": 10}]
        with open(analysis_dir / "repository_analyses.json", 'w') as f:
            json.dump(analyses_data, f)
        
        with open(analysis_dir / "functionality_matrix.json", 'w') as f:
            json.dump({"test": "matrix"}, f)
        
        status = orchestrator._get_analysis_status()
        
        assert status["status"] == "completed"
        assert status["repositories_analyzed"] == 1
        assert status["total_elements"] == 10
    
    def test_get_documentation_status_generated(self, orchestrator):
        """Test documentation status when generated."""
        docs_dir = orchestrator.base_dir / "docs"
        docs_dir.mkdir()
        (docs_dir / "index.md").write_text("# Test")
        (docs_dir / "mkdocs.yml").write_text("site_name: Test")
        
        status = orchestrator._get_documentation_status()
        
        assert status["status"] == "generated"
        assert "location" in status
        assert "components" in status
    
    def test_get_configuration_status(self, orchestrator):
        """Test configuration status."""
        status = orchestrator._get_configuration_status()
        
        assert status["github_token_set"] is True
        assert "base_directory" in status
        assert "log_level" in status
        assert "supported_languages" in status
    
    @pytest.mark.asyncio
    @patch('src.main.OpentirOrchestrator._execute_step')
    async def test_execute_complete_workflow_success(self, mock_execute_step, orchestrator):
        """Test successful complete workflow execution."""
        # Mock all step executions
        mock_execute_step.side_effect = [
            {"name": "Test Org"},  # org_info
            {"repositories": [], "total_count": 0},  # repo_info
            {"success": True, "total_repositories": 0, "successful_clones": 0, "failed_clones": 0},  # clone_results
            {"success": True, "analyses": [], "functionality_matrix": {}},  # analysis_results
            {"success": True, "documentation_components": 5}  # docs_results
        ]
        
        result = await orchestrator.execute_complete_workflow()
        
        assert result["success"] is True
        assert "execution_time" in result
        assert "summary" in result
        assert "steps" in result
    
    @pytest.mark.asyncio
    @patch('src.main.OpentirOrchestrator._execute_step')
    async def test_execute_complete_workflow_failure(self, mock_execute_step, orchestrator):
        """Test workflow execution with failure."""
        mock_execute_step.side_effect = Exception("Test failure")
        
        result = await orchestrator.execute_complete_workflow()
        
        assert result["success"] is False
        assert "errors" in result
        assert len(result["errors"]) > 0


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    @pytest.mark.asyncio
    @patch('src.main.OpentirOrchestrator')
    async def test_build_complete_ecosystem(self, mock_orchestrator_class):
        """Test build_complete_ecosystem convenience function."""
        # Mock orchestrator
        mock_orchestrator = Mock()
        mock_orchestrator.execute_complete_workflow = AsyncMock(return_value={"success": True})
        mock_orchestrator_class.return_value = mock_orchestrator
        
        result = await build_complete_ecosystem(
            github_token="test_token",
            base_dir=Path("/test"),
            force_reclone=True
        )
        
        assert result["success"] is True
        mock_orchestrator.execute_complete_workflow.assert_called_once_with(force_reclone=True)
    
    @patch('src.main.OpentirOrchestrator')
    def test_get_workspace_status_function(self, mock_orchestrator_class):
        """Test get_workspace_status convenience function."""
        # Mock orchestrator
        mock_orchestrator = Mock()
        mock_orchestrator.get_workspace_status.return_value = {"status": "test"}
        mock_orchestrator_class.return_value = mock_orchestrator
        
        result = get_workspace_status(base_dir=Path("/test"))
        
        assert result["status"] == "test"
        mock_orchestrator.get_workspace_status.assert_called_once()


class TestErrorHandling:
    """Test error handling in orchestrator."""
    
    @pytest.fixture
    def orchestrator_with_errors(self, temp_base_dir):
        """Create orchestrator for error testing."""
        return OpentirOrchestrator(base_dir=temp_base_dir)
    
    @pytest.mark.asyncio
    async def test_analyze_repositories_no_directory(self, orchestrator_with_errors):
        """Test analysis when repos directory doesn't exist."""
        with pytest.raises(Exception) as exc_info:
            await orchestrator_with_errors._analyze_all_repositories()
        
        assert "No repositories found" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_generate_docs_without_analysis(self, orchestrator_with_errors):
        """Test docs generation without analysis results."""
        analysis_results = {"success": False}
        org_info = {}
        
        with pytest.raises(Exception) as exc_info:
            await orchestrator_with_errors._generate_comprehensive_docs(analysis_results, org_info)
        
        assert "Cannot generate docs without successful analysis" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__]) 