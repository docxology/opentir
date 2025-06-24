"""
Main orchestrator for Opentir package.
Provides high-level interface for comprehensive Palantir ecosystem analysis.
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from .github_client import GitHubClient
from .repo_manager import RepositoryManager
from .code_analyzer import CodeAnalyzer, RepositoryAnalysis
from .docs_generator import DocumentationGenerator
from .utils import Logger, ConfigManager, format_duration
from .config import config, ensure_directories

class OpentirOrchestrator:
    """
    Main orchestrator class for comprehensive Palantir ecosystem analysis.
    Coordinates all components for complete workflow execution.
    """
    
    def __init__(self, 
                 base_dir: Optional[Path] = None,
                 github_token: Optional[str] = None):
        """Initialize Opentir orchestrator."""
        self.base_dir = base_dir or config.base_dir
        self.github_token = github_token or config.github_token
        self.logger = Logger("opentir_orchestrator")
        
        # Initialize components
        self.github_client = GitHubClient(self.github_token)
        self.repo_manager = RepositoryManager(self.base_dir)
        self.code_analyzer = CodeAnalyzer()
        self.docs_generator = DocumentationGenerator(self.base_dir / config.docs_dir)
        
        # Setup directories
        ensure_directories()
        
        # Execution state
        self.execution_state = {
            "started_at": None,
            "completed_at": None,
            "steps_completed": [],
            "current_step": None,
            "errors": [],
        }
    
    async def execute_complete_workflow(self, 
                                      force_reclone: bool = False,
                                      skip_analysis: bool = False,
                                      skip_docs: bool = False) -> Dict[str, Any]:
        """
        Execute complete Opentir workflow.
        
        Performs all steps:
        1. Fetch repository information
        2. Clone all repositories
        3. Analyze code across all repositories  
        4. Generate comprehensive documentation
        
        Returns comprehensive results and statistics.
        """
        self.logger.info("🚀 Starting complete Opentir workflow...")
        start_time = datetime.now()
        self.execution_state["started_at"] = start_time.isoformat()
        
        workflow_results = {
            "success": False,
            "execution_time": None,
            "steps": {},
            "summary": {},
            "errors": [],
        }
        
        try:
            # Step 1: Fetch organization and repository information
            self.execution_state["current_step"] = "fetch_info"
            self.logger.info("📡 Step 1: Fetching organization and repository information...")
            
            org_info = await self._execute_step("fetch_org_info", self._fetch_organization_info)
            repo_info = await self._execute_step("fetch_repo_list", self._fetch_repository_list)
            
            workflow_results["steps"]["fetch_info"] = {
                "organization": org_info,
                "repositories": repo_info,
            }
            
            # Step 2: Clone all repositories
            self.execution_state["current_step"] = "clone_repos"
            self.logger.info("📂 Step 2: Cloning all repositories...")
            
            clone_results = await self._execute_step("clone_repositories", 
                                                   lambda: self.repo_manager.clone_all_repositories(force_reclone))
            
            workflow_results["steps"]["clone_repos"] = clone_results
            
            if not clone_results.get("success", False):
                raise Exception("Repository cloning failed")
            
            # Step 3: Analyze code (optional)
            if not skip_analysis:
                self.execution_state["current_step"] = "analyze_code"
                self.logger.info("🔍 Step 3: Analyzing code across all repositories...")
                
                analysis_results = await self._execute_step("analyze_code", self._analyze_all_repositories)
                workflow_results["steps"]["analyze_code"] = analysis_results
            else:
                self.logger.info("⏭️  Skipping code analysis")
            
            # Step 4: Generate documentation (optional)
            if not skip_docs:
                self.execution_state["current_step"] = "generate_docs"
                self.logger.info("📚 Step 4: Generating comprehensive documentation...")
                
                docs_results = await self._execute_step("generate_docs", 
                                                      lambda: self._generate_comprehensive_docs(
                                                          workflow_results["steps"].get("analyze_code", {}),
                                                          org_info
                                                      ))
                workflow_results["steps"]["generate_docs"] = docs_results
            else:
                self.logger.info("⏭️  Skipping documentation generation")
            
            # Calculate execution time and finalize
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            workflow_results.update({
                "success": True,
                "execution_time": execution_time,
                "execution_time_formatted": format_duration(execution_time),
                "summary": self._generate_workflow_summary(workflow_results["steps"]),
            })
            
            self.execution_state["completed_at"] = end_time.isoformat()
            self.logger.info(f"✅ Complete workflow finished in {format_duration(execution_time)}")
            
            return workflow_results
            
        except Exception as e:
            self.execution_state["errors"].append(str(e))
            workflow_results["errors"].append(str(e))
            workflow_results["success"] = False
            self.logger.error(f"❌ Workflow failed: {e}")
            return workflow_results
    
    async def _execute_step(self, step_name: str, step_function) -> Any:
        """Execute a workflow step with error handling and logging."""
        try:
            self.logger.info(f"▶️  Executing step: {step_name}")
            result = await step_function() if asyncio.iscoroutinefunction(step_function) else step_function()
            self.execution_state["steps_completed"].append(step_name)
            self.logger.info(f"✅ Step completed: {step_name}")
            return result
        except Exception as e:
            error_msg = f"Step {step_name} failed: {str(e)}"
            self.execution_state["errors"].append(error_msg)
            self.logger.error(f"❌ {error_msg}")
            raise
    
    async def _fetch_organization_info(self) -> Dict[str, Any]:
        """Fetch comprehensive organization information."""
        return self.github_client.get_organization_info()
    
    async def _fetch_repository_list(self) -> Dict[str, Any]:
        """Fetch comprehensive repository list."""
        repositories = self.github_client.get_all_repositories()
        return {
            "repositories": [repo.__dict__ for repo in repositories],
            "total_count": len(repositories),
            "by_language": self._group_repos_by_language(repositories),
            "by_activity": self._categorize_repos_by_activity(repositories),
        }
    
    def _group_repos_by_language(self, repositories) -> Dict[str, int]:
        """Group repositories by primary language."""
        language_counts = {}
        for repo in repositories:
            lang = repo.language or "Unknown"
            language_counts[lang] = language_counts.get(lang, 0) + 1
        return dict(sorted(language_counts.items(), key=lambda x: x[1], reverse=True))
    
    def _categorize_repos_by_activity(self, repositories) -> Dict[str, int]:
        """Categorize repositories by activity level."""
        categories = {"high": 0, "medium": 0, "low": 0, "archived": 0}
        
        for repo in repositories:
            if repo.archived:
                categories["archived"] += 1
            elif repo.stars > 1000:
                categories["high"] += 1
            elif repo.stars > 100:
                categories["medium"] += 1
            else:
                categories["low"] += 1
        
        return categories
    
    async def _analyze_all_repositories(self) -> Dict[str, Any]:
        """Analyze code across all cloned repositories."""
        repos_dir = self.base_dir / config.repos_dir / "all_repos"
        
        if not repos_dir.exists():
            raise Exception("No repositories found for analysis")
        
        analyses = []
        failed_analyses = []
        
        # Analyze each repository
        for repo_dir in repos_dir.iterdir():
            if repo_dir.is_dir() and not repo_dir.name.startswith('.'):
                try:
                    analysis = self.code_analyzer.analyze_repository(repo_dir)
                    analyses.append(analysis)
                    self.logger.debug(f"Analyzed {repo_dir.name}: {len(analysis.all_elements)} elements")
                except Exception as e:
                    failed_analyses.append({"repo": repo_dir.name, "error": str(e)})
                    self.logger.warning(f"Failed to analyze {repo_dir.name}: {e}")
        
        # Generate functionality matrix
        functionality_matrix = self.code_analyzer.generate_functionality_matrix(analyses)
        
        # Save analysis results
        analysis_dir = self.base_dir / "analysis_results"
        analysis_dir.mkdir(exist_ok=True)
        
        # Save detailed analyses
        with open(analysis_dir / "repository_analyses.json", 'w') as f:
            json.dump([self._serialize_analysis(a) for a in analyses], f, indent=2, default=str)
        
        # Save functionality matrix
        with open(analysis_dir / "functionality_matrix.json", 'w') as f:
            json.dump(functionality_matrix, f, indent=2, default=str)
        
        return {
            "success": True,
            "total_repositories_analyzed": len(analyses),
            "failed_analyses": len(failed_analyses),
            "total_elements_extracted": sum(len(a.all_elements) for a in analyses),
            "total_lines_analyzed": sum(a.total_lines for a in analyses),
            "functionality_matrix": functionality_matrix,
            "analyses": analyses,
            "failures": failed_analyses,
        }
    
    def _serialize_analysis(self, analysis: RepositoryAnalysis) -> Dict[str, Any]:
        """Serialize repository analysis for JSON output."""
        return {
            "repo_name": analysis.repo_name,
            "total_files": analysis.total_files,
            "total_lines": analysis.total_lines,
            "languages": analysis.languages,
            "functionality_summary": analysis.functionality_summary,
            "metrics": analysis.metrics,
            "element_count": len(analysis.all_elements),
        }
    
    async def _generate_comprehensive_docs(self, analysis_results: Dict[str, Any], org_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive documentation."""
        if not analysis_results.get("success"):
            raise Exception("Cannot generate docs without successful analysis")
        
        analyses = analysis_results.get("analyses", [])
        functionality_matrix = analysis_results.get("functionality_matrix", {})
        
        docs_results = self.docs_generator.generate_complete_documentation(
            analyses, functionality_matrix, org_info
        )
        
        return {
            "success": True,
            "documentation_components": len(docs_results),
            "docs_location": str(self.docs_generator.docs_dir),
            "components": list(docs_results.keys()),
        }
    
    def _generate_workflow_summary(self, steps: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive workflow summary."""
        summary = {
            "steps_completed": len(steps),
            "total_repositories": 0,
            "successful_clones": 0,
            "repositories_analyzed": 0,
            "total_elements_extracted": 0,
            "documentation_generated": False,
        }
        
        # Extract metrics from steps
        if "clone_repos" in steps:
            clone_data = steps["clone_repos"]
            summary["total_repositories"] = clone_data.get("total_repositories", 0)
            summary["successful_clones"] = clone_data.get("successful_clones", 0)
        
        if "analyze_code" in steps:
            analysis_data = steps["analyze_code"]
            summary["repositories_analyzed"] = analysis_data.get("total_repositories_analyzed", 0)
            summary["total_elements_extracted"] = analysis_data.get("total_elements_extracted", 0)
        
        if "generate_docs" in steps:
            summary["documentation_generated"] = steps["generate_docs"].get("success", False)
        
        return summary
    
    def get_workspace_status(self) -> Dict[str, Any]:
        """Get comprehensive status of Opentir workspace."""
        status = {
            "workspace_initialized": True,
            "base_directory": str(self.base_dir),
            "repositories": self._get_repository_status(),
            "analysis": self._get_analysis_status(),
            "documentation": self._get_documentation_status(),
            "configuration": self._get_configuration_status(),
        }
        
        return status
    
    def _get_repository_status(self) -> Dict[str, Any]:
        """Get repository cloning status."""
        repos_dir = self.base_dir / config.repos_dir
        
        if not repos_dir.exists():
            return {"status": "not_initialized", "count": 0}
        
        all_repos_dir = repos_dir / "all_repos"
        if not all_repos_dir.exists():
            return {"status": "initialized", "count": 0}
        
        cloned_repos = [d for d in all_repos_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        # Check organization structure
        organization = {
            "by_language": (repos_dir / "by_language").exists(),
            "by_category": (repos_dir / "by_category").exists(),
            "popular": (repos_dir / "popular").exists(),
        }
        
        return {
            "status": "cloned",
            "count": len(cloned_repos),
            "organization": organization,
            "total_size_mb": sum(self.repo_manager._get_repo_size_mb(repo) for repo in cloned_repos),
        }
    
    def _get_analysis_status(self) -> Dict[str, Any]:
        """Get code analysis status."""
        analysis_dir = self.base_dir / "analysis_results"
        
        if not analysis_dir.exists():
            return {"status": "not_started"}
        
        analyses_file = analysis_dir / "repository_analyses.json"
        matrix_file = analysis_dir / "functionality_matrix.json"
        
        if analyses_file.exists() and matrix_file.exists():
            try:
                with open(analyses_file, 'r') as f:
                    analyses = json.load(f)
                
                return {
                    "status": "completed",
                    "repositories_analyzed": len(analyses),
                    "total_elements": sum(a.get("element_count", 0) for a in analyses),
                    "last_updated": analyses_file.stat().st_mtime,
                }
            except Exception:
                return {"status": "corrupted"}
        
        return {"status": "partial"}
    
    def _get_documentation_status(self) -> Dict[str, Any]:
        """Get documentation generation status."""
        docs_dir = self.base_dir / config.docs_dir
        
        if not docs_dir.exists():
            return {"status": "not_generated"}
        
        key_files = ["index.md", "mkdocs.yml"]
        existing_files = [f for f in key_files if (docs_dir / f).exists()]
        
        if len(existing_files) == len(key_files):
            return {
                "status": "generated",
                "location": str(docs_dir),
                "components": len(list(docs_dir.rglob("*.md"))),
            }
        
        return {"status": "partial", "existing_files": existing_files}
    
    def _get_configuration_status(self) -> Dict[str, Any]:
        """Get configuration status."""
        return {
            "github_token_set": bool(self.github_token),
            "base_directory": str(self.base_dir),
            "log_level": config.log_level,
            "supported_languages": config.supported_languages,
        }

# Convenience functions for direct usage
async def build_complete_ecosystem(github_token: Optional[str] = None, 
                                 base_dir: Optional[Path] = None,
                                 force_reclone: bool = False) -> Dict[str, Any]:
    """
    Convenience function to build complete Palantir ecosystem analysis.
    
    Args:
        github_token: GitHub API token for rate limiting
        base_dir: Base directory for workspace (defaults to current directory)
        force_reclone: Whether to force re-clone existing repositories
    
    Returns:
        Comprehensive results dictionary
    """
    orchestrator = OpentirOrchestrator(base_dir=base_dir, github_token=github_token)
    return await orchestrator.execute_complete_workflow(force_reclone=force_reclone)

def get_workspace_status(base_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Get current workspace status."""
    orchestrator = OpentirOrchestrator(base_dir=base_dir)
    return orchestrator.get_workspace_status()


# Module-level convenience functions remain available for import 