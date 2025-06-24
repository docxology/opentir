"""
Repository manager for cloning, organizing, and managing Palantir repositories.
Handles git operations and local repository organization.
"""

import os
import shutil
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from git import Repo, GitCommandError
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import config, ensure_directories
from .utils import Logger, RepositoryInfo, FileUtils, format_bytes
from .github_client import GitHubClient

class RepositoryManager:
    """
    Comprehensive repository management for Palantir's open source ecosystem.
    Handles cloning, updating, organizing, and analyzing local repositories.
    """
    
    def __init__(self, base_dir: Optional[Path] = None):
        """Initialize repository manager with base directory."""
        self.base_dir = base_dir or config.base_dir
        self.repos_dir = self.base_dir / config.repos_dir
        self.logger = Logger("repo_manager")
        
        # Ensure directories exist
        ensure_directories()
        FileUtils.ensure_directory(self.repos_dir)
        
        # Initialize GitHub client
        self.github_client = GitHubClient()
        
    def setup_repository_structure(self) -> None:
        """Set up organized directory structure for repositories."""
        structure_dirs = [
            self.repos_dir / "by_language",
            self.repos_dir / "by_category", 
            self.repos_dir / "popular",
            self.repos_dir / "archived",
            self.repos_dir / "all_repos",
        ]
        
        for directory in structure_dirs:
            FileUtils.ensure_directory(directory)
            
        self.logger.info("Repository structure created")
    
    async def clone_all_repositories(self, force_update: bool = False) -> Dict[str, Any]:
        """
        Clone all Palantir repositories with organized structure.
        Intelligently skips existing repositories and only downloads missing ones.
        Fast local check first, only fetches from GitHub if needed.
        Returns comprehensive results and statistics.
        """
        self.logger.info("Starting intelligent repository management...")
        
        # Setup directory structure
        self.setup_repository_structure()
        
        # Step 1: Fast local repository check
        local_status = self._get_fast_local_repository_status()
        self.logger.info(f"Local repository status:")
        self.logger.info(f"  - Local repositories found: {local_status['local_count']}")
        
        # If we have a reasonable number of repositories locally and not forcing update,
        # skip the GitHub API call entirely
        if local_status['local_count'] >= 200 and not force_update:
            self.logger.info("✅ Found substantial repository collection locally! Skipping GitHub API check.")
            self.logger.info("   Use --force flag if you want to check for new repositories from GitHub.")
            
            # Organize existing repositories
            existing_results = []
            for repo_name in local_status['local_repos']:
                repo_path = self.repos_dir / "all_repos" / repo_name
                existing_results.append({
                    "success": True,
                    "name": repo_name,
                    "action": "existing",
                    "path": str(repo_path),
                    "size_mb": self._get_repo_size_mb(repo_path),
                })
            
            await self._organize_repositories(existing_results)
            
            return {
                "success": True,
                "total_repositories": local_status['local_count'],
                "existing_repositories": local_status['local_count'],
                "new_repositories": 0,
                "failed_clones": 0,
                "message": f"All {local_status['local_count']} repositories already exist locally",
                "repositories": existing_results,
            }
        
        # Step 2: If we need to check GitHub (few local repos or force_update)
        self.logger.info("📡 Fetching repository list from GitHub for comparison...")
        repositories = self.github_client.get_all_repositories()
        
        if not repositories:
            self.logger.error("No repositories found from GitHub API")
            return {"success": False, "repositories": []}
        
        # Step 3: Compare local vs GitHub repositories
        repo_analysis = self._analyze_repository_status(repositories, local_status)
        
        self.logger.info(f"Repository comparison results:")
        self.logger.info(f"  - Total available on GitHub: {repo_analysis['total_available']}")
        self.logger.info(f"  - Already downloaded locally: {repo_analysis['existing_count']}")
        self.logger.info(f"  - Missing/new to download: {repo_analysis['missing_count']}")
        
        if repo_analysis['missing_count'] == 0 and not force_update:
            self.logger.info("✅ All GitHub repositories already downloaded! No cloning needed.")
            # Still organize existing repositories
            existing_results = []
            for repo_name in repo_analysis['existing_repos']:
                repo_path = self.repos_dir / "all_repos" / repo_name
                existing_results.append({
                    "success": True,
                    "name": repo_name,
                    "action": "existing",
                    "path": str(repo_path),
                    "size_mb": self._get_repo_size_mb(repo_path),
                })
            
            await self._organize_repositories(existing_results)
            
            return {
                "success": True,
                "total_repositories": len(repositories),
                "existing_repositories": repo_analysis['existing_count'],
                "new_repositories": 0,
                "failed_clones": 0,
                "message": "All repositories already exist",
                "repositories": existing_results,
            }
        
        # Step 4: Clone only missing repositories (or all if force_update)
        repos_to_process = repositories if force_update else repo_analysis['missing_repos']
        
        if repos_to_process:
            self.logger.info(f"📥 Downloading {len(repos_to_process)} repositories...")
            
            # Clone repositories with concurrency control
            semaphore = asyncio.Semaphore(5)  # Limit concurrent clones
            tasks = []
            
            for repo_info in repos_to_process:
                task = self._clone_single_repository(semaphore, repo_info, force_update)
                tasks.append(task)
            
            # Execute cloning tasks
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            successful_clones = []
            failed_clones = []
            total_size = 0
            
            for repo_info, result in zip(repos_to_process, results):
                if isinstance(result, Exception):
                    failed_clones.append({"name": repo_info.name, "error": str(result)})
                elif result["success"]:
                    successful_clones.append(result)
                    total_size += result.get("size_mb", 0)
                else:
                    failed_clones.append(result)
        else:
            successful_clones = []
            failed_clones = []
            total_size = 0
        
        # Include existing repositories in organization
        all_successful = successful_clones.copy()
        if not force_update:
            # Add existing repos to the successful list for organization
            for repo_name in repo_analysis['existing_repos']:
                repo_path = self.repos_dir / "all_repos" / repo_name
                if repo_path.exists():
                    all_successful.append({
                        "success": True,
                        "name": repo_name,
                        "action": "existing",
                        "path": str(repo_path),
                        "size_mb": self._get_repo_size_mb(repo_path),
                    })
        
        # Organize repositories by categories
        await self._organize_repositories(all_successful)
        
        summary = {
            "success": True,
            "total_repositories": len(repositories),
            "existing_repositories": repo_analysis['existing_count'],
            "new_repositories": len(successful_clones),
            "failed_clones": len(failed_clones),
            "total_size_mb": total_size,
            "repositories": all_successful,
            "failures": failed_clones,
        }
        
        if len(successful_clones) > 0:
            self.logger.info(f"✅ Downloaded {len(successful_clones)} new repositories")
        self.logger.info(f"📊 Total: {len(all_successful)} repositories available locally")
        
        return summary
    
    def _get_fast_local_repository_status(self) -> Dict[str, Any]:
        """Fast check of what repositories exist locally."""
        all_repos_dir = self.repos_dir / "all_repos"
        
        # Get list of existing local repositories
        local_repos = set()
        if all_repos_dir.exists():
            local_repos = {d.name for d in all_repos_dir.iterdir() 
                          if d.is_dir() and not d.name.startswith('.')}
        
        return {
            "local_count": len(local_repos),
            "local_repos": local_repos,
        }
    
    def _analyze_repository_status(self, available_repos: List, local_status: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze which repositories exist locally vs what's available on GitHub."""
        if local_status is None:
            local_status = self._get_fast_local_repository_status()
        
        existing_repos = local_status['local_repos']
        
        # Get list of available repositories from GitHub
        available_repo_names = {repo.name for repo in available_repos}
        
        # Find missing repositories
        missing_repo_names = available_repo_names - existing_repos
        missing_repos = [repo for repo in available_repos if repo.name in missing_repo_names]
        
        return {
            "total_available": len(available_repos),
            "existing_count": len(existing_repos),
            "missing_count": len(missing_repos),
            "existing_repos": existing_repos,
            "missing_repos": missing_repos,
            "available_repo_names": available_repo_names,
        }
    
    async def _clone_single_repository(self, semaphore: asyncio.Semaphore, repo_info: RepositoryInfo, force_update: bool) -> Dict[str, Any]:
        """Clone a single repository with error handling."""
        async with semaphore:
            return await asyncio.get_event_loop().run_in_executor(
                None, self._clone_repository_sync, repo_info, force_update
            )
    
    def _clone_repository_sync(self, repo_info: RepositoryInfo, force_update: bool) -> Dict[str, Any]:
        """Synchronous repository cloning operation."""
        repo_path = self.repos_dir / "all_repos" / repo_info.name
        
        try:
            # Check if repository already exists
            if repo_path.exists():
                if force_update:
                    self.logger.info(f"Updating existing repository: {repo_info.name}")
                    repo = Repo(repo_path)
                    repo.remotes.origin.pull()
                else:
                    self.logger.debug(f"Repository already exists: {repo_info.name}")
                    return {
                        "success": True,
                        "name": repo_info.name,
                        "action": "skipped",
                        "path": str(repo_path),
                        "size_mb": self._get_repo_size_mb(repo_path),
                    }
            else:
                self.logger.info(f"Cloning repository: {repo_info.name}")
                Repo.clone_from(repo_info.clone_url, repo_path, depth=1)  # Shallow clone for efficiency
            
            return {
                "success": True,
                "name": repo_info.name,
                "action": "updated" if force_update and repo_path.exists() else "cloned",
                "path": str(repo_path),
                "size_mb": self._get_repo_size_mb(repo_path),
                "language": repo_info.language,
                "stars": repo_info.stars,
                "archived": repo_info.archived,
            }
            
        except GitCommandError as e:
            self.logger.error(f"Git error cloning {repo_info.name}: {e}")
            return {
                "success": False,
                "name": repo_info.name,
                "error": f"Git error: {str(e)}",
            }
        except Exception as e:
            self.logger.error(f"Error cloning {repo_info.name}: {e}")
            return {
                "success": False,
                "name": repo_info.name,
                "error": str(e),
            }
    
    def _get_repo_size_mb(self, repo_path: Path) -> float:
        """Get repository size in MB."""
        try:
            total_size = sum(
                f.stat().st_size for f in repo_path.rglob('*') if f.is_file()
            )
            return total_size / (1024 * 1024)
        except:
            return 0.0
    
    async def _organize_repositories(self, successful_repos: List[Dict[str, Any]]) -> None:
        """Organize repositories into categorical directories."""
        self.logger.info("Organizing repositories by categories...")
        
        # Group by language
        language_groups = {}
        popular_repos = []
        archived_repos = []
        
        for repo in successful_repos:
            language = repo.get("language", "Unknown").lower()
            if language not in language_groups:
                language_groups[language] = []
            language_groups[language].append(repo)
            
            # Popular repositories (>1000 stars)
            if repo.get("stars", 0) > 1000:
                popular_repos.append(repo)
            
            # Archived repositories
            if repo.get("archived", False):
                archived_repos.append(repo)
        
        # Create symbolic links for organization
        await asyncio.get_event_loop().run_in_executor(
            None, self._create_category_links, language_groups, popular_repos, archived_repos
        )
    
    def _create_category_links(self, language_groups: Dict[str, List], popular_repos: List, archived_repos: List) -> None:
        """Create symbolic links for repository organization."""
        # Organize by language
        for language, repos in language_groups.items():
            language_dir = self.repos_dir / "by_language" / language
            FileUtils.ensure_directory(language_dir)
            
            for repo in repos:
                source_path = Path(repo["path"])
                link_path = language_dir / repo["name"]
                
                if source_path.exists() and not link_path.exists():
                    try:
                        if os.name == 'nt':  # Windows
                            shutil.copytree(source_path, link_path)
                        else:  # Unix-like systems
                            link_path.symlink_to(source_path)
                    except Exception as e:
                        self.logger.debug(f"Could not create link for {repo['name']}: {e}")
        
        # Organize popular repositories
        popular_dir = self.repos_dir / "popular"
        for repo in popular_repos:
            source_path = Path(repo["path"])
            link_path = popular_dir / repo["name"]
            
            if source_path.exists() and not link_path.exists():
                try:
                    if os.name == 'nt':
                        shutil.copytree(source_path, link_path)
                    else:
                        link_path.symlink_to(source_path)
                except Exception as e:
                    self.logger.debug(f"Could not create popular link for {repo['name']}: {e}")
        
        # Organize archived repositories
        archived_dir = self.repos_dir / "archived"
        for repo in archived_repos:
            source_path = Path(repo["path"])
            link_path = archived_dir / repo["name"]
            
            if source_path.exists() and not link_path.exists():
                try:
                    if os.name == 'nt':
                        shutil.copytree(source_path, link_path)
                    else:
                        link_path.symlink_to(source_path)
                except Exception as e:
                    self.logger.debug(f"Could not create archived link for {repo['name']}: {e}")
    
    def get_local_repository_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about local repositories."""
        if not self.repos_dir.exists():
            return {"error": "Repository directory does not exist"}
        
        all_repos_dir = self.repos_dir / "all_repos"
        if not all_repos_dir.exists():
            return {"error": "No repositories have been cloned"}
        
        stats = {
            "total_repositories": 0,
            "total_size_mb": 0,
            "languages": {},
            "repositories": [],
        }
        
        for repo_dir in all_repos_dir.iterdir():
            if repo_dir.is_dir() and not repo_dir.name.startswith('.'):
                repo_size = self._get_repo_size_mb(repo_dir)
                language = self._detect_primary_language(repo_dir)
                
                stats["total_repositories"] += 1
                stats["total_size_mb"] += repo_size
                
                if language not in stats["languages"]:
                    stats["languages"][language] = {"count": 0, "total_size_mb": 0}
                
                stats["languages"][language]["count"] += 1
                stats["languages"][language]["total_size_mb"] += repo_size
                
                stats["repositories"].append({
                    "name": repo_dir.name,
                    "size_mb": repo_size,
                    "language": language,
                    "path": str(repo_dir),
                })
        
        # Sort repositories by size
        stats["repositories"].sort(key=lambda x: x["size_mb"], reverse=True)
        
        return stats
    
    def _detect_primary_language(self, repo_path: Path) -> str:
        """Detect primary programming language of repository."""
        language_files = {
            "python": [".py"],
            "javascript": [".js", ".jsx"], 
            "typescript": [".ts", ".tsx"],
            "java": [".java"],
            "go": [".go"],
            "rust": [".rs"],
            "scala": [".scala", ".sc"],
            "shell": [".sh", ".bash"],
        }
        
        language_counts = {}
        
        try:
            for file_path in repo_path.rglob("*"):
                if file_path.is_file():
                    suffix = file_path.suffix.lower()
                    for language, extensions in language_files.items():
                        if suffix in extensions:
                            language_counts[language] = language_counts.get(language, 0) + 1
        except Exception:
            return "Unknown"
        
        if language_counts:
            return max(language_counts, key=language_counts.get)
        
        return "Unknown"
    
    def cleanup_repositories(self, keep_popular: bool = True) -> Dict[str, Any]:
        """Clean up repositories with optional filters."""
        if not self.repos_dir.exists():
            return {"error": "Repository directory does not exist"}
        
        removed_repos = []
        total_size_freed = 0
        
        try:
            all_repos_dir = self.repos_dir / "all_repos"
            
            for repo_dir in all_repos_dir.iterdir():
                if repo_dir.is_dir() and not repo_dir.name.startswith('.'):
                    repo_size = self._get_repo_size_mb(repo_dir)
                    
                    # Logic for cleanup (can be enhanced with more criteria)
                    should_remove = False
                    
                    if not keep_popular:
                        should_remove = True
                    
                    if should_remove:
                        shutil.rmtree(repo_dir)
                        removed_repos.append({
                            "name": repo_dir.name,
                            "size_mb": repo_size,
                        })
                        total_size_freed += repo_size
            
            return {
                "success": True,
                "removed_repositories": len(removed_repos),
                "total_size_freed_mb": total_size_freed,
                "removed_repos": removed_repos,
            }
            
        except Exception as e:
            return {"error": f"Cleanup failed: {str(e)}"}
    
    async def update_all_repositories(self) -> Dict[str, Any]:
        """Update all existing repositories."""
        return await self.clone_all_repositories(force_update=True) 