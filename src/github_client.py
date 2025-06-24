"""
GitHub API client for fetching Palantir repositories and metadata.
Handles rate limiting, pagination, and comprehensive data extraction.
"""

import asyncio
import aiohttp
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime
from github import Github, Repository as GitHubRepo
from github.GithubException import GithubException, RateLimitExceededException
import time

from .config import config, PALANTIR_ORG
from .utils import Logger, RepositoryInfo, AsyncUtils

class GitHubClient:
    """
    Enhanced GitHub API client for comprehensive repository fetching.
    Supports both synchronous and asynchronous operations with rate limiting.
    """
    
    def __init__(self, token: Optional[str] = None):
        """Initialize GitHub client with optional token."""
        self.token = token or config.github_token
        self.logger = Logger("github_client")
        self.github = Github(self.token) if self.token else Github()
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Rate limiting
        self.last_request_time = 0
        self.requests_made = 0
        self.rate_limit_reset = 0
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    def _check_rate_limit(self) -> None:
        """Check and handle rate limits."""
        try:
            rate_limit = self.github.get_rate_limit()
            core_limit = rate_limit.core
            
            if core_limit.remaining < 10:  # Conservative threshold
                reset_time = core_limit.reset.timestamp()
                sleep_time = max(0, reset_time - time.time() + 1)
                
                if sleep_time > 0:
                    self.logger.warning(f"Rate limit nearly exceeded. Sleeping for {sleep_time:.1f} seconds")
                    time.sleep(sleep_time)
                    
        except Exception as e:
            self.logger.warning(f"Could not check rate limit: {e}")
    
    def get_organization_info(self, org_name: str = PALANTIR_ORG) -> Dict[str, Any]:
        """Get comprehensive organization information."""
        try:
            self._check_rate_limit()
            org = self.github.get_organization(org_name)
            
            return {
                "login": org.login,
                "name": org.name,
                "description": org.description,
                "company": org.company,
                "location": org.location,
                "email": org.email,
                "blog": org.blog,
                "twitter_username": org.twitter_username,
                "public_repos": org.public_repos,
                "public_gists": org.public_gists,
                "followers": org.followers,
                "following": org.following,
                "html_url": org.html_url,
                "avatar_url": org.avatar_url,
                "type": org.type,
                "created_at": org.created_at,
                "updated_at": org.updated_at,
            }
        except GithubException as e:
            self.logger.error(f"GitHub API error fetching organization info: {e}")
            if e.status == 401:
                return {"error": "authentication_failed", "message": "Invalid GitHub token"}
            elif e.status == 403:
                return {"error": "rate_limit_exceeded", "message": "Rate limit exceeded"}
            elif e.status == 404:
                return {"error": "not_found", "message": f"Organization {org_name} not found"}
            return {"error": "github_error", "message": str(e)}
        except Exception as e:
            self.logger.error(f"Error fetching organization info: {e}")
            return {"error": "unknown_error", "message": str(e)}
    
    def get_all_repositories(self, org_name: str = PALANTIR_ORG) -> List[RepositoryInfo]:
        """
        Fetch all repositories for the organization.
        Returns comprehensive repository information.
        """
        repositories = []
        
        try:
            self.logger.info(f"Fetching repositories for organization: {org_name}")
            org = self.github.get_organization(org_name)
            
            # Get all repositories with pagination
            repos = org.get_repos(type="all", sort="updated", direction="desc")
            
            total_repos = repos.totalCount
            self.logger.info(f"Found {total_repos} repositories")
            
            for i, repo in enumerate(repos, 1):
                try:
                    self._check_rate_limit()
                    
                    repo_info = self._extract_repository_info(repo)
                    repositories.append(repo_info)
                    
                    if i % 10 == 0:
                        self.logger.info(f"Processed {i}/{total_repos} repositories")
                        
                except GithubException as e:
                    self.logger.error(f"GitHub API error processing repository {repo.name}: {e}")
                    if e.status in (401, 403):  # Auth error or rate limit
                        break
                    continue
                except Exception as e:
                    self.logger.error(f"Error processing repository {repo.name}: {e}")
                    continue
            
            self.logger.info(f"Successfully fetched {len(repositories)} repositories")
            return repositories
            
        except GithubException as e:
            self.logger.error(f"GitHub API error fetching repositories: {e}")
            if e.status == 401:
                self.logger.error("Authentication failed. Please check your GitHub token.")
            elif e.status == 403:
                self.logger.error("Rate limit exceeded. Please try again later.")
            return repositories
        except Exception as e:
            self.logger.error(f"Error fetching repositories: {e}")
            return repositories
    
    def _extract_repository_info(self, repo: GitHubRepo) -> RepositoryInfo:
        """Extract comprehensive information from GitHub repository object."""
        return RepositoryInfo(
            name=repo.name,
            full_name=repo.full_name,
            description=repo.description or "",
            url=repo.html_url,
            clone_url=repo.clone_url,
            language=repo.language or "Unknown",
            stars=repo.stargazers_count,
            forks=repo.forks_count,
            size=repo.size,
            topics=list(repo.get_topics()),
            created_at=repo.created_at,
            updated_at=repo.updated_at,
            archived=repo.archived,
            private=repo.private,
        )
    
    async def get_repository_content_async(self, repo_name: str, path: str = "") -> Dict[str, Any]:
        """
        Asynchronously fetch repository content.
        Returns file structure and metadata.
        """
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        url = f"{config.github_api_url}/repos/{PALANTIR_ORG}/{repo_name}/contents/{path}"
        headers = {}
        
        if self.token:
            headers["Authorization"] = f"token {self.token}"
        
        try:
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    self.logger.warning(f"Failed to fetch content for {repo_name}/{path}: {response.status}")
                    return {}
                    
        except Exception as e:
            self.logger.error(f"Error fetching content for {repo_name}/{path}: {e}")
            return {}
    
    async def get_repository_languages_async(self, repo_name: str) -> Dict[str, int]:
        """Asynchronously fetch repository language statistics."""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        url = f"{config.github_api_url}/repos/{PALANTIR_ORG}/{repo_name}/languages"
        headers = {}
        
        if self.token:
            headers["Authorization"] = f"token {self.token}"
        
        try:
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {}
                    
        except Exception as e:
            self.logger.error(f"Error fetching languages for {repo_name}: {e}")
            return {}
    
    async def get_repository_contributors_async(self, repo_name: str) -> List[Dict[str, Any]]:
        """Asynchronously fetch repository contributors."""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        url = f"{config.github_api_url}/repos/{PALANTIR_ORG}/{repo_name}/contributors"
        headers = {}
        
        if self.token:
            headers["Authorization"] = f"token {self.token}"
        
        try:
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return []
                    
        except Exception as e:
            self.logger.error(f"Error fetching contributors for {repo_name}: {e}")
            return []
    
    async def get_repository_releases_async(self, repo_name: str) -> List[Dict[str, Any]]:
        """Asynchronously fetch repository releases."""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        url = f"{config.github_api_url}/repos/{PALANTIR_ORG}/{repo_name}/releases"
        headers = {}
        
        if self.token:
            headers["Authorization"] = f"token {self.token}"
        
        try:
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return []
                    
        except Exception as e:
            self.logger.error(f"Error fetching releases for {repo_name}: {e}")
            return []
    
    def get_repository_detailed_info(self, repo_name: str) -> Dict[str, Any]:
        """Get detailed repository information including README, dependencies, etc."""
        try:
            self._check_rate_limit()
            repo = self.github.get_repo(f"{PALANTIR_ORG}/{repo_name}")
            
            detailed_info = {
                "basic_info": self._extract_repository_info(repo),
                "readme": self._get_readme_content(repo),
                "license": self._get_license_info(repo),
                "dependencies": self._get_dependencies(repo),
                "file_structure": self._get_file_structure(repo),
                "recent_commits": self._get_recent_commits(repo),
                "issues_stats": self._get_issues_stats(repo),
                "pull_requests_stats": self._get_pull_requests_stats(repo),
            }
            
            return detailed_info
            
        except GithubException as e:
            self.logger.error(f"GitHub API error fetching detailed info for {repo_name}: {e}")
            if e.status == 401:
                return {"error": "authentication_failed", "message": "Invalid GitHub token"}
            elif e.status == 403:
                return {"error": "rate_limit_exceeded", "message": "Rate limit exceeded"}
            elif e.status == 404:
                return {"error": "not_found", "message": f"Repository {repo_name} not found"}
            return {"error": "github_error", "message": str(e)}
        except Exception as e:
            self.logger.error(f"Error fetching detailed info for {repo_name}: {e}")
            return {"error": "unknown_error", "message": str(e)}
    
    def _get_readme_content(self, repo: GitHubRepo) -> str:
        """Extract README content from repository."""
        try:
            readme = repo.get_readme()
            return readme.decoded_content.decode('utf-8')
        except:
            return ""
    
    def _get_license_info(self, repo: GitHubRepo) -> Dict[str, Any]:
        """Extract license information."""
        try:
            if repo.license:
                return {
                    "name": repo.license.name,
                    "key": repo.license.key,
                    "url": repo.license.url,
                }
        except:
            pass
        return {}
    
    def _get_dependencies(self, repo: GitHubRepo) -> Dict[str, List[str]]:
        """Extract dependency information from common dependency files."""
        dependencies = {
            "package.json": [],
            "requirements.txt": [],
            "pom.xml": [],
            "build.gradle": [],
            "Cargo.toml": [],
            "go.mod": [],
        }
        
        dependency_files = [
            "package.json", "requirements.txt", "pom.xml", 
            "build.gradle", "Cargo.toml", "go.mod"
        ]
        
        for file_name in dependency_files:
            try:
                content = repo.get_contents(file_name)
                if hasattr(content, 'decoded_content'):
                    file_content = content.decoded_content.decode('utf-8')
                    dependencies[file_name] = self._parse_dependencies(file_name, file_content)
            except:
                continue
        
        return dependencies
    
    def _parse_dependencies(self, file_name: str, content: str) -> List[str]:
        """Parse dependencies from file content."""
        # Simple dependency parsing - can be enhanced
        dependencies = []
        
        if file_name == "requirements.txt":
            for line in content.split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    dependencies.append(line.split('==')[0].split('>=')[0].split('<=')[0])
        
        # Add more parsers for other file types as needed
        
        return dependencies
    
    def _get_file_structure(self, repo: GitHubRepo, max_depth: int = 2) -> Dict[str, Any]:
        """Get basic file structure of repository."""
        try:
            contents = repo.get_contents("")
            structure = {}
            
            for content in contents:
                if content.type == "dir" and max_depth > 0:
                    try:
                        subcontents = repo.get_contents(content.path)
                        structure[content.name] = {
                            "type": "directory",
                            "files": [sub.name for sub in subcontents[:10]]  # Limit for performance
                        }
                    except:
                        structure[content.name] = {"type": "directory", "files": []}
                else:
                    structure[content.name] = {"type": "file", "size": content.size}
            
            return structure
        except:
            return {}
    
    def _get_recent_commits(self, repo: GitHubRepo, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent commits information."""
        try:
            commits = repo.get_commits()[:limit]
            return [
                {
                    "sha": commit.sha[:8],
                    "message": commit.commit.message.split('\n')[0],
                    "author": commit.commit.author.name,
                    "date": commit.commit.author.date,
                }
                for commit in commits
            ]
        except:
            return []
    
    def _get_issues_stats(self, repo: GitHubRepo) -> Dict[str, int]:
        """Get issues statistics."""
        try:
            return {
                "open_issues": repo.open_issues_count,
                "total_issues": repo.open_issues_count,  # GitHub API limitation
            }
        except:
            return {"open_issues": 0, "total_issues": 0}
    
    def _get_pull_requests_stats(self, repo: GitHubRepo) -> Dict[str, int]:
        """Get pull requests statistics."""
        try:
            open_prs = repo.get_pulls(state="open")
            return {
                "open_pull_requests": open_prs.totalCount,
            }
        except:
            return {"open_pull_requests": 0}
    
    async def fetch_all_repositories_with_details(self) -> List[Dict[str, Any]]:
        """
        Asynchronously fetch all repositories with comprehensive details.
        This is the main method for comprehensive data collection.
        """
        self.logger.info("Starting comprehensive repository fetch...")
        
        # First get basic repository list
        basic_repos = self.get_all_repositories()
        
        if not basic_repos:
            self.logger.error("No repositories found")
            return []
        
        async with self:
            # Fetch detailed information for each repository
            tasks = []
            semaphore = asyncio.Semaphore(config.max_concurrent_requests)
            
            for repo_info in basic_repos:
                task = self._fetch_repository_comprehensive_details(semaphore, repo_info.name)
                tasks.append(task)
            
            # Execute with rate limiting
            detailed_repos = await AsyncUtils.gather_with_limit(
                config.max_concurrent_requests, *tasks
            )
            
            # Combine basic and detailed info
            comprehensive_repos = []
            for basic_repo, detailed_info in zip(basic_repos, detailed_repos):
                repo_data = {
                    "basic_info": basic_repo,
                    "detailed_info": detailed_info,
                    "analysis_timestamp": datetime.now().isoformat(),
                }
                comprehensive_repos.append(repo_data)
            
            self.logger.info(f"Completed comprehensive fetch for {len(comprehensive_repos)} repositories")
            return comprehensive_repos
    
    async def _fetch_repository_comprehensive_details(self, semaphore: asyncio.Semaphore, repo_name: str) -> Dict[str, Any]:
        """Fetch comprehensive details for a single repository."""
        async with semaphore:
            try:
                # Fetch multiple data points concurrently
                languages_task = self.get_repository_languages_async(repo_name)
                contributors_task = self.get_repository_contributors_async(repo_name)
                releases_task = self.get_repository_releases_async(repo_name)
                content_task = self.get_repository_content_async(repo_name)
                
                languages, contributors, releases, content = await asyncio.gather(
                    languages_task, contributors_task, releases_task, content_task,
                    return_exceptions=True
                )
                
                return {
                    "languages": languages if not isinstance(languages, Exception) else {},
                    "contributors": contributors if not isinstance(contributors, Exception) else [],
                    "releases": releases if not isinstance(releases, Exception) else [],
                    "content": content if not isinstance(content, Exception) else {},
                }
                
            except Exception as e:
                self.logger.error(f"Error fetching details for {repo_name}: {e}")
                return {
                    "languages": {},
                    "contributors": [],
                    "releases": [],
                    "content": {},
                } 