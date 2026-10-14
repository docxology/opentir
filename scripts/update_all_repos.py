#!/usr/bin/env python3
"""
Comprehensive script to re-pull and check all Palantir repositories.
This script performs a complete refresh of all repositories with validation.
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.main import OpentirOrchestrator
from src.utils import Logger
from src.config import config

logger = Logger("update_all_repos")

async def update_all_palantir_repos(
    github_token: str = None,
    base_dir: Path = None,
    force_reclone: bool = True,
    validate_repos: bool = True,
    check_analysis: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive function to update all Palantir repositories.
    
    This function:
    1. Fetches latest repository list from GitHub
    2. Clones/updates all repositories
    3. Validates repository integrity
    4. Optionally runs analysis and checks
    
    Args:
        github_token: GitHub API token for authentication
        base_dir: Base directory for workspace (defaults to config)
        force_reclone: Whether to force re-clone existing repos
        validate_repos: Whether to validate repository integrity
        check_analysis: Whether to run analysis validation
        
    Returns:
        Dict containing comprehensive results:
            - success: bool
            - total_repositories: int
            - updated_repositories: int
            - new_repositories: int
            - failed_updates: List[Dict]
            - validation_results: Dict
            - analysis_results: Dict (if check_analysis=True)
    """
    logger.info("="*70)
    logger.info("🚀 Starting Comprehensive Palantir Repository Update")
    logger.info("="*70)
    
    # Initialize orchestrator
    orchestrator = OpentirOrchestrator(
        base_dir=base_dir or config.base_dir,
        github_token=github_token or config.github_token
    )
    
    results = {
        "success": False,
        "total_repositories": 0,
        "updated_repositories": 0,
        "new_repositories": 0,
        "failed_updates": [],
        "validation_results": {},
        "analysis_results": {}
    }
    
    try:
        # Step 1: Fetch organization and repository information
        logger.info("\n📡 Step 1: Fetching latest repository information from GitHub...")
        org_info = await orchestrator._fetch_organization_info()
        
        if "error" in org_info:
            logger.error(f"Failed to fetch organization info: {org_info['error']}")
            results["error"] = f"Organization fetch failed: {org_info['error']}"
            return results
        
        logger.info(f"✅ Organization: {org_info.get('name', 'Palantir')}")
        logger.info(f"   Public repos: {org_info.get('public_repos', 0)}")
        
        # Step 2: Fetch repository list
        repo_info = await orchestrator._fetch_repository_list()
        total_repos = repo_info.get("total_count", 0)
        results["total_repositories"] = total_repos
        
        logger.info(f"✅ Found {total_repos} repositories on GitHub")
        logger.info(f"   Languages: {len(repo_info.get('by_language', {}))}")
        
        # Step 3: Clone/update all repositories
        logger.info("\n📥 Step 2: Cloning/updating all repositories...")
        clone_results = await orchestrator.repo_manager.clone_all_repositories(
            force_update=force_reclone
        )
        
        if not clone_results.get("success", False):
            logger.error("Repository cloning failed")
            results["error"] = "Repository cloning failed"
            return results
        
        results["updated_repositories"] = clone_results.get("existing_repositories", 0)
        results["new_repositories"] = clone_results.get("new_repositories", 0)
        failed_clones = clone_results.get("failures", [])
        results["failed_updates"] = failed_clones
        
        logger.info(f"✅ Repository update complete:")
        logger.info(f"   - Already existed: {results['updated_repositories']}")
        logger.info(f"   - Newly cloned: {results['new_repositories']}")
        logger.info(f"   - Failed: {len(failed_clones)}")
        
        if failed_clones:
            logger.warning(f"⚠️  {len(failed_clones)} repositories failed to update:")
            for failure in failed_clones[:10]:  # Show first 10
                logger.warning(f"   - {failure.get('name', 'unknown')}: {failure.get('error', 'unknown error')}")
        
        # Step 4: Validate repositories
        if validate_repos:
            logger.info("\n🔍 Step 3: Validating repository integrity...")
            validation_results = await validate_all_repositories(orchestrator)
            results["validation_results"] = validation_results
            if validation_results.get("invalid_count", 0):
                results["error"] = "Repository validation failed"
                return results
            
            logger.info(f"✅ Validation complete:")
            logger.info(f"   - Valid repositories: {validation_results.get('valid_count', 0)}")
            logger.info(f"   - Invalid repositories: {validation_results.get('invalid_count', 0)}")
            logger.info(f"   - Total size: {validation_results.get('total_size_mb', 0):.1f} MB")
        
        # Step 5: Run analysis check (optional)
        if check_analysis:
            logger.info("\n📊 Step 4: Running analysis validation...")
            try:
                analysis_results = await orchestrator._analyze_all_repositories()
                results["analysis_results"] = {
                    "repositories_analyzed": analysis_results.get("total_repositories_analyzed", 0),
                    "total_elements": analysis_results.get("total_elements_extracted", 0),
                    "failed_analyses": analysis_results.get("failed_analyses", 0)
                }
                
                logger.info(f"✅ Analysis validation complete:")
                logger.info(f"   - Repositories analyzed: {analysis_results.get('total_repositories_analyzed', 0)}")
                logger.info(f"   - Total elements: {analysis_results.get('total_elements_extracted', 0):,}")
            except Exception as e:
                logger.warning(f"⚠️  Analysis check failed: {e}")
                results["analysis_results"] = {"error": str(e)}
                results["error"] = "Analysis validation failed"
                return results
        
        results["success"] = True
        
        logger.info("\n" + "="*70)
        logger.info("✅ Comprehensive repository update completed successfully!")
        logger.info("="*70)
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Update process failed: {e}")
        results["error"] = str(e)
        results["success"] = False
        return results

async def validate_all_repositories(orchestrator: OpentirOrchestrator) -> Dict[str, Any]:
    """
    Validate all cloned repositories for integrity.
    
    Checks:
    - Repository directory exists
    - Repository has valid git structure
    - Repository has code files
    - Repository size is reasonable
    
    Args:
        orchestrator: OpentirOrchestrator instance
        
    Returns:
        Dict containing validation results
    """
    from git import Repo, InvalidGitRepositoryError
    
    repos_dir = orchestrator.base_dir / config.repos_dir / "all_repos"
    
    if not repos_dir.exists():
        return {
            "valid_count": 0,
            "invalid_count": 0,
            "total_size_mb": 0,
            "errors": ["Repository directory does not exist"]
        }
    
    valid_repos = []
    invalid_repos = []
    total_size = 0
    
    for repo_dir in repos_dir.iterdir():
        if not repo_dir.is_dir() or repo_dir.name.startswith('.'):
            continue
        
        repo_name = repo_dir.name
        is_valid = True
        errors = []
        
        # Check if directory exists and has content
        if not repo_dir.exists():
            is_valid = False
            errors.append("Directory does not exist")
        
        # Check if it's a valid git repository
        try:
            repo = Repo(repo_dir)
            if repo.bare:
                is_valid = False
                errors.append("Repository is bare")
        except (InvalidGitRepositoryError, Exception) as e:
            is_valid = False
            errors.append(f"Invalid git repository: {str(e)}")
        
        # Check if repository has code files
        code_files = list(repo_dir.rglob("*.py")) + list(repo_dir.rglob("*.java")) + \
                     list(repo_dir.rglob("*.js")) + list(repo_dir.rglob("*.ts")) + \
                     list(repo_dir.rglob("*.go"))
        
        if len(code_files) == 0:
            is_valid = False
            errors.append("No code files found")
        
        # Calculate repository size
        try:
            repo_size = sum(f.stat().st_size for f in repo_dir.rglob('*') if f.is_file())
            repo_size_mb = repo_size / (1024 * 1024)
            total_size += repo_size_mb
        except Exception:
            repo_size_mb = 0
        
        if is_valid:
            valid_repos.append({
                "name": repo_name,
                "size_mb": repo_size_mb,
                "code_files": len(code_files)
            })
        else:
            invalid_repos.append({
                "name": repo_name,
                "errors": errors,
                "size_mb": repo_size_mb
            })
    
    return {
        "valid_count": len(valid_repos),
        "invalid_count": len(invalid_repos),
        "total_size_mb": total_size,
        "valid_repos": valid_repos[:10],  # Sample of valid repos
        "invalid_repos": invalid_repos[:10],  # Sample of invalid repos
        "validation_method": "comprehensive_integrity_check"
    }

def main():
    """Main entry point for the update script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Update all Palantir repositories with comprehensive validation"
    )
    parser.add_argument(
        "--token", "-t",
        help="GitHub API token",
        default=None
    )
    parser.add_argument(
        "--base-dir", "-b",
        help="Base directory for workspace",
        type=Path,
        default=None
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-clone all repositories"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip repository validation step"
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip analysis validation step"
    )
    
    args = parser.parse_args()
    
    # Get token from environment if not provided
    github_token = args.token or config.github_token
    
    if not github_token:
        logger.warning("⚠️  No GitHub token provided. Rate limits may apply.")
        logger.info("   Set GITHUB_TOKEN environment variable or use --token")
    
    # Run update
    results = asyncio.run(update_all_palantir_repos(
        github_token=github_token,
        base_dir=args.base_dir,
        force_reclone=args.force,
        validate_repos=not args.skip_validation,
        check_analysis=not args.skip_analysis
    ))
    
    # Print summary
    if results.get("success"):
        print("\n📊 Summary:")
        print(f"   Total repositories: {results['total_repositories']}")
        print(f"   Updated: {results['updated_repositories']}")
        print(f"   New: {results['new_repositories']}")
        print(f"   Failed: {len(results['failed_updates'])}")
        
        if results.get("validation_results"):
            val = results["validation_results"]
            print(f"\n   Valid repos: {val.get('valid_count', 0)}")
            print(f"   Invalid repos: {val.get('invalid_count', 0)}")
            print(f"   Total size: {val.get('total_size_mb', 0):.1f} MB")
        
        sys.exit(0)
    else:
        print(f"\n❌ Update failed: {results.get('error', 'Unknown error')}")
        sys.exit(1)

if __name__ == "__main__":
    main()


