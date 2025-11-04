#!/usr/bin/env python3
"""
Comprehensive workflow with intelligent rate limit and repository handling.
Will work with existing repos or wait/proceed when rate limits allow.
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.main import OpentirOrchestrator
from src.multi_repo_analyzer import MultiRepositoryAnalyzer
from src.utils import Logger
from src.config import config

logger = Logger("rate_limit_workflow")

async def check_rate_limit_and_proceed(github_token=None):
    """Check rate limit status and proceed intelligently."""
    from src.github_client import GitHubClient
    
    client = GitHubClient(github_token or config.github_token)
    
    try:
        rate_limit = client.github.get_rate_limit()
        
        # Try to get remaining requests
        remaining = 0
        if hasattr(rate_limit, 'core'):
            remaining = rate_limit.core.remaining
        elif hasattr(rate_limit, 'resources'):
            remaining = rate_limit.resources.core.remaining
        else:
            remaining = getattr(rate_limit, 'remaining', 0)
        
        logger.info(f"GitHub API rate limit: {remaining} requests remaining")
        
        if remaining < 10:
            logger.warning("⚠️  Rate limit very low or exhausted")
            logger.info("   Options:")
            logger.info("   1. Wait for rate limit reset (usually ~1 hour)")
            logger.info("   2. Provide GitHub token: export GITHUB_TOKEN=your_token")
            logger.info("   3. Continue with existing repositories if available")
            return False
        
        return True
    except Exception as e:
        logger.warning(f"Could not check rate limit: {e}")
        return True  # Proceed optimistically

async def run_comprehensive_workflow_with_fallback(github_token=None, force_reclone=False):
    """Run comprehensive workflow with intelligent fallbacks."""
    
    logger.info("="*70)
    logger.info("🚀 OpenTIR Comprehensive Analysis Workflow")
    logger.info("="*70)
    
    # Check rate limit
    can_fetch = await check_rate_limit_and_proceed(github_token)
    
    orchestrator = OpentirOrchestrator(
        github_token=github_token or config.github_token
    )
    
    # Check existing repositories
    status = orchestrator.get_workspace_status()
    existing_repos = status["repositories"].get("count", 0)
    
    logger.info(f"\n📊 Current Status:")
    logger.info(f"   - Existing repositories: {existing_repos}")
    logger.info(f"   - Rate limit OK: {can_fetch}")
    
    if existing_repos == 0 and not can_fetch:
        logger.error("\n❌ No repositories available and rate limit exhausted")
        logger.info("\n💡 Solutions:")
        logger.info("   1. Wait for rate limit reset (~1 hour)")
        logger.info("   2. Set GITHUB_TOKEN environment variable")
        logger.info("   3. Manually clone some repositories to repos/all_repos/")
        return {"success": False, "error": "No repositories and rate limit exhausted"}
    
    # If we have repos, proceed with analysis
    if existing_repos > 0:
        logger.info(f"\n✅ Found {existing_repos} existing repositories")
        logger.info("   Proceeding with analysis on existing repositories...")
        
        # Skip repository update, proceed directly to analysis
        results = await orchestrator.execute_complete_workflow(
            force_reclone=False,
            skip_analysis=False,
            skip_docs=False
        )
        
        if results.get("success"):
            # Run multi-repo analysis
            logger.info("\n📊 Running comprehensive multi-repository analysis...")
            multi_analyzer = MultiRepositoryAnalyzer()
            
            metrics = multi_analyzer.analyze_all_repositories(force_reanalyze=False)
            visualizations = multi_analyzer.generate_comprehensive_visualizations(metrics)
            reports = multi_analyzer.generate_comprehensive_reports(metrics)
            patterns = multi_analyzer.extract_cross_repo_patterns(metrics)
            
            logger.info("\n✅ Comprehensive analysis completed!")
            logger.info(f"   - Repositories analyzed: {metrics.total_repositories}")
            logger.info(f"   - Visualizations: {len(visualizations)}")
            logger.info(f"   - Reports: {len(reports)}")
            logger.info(f"   - Patterns: {len(patterns)}")
            
            return {
                "success": True,
                "workflow": results,
                "analysis": {
                    "repositories": metrics.total_repositories,
                    "files": metrics.total_files,
                    "lines": metrics.total_lines_of_code,
                    "elements": metrics.total_code_elements
                }
            }
    
    # If no repos and can fetch, try to update
    if can_fetch:
        logger.info("\n📥 Fetching and cloning repositories...")
        results = await orchestrator.execute_complete_workflow(
            force_reclone=force_reclone,
            skip_analysis=False,
            skip_docs=False
        )
        return results
    
    return {"success": False, "error": "Cannot proceed without repositories or rate limit"}

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run comprehensive workflow with rate limit handling")
    parser.add_argument("--token", "-t", help="GitHub API token")
    parser.add_argument("--force", "-f", action="store_true", help="Force re-clone")
    
    args = parser.parse_args()
    
    github_token = args.token or config.github_token
    
    results = asyncio.run(run_comprehensive_workflow_with_fallback(
        github_token=github_token,
        force_reclone=args.force
    ))
    
    if results.get("success"):
        print("\n✅ Workflow completed successfully!")
        sys.exit(0)
    else:
        print(f"\n❌ Workflow failed: {results.get('error', 'Unknown error')}")
        sys.exit(1)

