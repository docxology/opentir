#!/usr/bin/env python3
"""
Comprehensive workflow script for complete Palantir ecosystem analysis.
Combines repository updates, analysis, and documentation generation.
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.main import OpentirOrchestrator
from src.multi_repo_analyzer import MultiRepositoryAnalyzer
from src.utils import Logger
from src.config import config

logger = Logger("comprehensive_workflow")

async def run_comprehensive_analysis(
    github_token: Optional[str] = None,
    base_dir: Optional[Path] = None,
    force_reclone: bool = False,
    skip_docs: bool = False
) -> dict:
    """
    Run comprehensive Palantir ecosystem analysis workflow.
    
    This function orchestrates the complete analysis pipeline:
    1. Update all repositories from GitHub
    2. Analyze all repositories
    3. Generate cross-repository metrics
    4. Extract patterns and insights
    5. Generate visualizations and reports
    6. Generate documentation (optional)
    
    Args:
        github_token: GitHub API token for authentication
        base_dir: Base directory for workspace
        force_reclone: Force re-clone all repositories
        skip_docs: Skip documentation generation
        
    Returns:
        Dict containing comprehensive results from all steps
    """
    logger.info("="*70)
    logger.info("🚀 Starting Comprehensive Palantir Ecosystem Analysis")
    logger.info("="*70)
    
    # Initialize orchestrator
    orchestrator = OpentirOrchestrator(
        base_dir=base_dir or config.base_dir,
        github_token=github_token or config.github_token
    )
    
    results = {
        "workflow": {},
        "multi_repo_analysis": {},
        "success": False
    }
    
    try:
        # Step 1: Complete workflow (fetch, clone, analyze, docs)
        logger.info("\n📋 Step 1: Running complete workflow...")
        workflow_results = await orchestrator.execute_complete_workflow(
            force_reclone=force_reclone,
            skip_docs=skip_docs
        )
        results["workflow"] = workflow_results
        
        if not workflow_results.get("success"):
            logger.error("Workflow failed, stopping analysis")
            return results
        
        logger.info(f"✅ Workflow completed: {workflow_results.get('summary', {})}")
        
        # Step 2: Multi-repository analysis
        logger.info("\n📊 Step 2: Running comprehensive multi-repository analysis...")
        multi_analyzer = MultiRepositoryAnalyzer(repos_base_path=orchestrator.base_dir / config.repos_dir)
        
        # Analyze all repositories
        metrics = multi_analyzer.analyze_all_repositories(force_reanalyze=False)
        
        logger.info(f"✅ Analyzed {metrics.total_repositories} repositories")
        logger.info(f"   Total files: {metrics.total_files:,}")
        logger.info(f"   Total lines: {metrics.total_lines_of_code:,}")
        logger.info(f"   Total elements: {metrics.total_code_elements:,}")
        
        # Generate visualizations
        logger.info("\n📈 Step 3: Generating visualizations...")
        visualizations = multi_analyzer.generate_comprehensive_visualizations(metrics)
        logger.info(f"✅ Generated {len(visualizations)} visualizations")
        
        # Generate reports
        logger.info("\n📋 Step 4: Generating reports...")
        reports = multi_analyzer.generate_comprehensive_reports(metrics)
        logger.info(f"✅ Generated {len(reports)} reports")
        
        # Extract patterns
        logger.info("\n🔍 Step 5: Extracting cross-repository patterns...")
        patterns = multi_analyzer.extract_cross_repo_patterns(metrics)
        logger.info(f"✅ Extracted {len(patterns)} pattern types")
        
        results["multi_repo_analysis"] = {
            "metrics": {
                "total_repositories": metrics.total_repositories,
                "total_files": metrics.total_files,
                "total_lines": metrics.total_lines_of_code,
                "total_elements": metrics.total_code_elements,
                "languages": len(metrics.language_distribution)
            },
            "visualizations": list(visualizations.keys()),
            "reports": list(reports.keys()),
            "patterns": list(patterns.keys())
        }
        
        results["success"] = True
        
        logger.info("\n" + "="*70)
        logger.info("✅ Comprehensive analysis completed successfully!")
        logger.info("="*70)
        
        # Print summary
        print_summary(results)
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Comprehensive analysis failed: {e}")
        results["error"] = str(e)
        results["success"] = False
        return results

def print_summary(results: dict) -> None:
    """Print comprehensive summary of results."""
    print("\n" + "="*70)
    print("📊 COMPREHENSIVE ANALYSIS SUMMARY")
    print("="*70)
    
    if results.get("success"):
        # Workflow summary
        workflow = results.get("workflow", {})
        summary = workflow.get("summary", {})
        
        print("\n📋 Workflow Results:")
        print(f"   ✅ Repositories cloned: {summary.get('total_repositories', 0)}")
        print(f"   ✅ Repositories analyzed: {summary.get('repositories_analyzed', 0)}")
        print(f"   ✅ Total elements: {summary.get('total_elements_extracted', 0):,}")
        print(f"   ✅ Documentation: {'Generated' if summary.get('documentation_generated') else 'Skipped'}")
        
        # Multi-repo analysis summary
        multi_analysis = results.get("multi_repo_analysis", {})
        metrics = multi_analysis.get("metrics", {})
        
        print("\n📊 Multi-Repository Analysis:")
        print(f"   ✅ Repositories: {metrics.get('total_repositories', 0)}")
        print(f"   ✅ Files: {metrics.get('total_files', 0):,}")
        print(f"   ✅ Lines of code: {metrics.get('total_lines', 0):,}")
        print(f"   ✅ Code elements: {metrics.get('total_elements', 0):,}")
        print(f"   ✅ Languages: {metrics.get('languages', 0)}")
        
        print(f"\n📈 Visualizations: {len(multi_analysis.get('visualizations', []))}")
        print(f"📋 Reports: {len(multi_analysis.get('reports', []))}")
        print(f"🔍 Pattern types: {len(multi_analysis.get('patterns', []))}")
        
        print("\n📁 Output Locations:")
        print(f"   - Repositories: {config.base_dir / config.repos_dir}")
        print(f"   - Analysis: {config.base_dir / 'multi_repo_analysis'}")
        print(f"   - Documentation: {config.base_dir / config.docs_dir}")
    else:
        print(f"\n❌ Analysis failed: {results.get('error', 'Unknown error')}")

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run comprehensive Palantir ecosystem analysis"
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
        "--skip-docs",
        action="store_true",
        help="Skip documentation generation"
    )
    
    args = parser.parse_args()
    
    # Get token from environment if not provided
    github_token = args.token or config.github_token
    
    if not github_token:
        logger.warning("⚠️  No GitHub token provided. Rate limits may apply.")
        logger.info("   Set GITHUB_TOKEN environment variable or use --token")
    
    # Run comprehensive analysis
    results = asyncio.run(run_comprehensive_analysis(
        github_token=github_token,
        base_dir=args.base_dir,
        force_reclone=args.force,
        skip_docs=args.skip_docs
    ))
    
    # Exit with appropriate code
    sys.exit(0 if results.get("success") else 1)

if __name__ == "__main__":
    main()


