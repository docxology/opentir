#!/usr/bin/env python3
"""
Main entry point for Opentir - Comprehensive Palantir OSS Analysis.
Works with existing repositories and runs comprehensive multi-repository analysis.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.main import OpentirOrchestrator
from src.multi_repo_analyzer import MultiRepositoryAnalyzer
from src.utils import Logger


async def main():
    """
    Main execution function for Opentir.
    Works with existing repositories and runs comprehensive multi-repository analysis.
    """
    logger = Logger("main")
    
    print("🚀 Opentir - Comprehensive Palantir OSS Analysis")
    print("=" * 60)
    
    try:
        # Initialize orchestrator
        orchestrator = OpentirOrchestrator()
        
        # Step 1: Check workspace status
        print("\n📊 Step 1: Checking workspace status...")
        status = orchestrator.get_workspace_status()
        
        print(f"Current status:")
        print(f"  - Local repositories: {status['repositories']['count']}")
        print(f"  - Analysis cache: {'✅ Found' if status['analysis']['status'] == 'completed' else '❌ Missing'}")
        print(f"  - Documentation: {'✅ Found' if status['documentation']['status'] == 'generated' else '❌ Missing'}")
        
        # Step 2: Skip repository updates for speed
        print("\n⚡ Step 2: Using existing repositories (skipping updates for speed)...")
        
        # Check if we have any repositories to work with
        repos_dir = Path("repos")
        if not repos_dir.exists() or not any(repos_dir.iterdir()):
            print("  ❌ No repositories found in 'repos' directory.")
            print("  💡 Run the full workflow first to download repositories:")
            print("     python -c \"from src.main import OpentirOrchestrator; import asyncio; asyncio.run(OpentirOrchestrator().run_workflow())\"")
            return
        
        # Count existing repositories
        existing_repos = list(repos_dir.rglob("*/.git"))
        print(f"  ✅ Found {len(existing_repos)} existing repositories")
        print(f"  ⚡ Skipping repository updates for faster analysis")
        
        # Step 3: Comprehensive multi-repository analysis
        print("\n🔍 Step 3: Running comprehensive multi-repository analysis...")
        
        analyzer = MultiRepositoryAnalyzer()
        
        # Use comprehensive analysis (with smart caching for performance)
        print(f"  📊 Running comprehensive analysis across all repositories...")
        
        # Run comprehensive analysis
        metrics = analyzer.analyze_all_repositories(force_reanalyze=False)
        
        print(f"\nAnalysis completed!")
        print(f"  - Repositories analyzed: {metrics.total_repositories}")
        print(f"  - Total files: {metrics.total_files:,}")
        print(f"  - Total lines of code: {metrics.total_lines_of_code:,}")
        print(f"  - Total code elements: {metrics.total_code_elements:,}")
        print(f"  - Languages detected: {len(metrics.language_distribution)}")
        
        # Step 4: Generate visualizations and reports
        print("\n📊 Step 4: Generating visualizations and reports...")
        
        visualizations = analyzer.generate_comprehensive_visualizations(metrics)
        reports = analyzer.generate_comprehensive_reports(metrics)
        patterns = analyzer.extract_cross_repo_patterns(metrics)
        
        print(f"Generated outputs:")
        print(f"  - Visualizations: {len(visualizations)} charts")
        print(f"  - Reports: {len(reports)} documents")
        print(f"  - Pattern analyses: {len(patterns)} types")
        print(f"  - Output directory: {analyzer.output_dir}")
        
        # Step 5: Summary
        print("\n✅ Complete analysis finished!")
        print(f"📁 All results saved to: {analyzer.output_dir}")
        
        # Show top languages
        if metrics.language_distribution:
            sorted_langs = sorted(metrics.language_distribution.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"\n🔤 Top programming languages:")
            for i, (lang, count) in enumerate(sorted_langs, 1):
                percentage = (count / sum(metrics.language_distribution.values())) * 100
                print(f"  {i}. {lang}: {count:,} files ({percentage:.1f}%)")
        
        print(f"\n🎯 Next steps:")
        print(f"  - View visualizations: open {analyzer.output_dir}/visualizations/")
        print(f"  - Read reports: open {analyzer.output_dir}/reports/")
        print(f"  - Explore data: check {analyzer.output_dir}/reports/csv_exports/")
        print(f"  - Run advanced analysis: python examples/advanced_statistical_analysis.py")
        
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        print(f"\n❌ Error: {e}")
        print(f"\n💡 Troubleshooting tips:")
        print(f"  - Ensure repositories exist in 'repos' directory")
        print(f"  - Check logs in 'logs/opentir.log' for details")
        print(f"  - Run with verbose logging: python -c \"import logging; logging.basicConfig(level=logging.DEBUG); exec(open('main.py').read())\"")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 