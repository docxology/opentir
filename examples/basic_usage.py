#!/usr/bin/env python3
"""
Basic example demonstrating how to use Opentir to analyze Palantir's open source ecosystem.
"""

import asyncio
from pathlib import Path
from src.main import build_complete_ecosystem, get_workspace_status

async def main():
    # Set your GitHub token (recommended for better rate limits)
    github_token = "your_github_token_here"  # Replace with your token
    
    # Build the complete ecosystem
    results = await build_complete_ecosystem(
        github_token=github_token,
        force_reclone=False  # Set to True to force re-cloning of repositories
    )
    
    # Print summary
    summary = results["summary"]
    print(f"\n🎉 Analysis Complete!")
    print(f"📊 Repositories analyzed: {summary['repositories_analyzed']}")
    print(f"📝 Total files: {summary['total_files']}")
    print(f"📚 Total code elements: {summary['total_elements']}")
    print(f"⏱️  Execution time: {results['execution_time_formatted']}")
    
    # Get workspace status
    status = get_workspace_status()
    print(f"\n📂 Workspace Status:")
    print(f"Repository directory: {status['repository_status']['repos_dir']}")
    print(f"Documentation directory: {status['documentation_status']['docs_dir']}")
    print(f"Total repositories: {status['repository_status']['total_repositories']}")

if __name__ == "__main__":
    asyncio.run(main()) 