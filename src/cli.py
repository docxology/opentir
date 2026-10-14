"""
Command-line interface for Opentir package.
Provides comprehensive CLI for managing Palantir ecosystem analysis.
"""

import click
import asyncio
import json
import os
import sys
from dataclasses import asdict
from functools import wraps
from pathlib import Path
from typing import Optional

from .github_client import GitHubClient
from .repo_manager import RepositoryManager
from .code_analyzer import CodeAnalyzer, RepositoryAnalysis
from .docs_generator import DocumentationGenerator
from .multi_repo_analyzer import MultiRepositoryAnalyzer
from .utils import Logger, ConfigManager
from .config import config, ensure_directories, update_config
from . import __version__

# Global logger
logger = Logger("opentir_cli")

def prompt_for_github_token() -> Optional[str]:
    """
    Interactive prompt for GitHub token with helpful instructions.
    Returns token if provided, None if user chooses to skip.
    """
    if not sys.stdin.isatty():
        return None
    click.echo("\n" + "="*70)
    click.echo("🔑 GitHub API Token Setup")
    click.echo("="*70)
    click.echo()
    click.echo("For best performance and to avoid rate limits, a GitHub API token is recommended.")
    click.echo("Without a token, you'll be limited to 60 requests per hour.")
    click.echo("With a token, you get 5,000 requests per hour.")
    click.echo()
    click.echo("📖 How to get a GitHub token:")
    click.echo("   1. Go to: https://github.com/settings/tokens")
    click.echo("   2. Click 'Generate new token' > 'Generate new token (classic)'")
    click.echo("   3. Give it a name like 'opentir-access'")
    click.echo("   4. Select scope: 'public_repo' (read access to public repositories)")
    click.echo("   5. Click 'Generate token' and copy the token")
    click.echo()
    click.echo("🔒 Note: Your token will only be used for this session and not stored.")
    click.echo()
    
    # Give user options for how to provide token
    click.echo("🔧 How would you like to provide your token?")
    click.echo("   1. Paste it securely (hidden input)")
    click.echo("   2. Paste it securely (hidden input)")
    click.echo("   3. Skip and continue without token")
    click.echo()
    
    choice = click.prompt(
        "Choose option (1/2/3)",
        default="3",
        show_default=False
    )
    
    token = ""
    if choice == "1":
        token = click.prompt(
            "Paste your GitHub token here",
            default="",
            hide_input=True,
            show_default=False
        )
    elif choice == "2":
        token = click.prompt(
            "Paste your GitHub token here (hidden)",
            default="",
            hide_input=True,
            show_default=False
        )
    else:
        token = ""
    
    if token.strip():
        click.echo("✅ Token provided! Using authenticated API access.")
        return token.strip()
    else:
        click.echo("⚠️  Continuing without token (rate-limited access).")
        click.echo("   If you hit rate limits, re-run with: opentir build-complete --token YOUR_TOKEN")
        return None

@click.group()
@click.version_option(version=__version__)
@click.option('--config-file', '-c', help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def main(ctx, config_file, verbose):
    """Opentir - Comprehensive Palantir OSS Ecosystem Tool"""
    ctx.ensure_object(dict)
    
    if verbose:
        config.log_level = "DEBUG"
    
    if config_file:
        try:
            with open(config_file, encoding="utf-8") as stream:
                update_config(**json.load(stream))
        except (OSError, ValueError, TypeError) as exc:
            raise click.ClickException(f"Invalid configuration: {exc}") from exc
    
    ensure_directories()
    logger._setup_logger()
    logger.info("Opentir CLI initialized")

@main.command()
@click.option('--token', '-t', help='GitHub API token')
@click.option('--output', '-o', help='Output file for repository data')
def fetch_repos(token, output):
    """Fetch all Palantir repositories from GitHub."""
    logger.info("Starting repository fetch...")
    
    # Check for GitHub token - prompt if not provided
    if not token and not config.github_token and not os.getenv('GITHUB_TOKEN'):
        token = prompt_for_github_token()
    
    client = GitHubClient(token)
    
    try:
        # Get organization info
        org_info = client.get_organization_info()
        if "error" in org_info:
            raise click.ClickException(f"Organization lookup failed: {org_info['error']}")
        logger.info(f"Fetching repos for: {org_info.get('name', 'Palantir')}")
        
        # Get all repositories
        repositories = client.get_all_repositories()
        
        result = {
            "organization": org_info,
            "repositories": [repo.__dict__ for repo in repositories],
            "total_count": len(repositories),
        }
        
        # Save to file if specified
        if output:
            output_path = Path(output)
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            logger.info(f"Repository data saved to: {output_path}")
        else:
            click.echo(json.dumps(result, indent=2, default=str))
        
        logger.info(f"Successfully fetched {len(repositories)} repositories")
        
    except Exception as e:
        logger.error(f"Error fetching repositories: {e}")
        raise click.ClickException(str(e))

@main.command()
@click.option('--force', '-f', is_flag=True, help='Force update existing repositories')
@click.option('--token', '-t', help='GitHub API token')
async def clone_all(force, token):
    """Clone all Palantir repositories to local repos/ directory."""
    logger.info("Starting comprehensive repository cloning...")
    
    try:
        # Check for GitHub token - prompt if not provided
        if not token and not config.github_token and not os.getenv('GITHUB_TOKEN'):
            token = prompt_for_github_token()
            
        repo_manager = RepositoryManager()
        
        # Override GitHub client token if provided
        if token:
            repo_manager.github_client = GitHubClient(token)
        
        # Clone all repositories
        results = await repo_manager.clone_all_repositories(force_update=force)
        
        if results["success"]:
            click.echo(f"✅ Successfully cloned {results['successful_clones']}/{results['total_repositories']} repositories")
            click.echo(f"📊 Total size: {results['total_size_mb']:.1f} MB")
            
            if results["failed_clones"]:
                click.echo(f"❌ Failed clones: {len(results['failed_clones'])}")
                for failure in results["failed_clones"][:5]:  # Show first 5 failures
                    click.echo(f"   - {failure['name']}: {failure['error']}")
        else:
            raise click.ClickException("Repository cloning failed")
            
    except Exception as e:
        logger.error(f"Error during cloning: {e}")
        raise click.ClickException(str(e))

@main.command()
@click.option('--repo-path', '-r', help='Specific repository path to analyze')
@click.option('--output-dir', '-o', help='Output directory for analysis results')
def analyze(repo_path, output_dir):
    """Analyze code in repositories and extract functionality."""
    logger.info("Starting comprehensive code analysis...")
    
    try:
        analyzer = CodeAnalyzer()
        output_path = Path(output_dir) if output_dir else config.base_dir / "analysis_results"
        output_path.mkdir(parents=True, exist_ok=True)
        
        if repo_path:
            # Analyze single repository
            repo_analysis = analyzer.analyze_repository(Path(repo_path))
            
            result_file = output_path / f"{repo_analysis.repo_name}_analysis.json"
            with open(result_file, 'w') as f:
                json.dump(asdict(repo_analysis), f, indent=2)

            with open(output_path / "all_analyses.json", 'w', encoding='utf-8') as f:
                json.dump([asdict(repo_analysis)], f, indent=2)
            with open(output_path / "functionality_matrix.json", 'w', encoding='utf-8') as f:
                json.dump(analyzer.generate_functionality_matrix([repo_analysis]), f, indent=2)
            
            click.echo(f"✅ Analysis completed for {repo_analysis.repo_name}")
            click.echo(f"📊 Found {len(repo_analysis.all_elements)} code elements")
            click.echo(f"📁 Results saved to: {result_file}")
            
        else:
            # Analyze all repositories
            repos_dir = config.base_dir / config.repos_dir / "all_repos"
            
            if not repos_dir.exists():
                raise click.ClickException("No repositories found. Run 'clone-all' first.")
            
            analyses = []
            total_repos = len(list(repos_dir.iterdir()))
            
            with click.progressbar(repos_dir.iterdir(), length=total_repos, label="Analyzing repositories") as repos:
                for repo_dir in repos:
                    if repo_dir.is_dir() and not repo_dir.name.startswith('.'):
                        try:
                            analysis = analyzer.analyze_repository(repo_dir)
                            analyses.append(analysis)
                        except Exception as e:
                            logger.warning(f"Failed to analyze {repo_dir.name}: {e}")
            
            # Generate functionality matrix
            functionality_matrix = analyzer.generate_functionality_matrix(analyses)
            
            # Save results
            analyses_file = output_path / "all_analyses.json"
            matrix_file = output_path / "functionality_matrix.json"
            
            with open(analyses_file, 'w') as f:
                json.dump([asdict(a) for a in analyses], f, indent=2)
            
            with open(matrix_file, 'w') as f:
                json.dump(functionality_matrix, f, indent=2, default=str)
            
            click.echo(f"✅ Analyzed {len(analyses)} repositories")
            click.echo(f"📊 Total elements extracted: {sum(len(a.all_elements) for a in analyses)}")
            click.echo(f"📁 Results saved to: {output_path}")
            
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        raise click.ClickException(str(e))

@main.command()
@click.option('--analysis-dir', '-a', help='Directory with analysis results')
@click.option('--output-dir', '-o', help='Output directory for documentation')
@click.option('--token', '-t', help='GitHub API token')
@click.option('--offline', is_flag=True, help='Generate from saved analysis without GitHub access')
def generate_docs(analysis_dir, output_dir, token, offline):
    """Generate comprehensive documentation with vast functionality tables."""
    logger.info("Starting documentation generation...")
    
    try:
        # Setup paths
        analysis_path = Path(analysis_dir) if analysis_dir else config.base_dir / "analysis_results"
        docs_path = Path(output_dir) if output_dir else config.base_dir / config.docs_dir
        
        # Load analysis results
        analyses_file = analysis_path / "all_analyses.json"
        matrix_file = analysis_path / "functionality_matrix.json"
        
        if not analyses_file.exists():
            raise click.ClickException("Analysis results not found. Run 'analyze' first.")
        
        with open(analyses_file, 'r') as f:
            analyses_data = json.load(f)
        
        with open(matrix_file, 'r') as f:
            functionality_matrix = json.load(f)
        
        # Get organization info
        if offline:
            org_info = {"name": config.github_org, "login": config.github_org}
        else:
            github_client = GitHubClient(token)
            org_info = github_client.get_organization_info(config.github_org)
            if "error" in org_info:
                raise click.ClickException(f"Organization lookup failed: {org_info['error']}")
        
        # Generate documentation
        docs_generator = DocumentationGenerator(docs_path)
        
        # Convert analyses data back to objects (simplified)
        from pydantic import TypeAdapter
        analyses = TypeAdapter(list[RepositoryAnalysis]).validate_python(analyses_data)
        
        results = docs_generator.generate_complete_documentation(
            analyses, functionality_matrix, org_info
        )
        
        click.echo(f"✅ Documentation generated successfully")
        click.echo(f"📁 Documentation location: {docs_path}")
        click.echo(f"📊 Generated {len(results)} documentation components")
        
        # Suggest next steps
        click.echo("\n🚀 Next steps:")
        click.echo(f"   uv run mkdocs serve -f {results['mkdocs_config']}")
        
    except Exception as e:
        logger.error(f"Error generating documentation: {e}")
        raise click.ClickException(str(e))

@main.command()
@click.option('--token', '-t', help='GitHub API token')
async def build_complete(token):
    """Complete build: fetch, clone, analyze, and generate docs."""
    logger.info("Starting complete Opentir build process...")
    
    try:
        # Check for GitHub token - prompt if not provided
        if not token and not config.github_token and not os.getenv('GITHUB_TOKEN'):
            token = prompt_for_github_token()
        
        # Step 1: Clone all repositories
        click.echo("🔄 Step 1: Cloning all repositories...")
        repo_manager = RepositoryManager()
        
        if token:
            repo_manager.github_client = GitHubClient(token)
        
        clone_results = await repo_manager.clone_all_repositories()
        
        if not clone_results["success"]:
            raise click.ClickException("Repository cloning failed")
        
        click.echo(f"✅ Cloned {clone_results['successful_clones']} repositories")
        
        # Step 2: Analyze all repositories
        click.echo("🔄 Step 2: Analyzing all repositories...")
        analyzer = CodeAnalyzer()
        repos_dir = config.base_dir / config.repos_dir / "all_repos"
        
        analyses = []
        for repo_dir in repos_dir.iterdir():
            if repo_dir.is_dir() and not repo_dir.name.startswith('.'):
                try:
                    analysis = analyzer.analyze_repository(repo_dir)
                    analyses.append(analysis)
                except Exception as e:
                    logger.warning(f"Failed to analyze {repo_dir.name}: {e}")
        
        functionality_matrix = analyzer.generate_functionality_matrix(analyses)
        click.echo(f"✅ Analyzed {len(analyses)} repositories")
        
        # Step 3: Generate documentation
        click.echo("🔄 Step 3: Generating comprehensive documentation...")
        github_client = GitHubClient(token)
        org_info = github_client.get_organization_info()
        if "error" in org_info:
            raise click.ClickException(f"Organization lookup failed: {org_info['error']}")
        
        docs_generator = DocumentationGenerator()
        docs_results = docs_generator.generate_complete_documentation(
            analyses, functionality_matrix, org_info
        )
        
        click.echo(f"✅ Generated comprehensive documentation")
        
        # Final summary
        click.echo("\n🎉 Complete build finished successfully!")
        click.echo(f"📊 Summary:")
        click.echo(f"   - Repositories cloned: {clone_results['successful_clones']}")
        click.echo(f"   - Repositories analyzed: {len(analyses)}")
        click.echo(f"   - Total code elements: {sum(len(a.all_elements) for a in analyses)}")
        click.echo(f"   - Documentation components: {len(docs_results)}")
        
        # Show directory structure
        click.echo(f"\n📁 Project structure:")
        click.echo(f"   - repos/: {config.base_dir / config.repos_dir}")
        click.echo(f"   - docs/: {config.base_dir / config.docs_dir}")
        click.echo(f"   - logs/: {config.base_dir / config.logs_dir}")
        
    except Exception as e:
        logger.error(f"Error during complete build: {e}")
        raise click.ClickException(str(e))

@main.command()
def status():
    """Show current status of Opentir workspace."""
    click.echo("📊 Opentir Workspace Status\n")
    
    # Repository status
    repos_dir = config.base_dir / config.repos_dir
    if repos_dir.exists():
        all_repos = list((repos_dir / "all_repos").iterdir()) if (repos_dir / "all_repos").exists() else []
        click.echo(f"📁 Repositories: {len(all_repos)} cloned")
        
        # Show organization structure
        by_lang_dir = repos_dir / "by_language"
        if by_lang_dir.exists():
            languages = [d.name for d in by_lang_dir.iterdir() if d.is_dir()]
            click.echo(f"🔤 Languages: {', '.join(languages)}")
    else:
        click.echo("📁 Repositories: Not initialized")
    
    # Documentation status
    docs_dir = config.base_dir / config.docs_dir
    if docs_dir.exists() and (docs_dir / "index.md").exists():
        click.echo(f"📚 Documentation: Generated")
        click.echo(f"   Location: {docs_dir}")
    else:
        click.echo("📚 Documentation: Not generated")
    
    # Analysis status
    analysis_dir = config.base_dir / "analysis_results"
    if analysis_dir.exists() and (analysis_dir / "functionality_matrix.json").exists():
        click.echo(f"🔍 Analysis: Completed")
    else:
        click.echo("🔍 Analysis: Not completed")
    
    # Configuration
    click.echo(f"\n⚙️  Configuration:")
    click.echo(f"   Base directory: {config.base_dir}")
    click.echo(f"   GitHub token: {'Set' if config.github_token else 'Not set'}")
    click.echo(f"   Log level: {config.log_level}")

@main.command()
@click.option('--force', '-f', is_flag=True, help='Force re-analysis of existing results')
@click.option('--output-dir', '-o', help='Output directory for analysis results')
def multi_analysis(force, output_dir):
    """Run comprehensive multi-repository analysis with visualizations."""
    logger.info("Starting comprehensive multi-repository analysis...")
    
    try:
        # Initialize multi-repo analyzer
        analyzer = MultiRepositoryAnalyzer()
        
        if output_dir:
            analyzer.output_dir = Path(output_dir)
            analyzer.output_dir.mkdir(parents=True, exist_ok=True)
        
        click.echo("🔍 Step 1: Analyzing all repositories...")
        # Analyze all repositories
        metrics = analyzer.analyze_all_repositories(force_reanalyze=force)
        
        click.echo("📊 Step 2: Generating visualizations...")
        # Generate visualizations
        visualizations = analyzer.generate_comprehensive_visualizations(metrics)
        
        click.echo("📋 Step 3: Generating reports...")
        # Generate reports
        reports = analyzer.generate_comprehensive_reports(metrics)
        
        click.echo("🔍 Step 4: Extracting patterns...")
        # Extract patterns
        patterns = analyzer.extract_cross_repo_patterns(metrics)
        
        # Display summary
        click.echo("\n✅ Multi-repository analysis completed successfully!")
        click.echo(f"📊 Summary:")
        click.echo(f"   - Repositories analyzed: {metrics.total_repositories}")
        click.echo(f"   - Total files: {metrics.total_files:,}")
        click.echo(f"   - Total lines of code: {metrics.total_lines_of_code:,}")
        click.echo(f"   - Total code elements: {metrics.total_code_elements:,}")
        click.echo(f"   - Languages detected: {len(metrics.language_distribution)}")
        
        click.echo(f"\n📁 Results saved to: {analyzer.output_dir}")
        click.echo(f"   - Visualizations: {len(visualizations)} charts")
        click.echo(f"   - Reports: {len(reports)} documents")
        click.echo(f"   - Patterns: {len(patterns)} pattern types")
        
        # Show top languages
        sorted_langs = sorted(metrics.language_distribution.items(), key=lambda x: x[1], reverse=True)[:5]
        click.echo(f"\n🔤 Top languages:")
        for lang, count in sorted_langs:
            click.echo(f"   - {lang}: {count} files")
        
    except Exception as e:
        logger.error(f"Error during multi-repository analysis: {e}")
        raise click.ClickException(str(e))

@main.command()
@click.option('--keep-popular', is_flag=True, help='Keep popular repositories')
@click.confirmation_option(prompt='Are you sure you want to clean up repositories?')
def cleanup(keep_popular):
    """Clean up downloaded repositories."""
    try:
        repo_manager = RepositoryManager()
        results = repo_manager.cleanup_repositories(keep_popular=keep_popular)
        
        if "error" not in results:
            click.echo(f"✅ Cleaned up {results['removed_repositories']} repositories")
            click.echo(f"💾 Freed {results['total_size_freed_mb']:.1f} MB")
        else:
            raise click.ClickException(results["error"])
            
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        raise click.ClickException(str(e))

# Async command wrapper
def async_command(f):
    """Wrapper to run async commands."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapper

# Apply async wrapper to async commands
clone_all.callback = async_command(clone_all.callback)
build_complete.callback = async_command(build_complete.callback)

if __name__ == '__main__':
    main()
