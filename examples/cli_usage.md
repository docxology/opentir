# Opentir CLI Usage Guide

Opentir provides a powerful command-line interface for analyzing Palantir's open source ecosystem. This guide covers all available commands and their usage.

## Installation

### Quick Start (Recommended)

```bash
# Use the provided run script (handles virtual environment automatically)
./run.sh
```

### Manual Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

### Development Setup

```bash
# Install the package in development mode
pip install -e .
```

## GitHub Token Setup

Opentir needs GitHub API access to download repositories. You have several options:

### Option 1: Interactive Setup (Easiest)
```bash
opentir build-complete
```
Opentir will guide you through token setup with step-by-step instructions.

### Option 2: Set Environment Variable
```bash
export GITHUB_TOKEN=your_github_token_here
opentir build-complete
```

### Option 3: Command Line Argument
```bash
opentir build-complete --token your_github_token_here
```

**Getting a token**: Go to https://github.com/settings/tokens → Generate new token → Select `public_repo` scope

## Basic Commands

### Complete Workflow (Recommended)

Build and analyze the complete ecosystem:

```bash
# Run complete workflow with interactive token setup
opentir build-complete

# Alternative methods
opentir build-complete --token YOUR_GITHUB_TOKEN
export GITHUB_TOKEN=YOUR_TOKEN && opentir build-complete
```

### Step-by-Step Workflow

Run individual steps of the workflow:

```bash
# 1. Fetch repository information
opentir fetch-repos  # Interactive token setup if needed

# 2. Clone repositories
opentir clone-all  # Interactive token setup if needed
opentir clone-all --force  # Force re-clone

# 3. Analyze code
opentir analyze
opentir analyze --repo-path path/to/repo  # Analyze specific repository

# 4. Generate documentation
opentir generate-docs  # Interactive token setup if needed
```

**Note**: All commands that need GitHub access will prompt for a token if one isn't already configured.

## Status and Information

Check workspace and analysis status:

```bash
# Get workspace status
opentir status

# Get repository information
opentir info repo_name

# List all repositories
opentir list-repos
opentir list-repos --by-language  # Group by language
opentir list-repos --by-category  # Group by category
```

## Configuration

Manage configuration settings:

```bash
# Set GitHub token
opentir config set github_token YOUR_GITHUB_TOKEN

# Set base directory
opentir config set base_dir /path/to/workspace

# View current configuration
opentir config show

# Reset configuration to defaults
opentir config reset
```

## Cleanup and Maintenance

Manage workspace and cached data:

```bash
# Clean workspace
opentir cleanup
opentir cleanup --keep-popular  # Keep popular repositories
opentir cleanup --older-than 30  # Clean repositories older than 30 days

# Clear cache
opentir clear-cache
```

## Advanced Usage

### Filtering and Selection

```bash
# Analyze specific languages
opentir analyze --languages python,java

# Filter by stars
opentir clone-repos --min-stars 100

# Filter by activity
opentir analyze --active-since "2023-01-01"
```

### Output Formats

```bash
# Generate JSON output
opentir analyze --output json

# Generate CSV reports
opentir generate-docs --format csv

# Generate detailed metrics
opentir analyze --detailed-metrics
```

### Parallel Processing

```bash
# Set concurrency for operations
opentir build-complete --workers 4

# Parallel repository cloning
opentir clone-repos --parallel
```

## Environment Variables

Opentir supports the following environment variables:

- `GITHUB_TOKEN`: GitHub API token
- `OPENTIR_BASE_DIR`: Base directory for workspace
- `OPENTIR_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `OPENTIR_CACHE_DIR`: Cache directory location

## Examples

### Basic Analysis

```bash
# Quick analysis of popular repositories
opentir build-complete --token YOUR_GITHUB_TOKEN --min-stars 1000

# Generate HTML documentation
opentir build-complete --token YOUR_GITHUB_TOKEN --docs-format html
```

### Focused Analysis

```bash
# Analyze specific repository with detailed metrics
opentir analyze --repo foundry --detailed-metrics

# Generate documentation for Java repositories
opentir generate-docs --languages java --format html
```

### Maintenance

```bash
# Update existing repositories
opentir update-repos

# Clean old data and regenerate
opentir cleanup && opentir build-complete
```

## Error Handling

Common error messages and solutions:

- `Rate limit exceeded`: Wait or use a GitHub token
- `Repository not found`: Check repository name
- `Permission denied`: Check token permissions
- `Invalid token`: Verify GitHub token

## Best Practices

1. Always use a GitHub token for better rate limits
2. Start with popular repositories first
3. Use parallel processing for large analyses
4. Regularly clean up old data
5. Check status before long operations

## Getting Help

```bash
# Show help
opentir --help

# Show command help
opentir COMMAND --help

# Show version
opentir --version
```

For more information, visit the [Opentir Documentation](https://github.com/username/opentir).

## Output Structure

After running the complete workflow, you'll have:

```
├── repos/                  # Cloned repositories
│   ├── all_repos/         # All repositories
│   ├── by_language/       # Organized by language
│   ├── by_category/       # Organized by category
│   └── popular/           # Popular repositories
├── docs/                  # Generated documentation
│   ├── index.md           # Main documentation
│   ├── repositories/      # Repository details
│   ├── api_reference/     # API documentation
│   ├── analysis/          # Analysis reports
│   └── mkdocs.yml         # MkDocs configuration
├── analysis_results/      # Raw analysis data
│   ├── repository_analyses.json
│   └── functionality_matrix.json
└── logs/                  # Application logs
    └── opentir.log
```

## Troubleshooting

### Rate Limiting

If you encounter rate limiting:

```bash
# Use GitHub token for higher limits
export GITHUB_TOKEN=your_token_here

# Check current status
opentir status
```

### Large Repository Size

```bash
# Clean up to free space
opentir cleanup --keep-popular

# Check repository sizes
opentir status
```

### Analysis Errors

```bash
# Run with verbose logging
opentir --verbose analyze

# Check logs
tail -f logs/opentir.log
```

## Tips

1. **GitHub Token**: Always use a GitHub token for better rate limits
2. **Disk Space**: The complete ecosystem requires ~5-10GB of storage
3. **Time**: Complete analysis can take 30-60 minutes depending on your system
4. **Documentation**: Generated docs are best viewed with MkDocs serve
5. **Updates**: Use `--force` flag to update existing repositories 