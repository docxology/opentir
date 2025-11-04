#!/bin/bash
# Comprehensive script to run full Palantir repository update and analysis
# This script performs: update, analyze, and generate documentation

set -e

echo "=========================================="
echo "🚀 OpenTIR Comprehensive Analysis Workflow"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: requirements.txt not found. Please run from project root."
    exit 1
fi

# Check if dependencies are installed
echo "📦 Checking dependencies..."
if ! python3 -c "import github" 2>/dev/null; then
    echo "⚠️  Dependencies not installed. Installing..."
    pip3 install -r requirements.txt
fi

# Get GitHub token if available
GITHUB_TOKEN="${GITHUB_TOKEN:-${1:-}}"

if [ -z "$GITHUB_TOKEN" ]; then
    echo "⚠️  No GitHub token provided."
    echo "   Set GITHUB_TOKEN environment variable or pass as first argument"
    echo "   Continuing without token (rate-limited access)..."
    TOKEN_ARG=""
else
    TOKEN_ARG="--token $GITHUB_TOKEN"
    echo "✅ GitHub token provided"
fi

echo ""
echo "📥 Step 1: Updating all Palantir repositories..."
echo "=========================================="
python3 scripts/update_all_repos.py $TOKEN_ARG --force

echo ""
echo "📊 Step 2: Running comprehensive analysis..."
echo "=========================================="
python3 scripts/comprehensive_workflow.py $TOKEN_ARG --force

echo ""
echo "✅ Complete analysis workflow finished!"
echo "=========================================="
echo ""
echo "📁 Results available in:"
echo "   - Repositories: repos/"
echo "   - Analysis: multi_repo_analysis/"
echo "   - Documentation: docs/"
echo ""


