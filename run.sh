#!/bin/bash

# OpenTIR - Comprehensive Analysis Run Script
# This script orchestrates the complete Palantir OSS ecosystem analysis

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the project directory
cd "$SCRIPT_DIR"

print_info "Starting OpenTIR Comprehensive Analysis Framework"
echo "=================================================="

# Parse command line arguments
RUN_ADVANCED_STATS=false
FORCE_REANALYSIS=false
SKIP_DEPS_CHECK=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --advanced-stats|--stats)
            RUN_ADVANCED_STATS=true
            shift
            ;;
        --force-reanalysis|--force)
            FORCE_REANALYSIS=true
            shift
            ;;
        --skip-deps)
            SKIP_DEPS_CHECK=true
            shift
            ;;
        --help|-h)
            echo "OpenTIR Usage:"
            echo "  ./run.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --advanced-stats    Run comprehensive statistical analysis with ANOVA, PCA, clustering"
            echo "  --force-reanalysis  Force fresh analysis (ignore cached results)"
            echo "  --skip-deps        Skip dependency installation check"
            echo "  --help             Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./run.sh                           # Basic analysis"
            echo "  ./run.sh --advanced-stats          # Full statistical analysis"
            echo "  ./run.sh --force --advanced-stats  # Fresh comprehensive analysis"
            exit 0
            ;;
        *)
            print_warning "Unknown option: $1"
            print_info "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    print_info "Virtual environment not found. Creating one..."
    python3 -m venv venv
    print_status "Virtual environment created"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate

# Function to check if a Python package is installed
check_package() {
    python -c "import $1" 2>/dev/null
    return $?
}

# Install or upgrade dependencies if needed
if [ "$SKIP_DEPS_CHECK" = false ]; then
    print_info "Checking dependencies..."
    
    # Check core dependencies
    NEED_INSTALL=false
    
    if ! check_package "aiohttp"; then
        NEED_INSTALL=true
    fi
    
    if ! check_package "pandas"; then
        NEED_INSTALL=true
    fi
    
    if ! check_package "numpy"; then
        NEED_INSTALL=true
    fi
    
    if ! check_package "matplotlib"; then
        NEED_INSTALL=true
    fi
    
    if ! check_package "seaborn"; then
        NEED_INSTALL=true
    fi
    
    if ! check_package "scipy"; then
        NEED_INSTALL=true
    fi
    
    if ! check_package "sklearn"; then
        NEED_INSTALL=true
    fi
    
    if ! check_package "plotly"; then
        NEED_INSTALL=true
    fi
    
    if [ "$NEED_INSTALL" = true ]; then
        print_info "Installing/updating dependencies..."
        pip install --upgrade pip
        pip install -r requirements.txt
        print_status "Dependencies installed/updated"
    else
        print_status "All dependencies are available"
    fi
else
    print_warning "Skipping dependency check (--skip-deps flag used)"
fi

# Verify repository structure
print_info "Verifying project structure..."

if [ ! -d "src" ]; then
    print_error "src/ directory not found!"
    exit 1
fi

if [ ! -d "repos" ]; then
    print_warning "repos/ directory not found. Analysis will attempt to create it."
fi

if [ ! -f "src/main.py" ]; then
    print_error "src/main.py not found!"
    exit 1
fi

print_status "Project structure verified"

# Display analysis configuration
echo ""
print_info "Analysis Configuration:"
echo "  - Force Reanalysis: $FORCE_REANALYSIS"
echo "  - Advanced Statistics: $RUN_ADVANCED_STATS"
echo "  - Working Directory: $SCRIPT_DIR"
echo ""

# Function to run analysis with error handling
run_analysis() {
    local analysis_type=$1
    local script_path=$2
    
    print_info "Starting $analysis_type..."
    
    if [ "$FORCE_REANALYSIS" = true ]; then
        print_info "Force reanalysis enabled - clearing cache..."
        rm -f multi_repo_analysis/cross_repo_analysis.json
        rm -f multi_repo_analysis/cross_repo_patterns.json
    fi
    
    if python "$script_path" "$@"; then
        print_status "$analysis_type completed successfully"
        return 0
    else
        print_error "$analysis_type failed with exit code $?"
        return 1
    fi
}

# Run main analysis
echo "🚀 Starting Primary Analysis..."
echo "================================"

if run_analysis "Primary Repository Analysis" "main.py"; then
    print_status "Primary analysis completed"
    
    # Check if outputs were generated
    if [ -d "multi_repo_analysis" ]; then
        echo ""
        print_info "Generated outputs:"
        find multi_repo_analysis -name "*.md" -o -name "*.png" -o -name "*.json" -o -name "*.csv" | head -10 | while read file; do
            echo "  📄 $file"
        done
        
        file_count=$(find multi_repo_analysis -type f | wc -l)
        print_status "Total output files: $file_count"
    fi
else
    print_error "Primary analysis failed!"
    exit 1
fi

# Run advanced statistical analysis if requested
if [ "$RUN_ADVANCED_STATS" = true ]; then
    echo ""
    echo "📊 Starting Advanced Statistical Analysis..."
    echo "==========================================="
    
    if [ -f "examples/advanced_statistical_analysis.py" ]; then
        if run_analysis "Advanced Statistical Analysis" "examples/advanced_statistical_analysis.py"; then
            print_status "Advanced statistical analysis completed"
            
            # Check for additional statistical outputs
            if [ -d "advanced_analysis_results" ]; then
                echo ""
                print_info "Statistical analysis outputs:"
                find advanced_analysis_results -name "*.png" -o -name "*.html" -o -name "*.json" | head -10 | while read file; do
                    echo "  📊 $file"
                done
                
                stats_file_count=$(find advanced_analysis_results -type f | wc -l)
                print_status "Statistical output files: $stats_file_count"
            fi
        else
            print_warning "Advanced statistical analysis failed, but continuing..."
        fi
    else
        print_warning "Advanced statistical analysis script not found"
    fi
fi

# Summary and next steps
echo ""
echo "🎉 Analysis Complete!"
echo "===================="

# Generate clickable file summary
absolute_path=$(pwd)

print_info "📄 Generated Analysis Files (ctrl+click to open):"
echo ""

# Executive and Technical Reports
if [ -f "multi_repo_analysis/reports/executive_summary.md" ]; then
    echo "  📊 Executive Summary:"
    echo "     file://$absolute_path/multi_repo_analysis/reports/executive_summary.md"
fi

if [ -f "multi_repo_analysis/reports/technical_report.md" ]; then
    echo "  🔬 Technical Report:"
    echo "     file://$absolute_path/multi_repo_analysis/reports/technical_report.md"
fi

if [ -f "multi_repo_analysis/reports/language_analysis.md" ]; then
    echo "  🔤 Language Analysis:"
    echo "     file://$absolute_path/multi_repo_analysis/reports/language_analysis.md"
fi

if [ -f "multi_repo_analysis/reports/repository_rankings.md" ]; then
    echo "  🏆 Repository Rankings:"
    echo "     file://$absolute_path/multi_repo_analysis/reports/repository_rankings.md"
fi

echo ""
print_info "📈 Visualizations (ctrl+click to open):"

# Visualizations
if [ -d "multi_repo_analysis/visualizations" ]; then
    for viz_file in multi_repo_analysis/visualizations/*.png; do
        if [ -f "$viz_file" ]; then
            viz_name=$(basename "$viz_file" .png)
            echo "  🖼️  $(echo $viz_name | sed 's/_/ /g' | sed 's/\b\w/\U&/g'):"
            echo "     file://$absolute_path/$viz_file"
        fi
    done
fi

echo ""
print_info "📊 Data Files (ctrl+click to open):"

# CSV exports
if [ -d "multi_repo_analysis/reports/csv_exports" ]; then
    for csv_file in multi_repo_analysis/reports/csv_exports/*.csv; do
        if [ -f "$csv_file" ]; then
            csv_name=$(basename "$csv_file" .csv)
            echo "  📋 $(echo $csv_name | sed 's/_/ /g' | sed 's/\b\w/\U&/g'):"
            echo "     file://$absolute_path/$csv_file"
        fi
    done
fi

# JSON analysis files
if [ -f "multi_repo_analysis/cross_repo_analysis.json" ]; then
    echo "  🔗 Cross-Repository Analysis:"
    echo "     file://$absolute_path/multi_repo_analysis/cross_repo_analysis.json"
fi

if [ -f "multi_repo_analysis/cross_repo_patterns.json" ]; then
    echo "  🎯 Cross-Repository Patterns:"
    echo "     file://$absolute_path/multi_repo_analysis/cross_repo_patterns.json"
fi

# Advanced statistical analysis outputs
if [ "$RUN_ADVANCED_STATS" = true ] && [ -d "advanced_analysis_results" ]; then
    echo ""
    print_info "📊 Advanced Statistical Analysis (ctrl+click to open):"
    
    for stats_file in advanced_analysis_results/*.png advanced_analysis_results/*.html; do
        if [ -f "$stats_file" ]; then
            stats_name=$(basename "$stats_file")
            extension=$(echo "$stats_name" | sed 's/.*\.//')
            name_without_ext=$(echo "$stats_name" | sed 's/\.[^.]*$//')
            echo "  📈 $(echo $name_without_ext | sed 's/_/ /g' | sed 's/\b\w/\U&/g'):"
            echo "     file://$absolute_path/$stats_file"
        fi
    done
fi

echo ""
print_info "📁 Directory Navigation (ctrl+click to open):"
echo "  📂 All Analysis Results:"
echo "     file://$absolute_path/multi_repo_analysis/"
echo "  📊 Visualizations Directory:"
echo "     file://$absolute_path/multi_repo_analysis/visualizations/"
echo "  📄 Reports Directory:"
echo "     file://$absolute_path/multi_repo_analysis/reports/"
echo "  🔢 CSV Data Exports:"
echo "     file://$absolute_path/multi_repo_analysis/reports/csv_exports/"
echo "  📋 Individual Analyses:"
echo "     file://$absolute_path/multi_repo_analysis/individual_analyses/"

# Count total outputs
total_files=0
if [ -d "multi_repo_analysis" ]; then
    primary_files=$(find multi_repo_analysis -type f | wc -l)
    total_files=$((total_files + primary_files))
fi

if [ -d "advanced_analysis_results" ] && [ "$RUN_ADVANCED_STATS" = true ]; then
    stats_files=$(find advanced_analysis_results -type f | wc -l)
    total_files=$((total_files + stats_files))
fi

echo ""
print_info "Summary:"
echo "  📁 Total output files generated: $total_files"
echo "  🔍 Primary analysis: ✅"
if [ "$RUN_ADVANCED_STATS" = true ]; then
    echo "  📊 Advanced statistics: ✅"
else
    echo "  📊 Advanced statistics: ⏭️  (use --advanced-stats to enable)"
fi

echo ""
print_info "Quick Commands:"
echo "  • Run with advanced stats: ./run.sh --advanced-stats"
echo "  • Force reanalysis: ./run.sh --force-reanalysis"
echo "  • Both options: ./run.sh --advanced-stats --force-reanalysis"

echo ""
print_status "OpenTIR analysis framework completed successfully!"

# Optional: Open results in browser (commented out by default)
# if command -v open >/dev/null 2>&1; then
#     print_info "Opening results in browser..."
#     open multi_repo_analysis/reports/executive_summary.md
# fi 