#!/bin/bash

echo "=========================================="
echo "DiRoCA TBS Comprehensive Analysis Runner"
echo "=========================================="

# Check if Python script exists
if [ ! -f "comprehensive_analysis.py" ]; then
    echo "Error: comprehensive_analysis.py not found!"
    exit 1
fi

# Run the comprehensive analysis
echo "Starting comprehensive analysis..."
python comprehensive_analysis.py

echo ""
echo "=========================================="
echo "Analysis Complete!"
echo "=========================================="
echo ""
echo "Check the 'analysis_outputs' directory for all results."
echo "Use 'analysis_outputs/analysis_dashboard.md' to navigate the outputs."
echo ""
echo "Generated outputs include:"
echo "  - Summary tables (CSV and text formats)"
echo "  - Publication-ready plots (PDF and PNG)"
echo "  - LaTeX-ready tables"
echo "  - Analysis dashboard and summary" 