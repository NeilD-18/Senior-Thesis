#!/usr/bin/env python3
"""Quick script to plot severity grid results."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cli.analyze_severity_grid import main

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Plot severity grid results')
    parser.add_argument('--summary', type=str, 
                       default='outputs/severity_grids/severity_grid_summary.yaml',
                       help='Path to severity grid summary YAML')
    parser.add_argument('--plot', type=str, 
                       default='outputs/severity_grids/degradation_curves.png',
                       help='Path to save plot')
    parser.add_argument('--no-summary', action='store_true',
                       help='Skip printing summary table')
    
    args = parser.parse_args()
    
    # Temporarily override sys.argv for main()
    old_argv = sys.argv
    sys.argv = ['analyze_severity_grid', '--summary', args.summary, '--plot', args.plot]
    if args.no_summary:
        sys.argv.append('--no-plot')  # This will skip summary
    
    try:
        main()
    finally:
        sys.argv = old_argv
