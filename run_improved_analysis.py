#!/usr/bin/env python3
"""
Run Improved Analysis for Gender Bias Study
Focus on Bias Reduction, BLEU-4, and ANOVA Analysis
"""

import sys
from pathlib import Path
import argparse
from colorama import init, Fore, Style

# Initialize colorama
init()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from improved_analysis import run_improved_analysis
from config import RESULTS_DIR

def main():
    parser = argparse.ArgumentParser(description="Run improved analysis for Gender Bias study")
    parser.add_argument("--results-file", "-f", type=str, help="Specific results file to analyze")
    parser.add_argument("--latest", "-l", action="store_true", help="Use the latest results file")
    
    args = parser.parse_args()
    
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}IMPROVED GENDER BIAS ANALYSIS{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Focus: Bias Reduction, BLEU-4, ANOVA{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    
    # Find results file
    if args.results_file:
        results_file = Path(args.results_file)
        if not results_file.exists():
            print(f"{Fore.RED}Error: Results file not found: {results_file}{Style.RESET_ALL}")
            return 1
    else:
        # Find latest results file
        results_files = list(RESULTS_DIR.glob("experiment_results_*.json"))
        if not results_files:
            print(f"{Fore.RED}Error: No experiment results found in {RESULTS_DIR}{Style.RESET_ALL}")
            return 1
        
        results_file = max(results_files, key=lambda x: x.stat().st_mtime)
        print(f"{Fore.YELLOW}Using latest results file: {results_file.name}{Style.RESET_ALL}")
    
    # Run improved analysis
    try:
        results = run_improved_analysis(results_file)
        
        if "error" in results:
            print(f"{Fore.RED}Analysis failed: {results['error']}{Style.RESET_ALL}")
            return 1
        
        # Print key findings
        print(f"\\n{Fore.GREEN}{'=' * 60}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}KEY FINDINGS{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{'=' * 60}{Style.RESET_ALL}")
        
        findings = results["key_findings"]
        
        print(f"🏆 {Fore.YELLOW}Best Strategy:{Style.RESET_ALL} {findings['best_strategy']}")
        print(f"📊 {Fore.YELLOW}Best Bias Reduction:{Style.RESET_ALL} {findings['best_bias_reduction']:.1f}%")
        print(f"📈 {Fore.YELLOW}Statistical Significance:{Style.RESET_ALL} {'Yes' if findings['statistical_significance'] else 'No'}")
        
        print(f"\\n{Fore.CYAN}Effect Sizes:{Style.RESET_ALL}")
        for metric, effect in findings['effect_sizes'].items():
            print(f"  • {metric}: {effect}")
        
        print(f"\\n{Fore.CYAN}Visualizations Created:{Style.RESET_ALL}")
        for viz_file in results["visualization_files"]:
            print(f"  • {Path(viz_file).name}")
        
        print(f"\\n{Fore.GREEN}✓ Analysis complete! Check the improved_visualizations folder for results.{Style.RESET_ALL}")
        
    except Exception as e:
        print(f"{Fore.RED}Error running analysis: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
