"""
Main entry point for the Gender Bias in LLMs study
"""
import sys
import argparse
from pathlib import Path
from colorama import Fore, Style, init

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils import validate_environment, setup_logging
from src.data_processing import CorpusProcessor, create_corpus_template
from src.llm_interface import LLMManager
from src.prompting import PromptManager
from src.experiment_runner import ExperimentRunner
from src.analysis import ComprehensiveAnalyzer
from src.config import RESULTS_DIR

init(autoreset=True)

def setup_project():
    """Setup the project environment and create templates"""
    print(f"{Fore.CYAN}Setting up Gender Bias Study project...{Style.RESET_ALL}")
    
    # Validate environment
    if not validate_environment():
        print(f"{Fore.YELLOW}Please fix the environment issues and try again{Style.RESET_ALL}")
        return False
    
    # Create corpus template
    create_corpus_template()
    
    # Create sample corpus for testing
    processor = CorpusProcessor()
    processor.create_sample_corpus_file()
    
    print(f"{Fore.GREEN}✓ Project setup complete!{Style.RESET_ALL}")
    print(f"\nNext steps:")
    print(f"1. Copy .env.example to .env and add your API keys")
    print(f"2. Fill in your paragraphs in data/corpus/corpus_template.csv")
    print(f"3. Rename it to paragraphs.csv")
    print(f"4. Run: python main.py run-experiment")
    
    return True

def test_components():
    """Test all components of the system"""
    print(f"{Fore.CYAN}Testing system components...{Style.RESET_ALL}")
    
    # Test LLM interfaces
    try:
        llm_manager = LLMManager()
        test_results = llm_manager.test_all_interfaces()
        
        print(f"\n{Fore.CYAN}LLM Interface Test Results:{Style.RESET_ALL}")
        for model, success in test_results.items():
            status = "✓ Working" if success else "✗ Failed"
            color = Fore.GREEN if success else Fore.RED
            print(f"{color}{model}: {status}{Style.RESET_ALL}")
    
    except Exception as e:
        print(f"{Fore.RED}LLM interface test failed: {e}{Style.RESET_ALL}")
        return False
    
    # Test prompting strategies
    print(f"\n{Fore.CYAN}Testing prompting strategies...{Style.RESET_ALL}")
    prompt_manager = PromptManager()
    prompt_manager.print_all_strategies()
    
    # Test corpus processing
    print(f"\n{Fore.CYAN}Testing corpus processing...{Style.RESET_ALL}")
    processor = CorpusProcessor()
    if processor.corpus_file.exists():
        if processor.load_corpus():
            results = processor.validate_paragraphs()
            print(f"{Fore.GREEN}✓ Corpus processing test passed{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}⚠ Corpus file exists but could not be loaded{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}⚠ No corpus file found (use setup to create template){Style.RESET_ALL}")
    
    print(f"{Fore.GREEN}✓ Component testing complete{Style.RESET_ALL}")
    return True

def run_experiment(resume: bool = True):
    """Run the main experiment"""
    print(f"{Fore.CYAN}Starting Gender Bias Experiment...{Style.RESET_ALL}")
    
    runner = ExperimentRunner()
    
    # Print experiment summary
    runner.print_experiment_summary()
    
    # Ask for confirmation
    print(f"\n{Fore.YELLOW}This will run multiple experiments and may take significant time.{Style.RESET_ALL}")
    print(f"Estimated time: 30-60 minutes depending on API response times")
    
    response = input("Continue? (y/N): ").lower().strip()
    if response != 'y':
        print("Experiment cancelled.")
        return False
    
    # Run experiments
    success = runner.run_all_experiments(resume=resume)
    
    if success:
        print(f"\n{Fore.GREEN}✓ Experiment completed successfully!{Style.RESET_ALL}")
        print(f"Results saved in: {runner.results_file}")
        
        # Offer to run analysis
        response = input("Run analysis now? (Y/n): ").lower().strip()
        if response != 'n':
            run_analysis(runner.results_file)
    else:
        print(f"\n{Fore.YELLOW}Experiment stopped or failed.{Style.RESET_ALL}")
        print("You can resume later by running the same command.")
    
    return success

def run_analysis(results_file: Path = None):
    """Run statistical analysis and generate visualizations"""
    print(f"{Fore.CYAN}Running analysis and generating visualizations...{Style.RESET_ALL}")
    
    if results_file is None:
        # Find the most recent results file
        results_files = list(RESULTS_DIR.glob("experiment_results_*.json"))
        
        if not results_files:
            print(f"{Fore.RED}No experiment results found in {RESULTS_DIR}{Style.RESET_ALL}")
            print("Run the experiment first using: python main.py run-experiment")
            return False
        
        results_file = max(results_files, key=lambda f: f.stat().st_mtime)
        print(f"Using most recent results: {results_file.name}")
    
    try:
        analyzer = ComprehensiveAnalyzer(results_file)
        analysis_results = analyzer.run_complete_analysis()
        analyzer.print_analysis_summary(analysis_results)
        
        print(f"\n{Fore.GREEN}✓ Analysis complete!{Style.RESET_ALL}")
        return True
        
    except Exception as e:
        print(f"{Fore.RED}Analysis failed: {e}{Style.RESET_ALL}")
        return False

def list_results():
    """List all available experiment results"""
    results_files = list(RESULTS_DIR.glob("experiment_results_*.json"))
    
    if not results_files:
        print(f"{Fore.YELLOW}No experiment results found in {RESULTS_DIR}{Style.RESET_ALL}")
        return
    
    print(f"{Fore.CYAN}Available experiment results:{Style.RESET_ALL}")
    for i, file in enumerate(sorted(results_files, key=lambda f: f.stat().st_mtime, reverse=True)):
        print(f"{i+1}. {file.name} (modified: {file.stat().st_mtime})")

def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(
        description="Gender Bias in LLMs Study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py setup                 # Setup project and create templates
  python main.py test                  # Test all components
  python main.py run-experiment        # Run the full experiment
  python main.py analyze               # Analyze most recent results
  python main.py list-results          # List all available results
        """
    )
    
    parser.add_argument(
        "command",
        choices=["setup", "test", "run-experiment", "analyze", "list-results"],
        help="Command to run"
    )
    
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume previous experiment (start fresh)"
    )
    
    parser.add_argument(
        "--results-file",
        type=Path,
        help="Specific results file to analyze"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        setup_logging()
    
    # Execute commands
    if args.command == "setup":
        setup_project()
    
    elif args.command == "test":
        test_components()
    
    elif args.command == "run-experiment":
        run_experiment(resume=not args.no_resume)
    
    elif args.command == "analyze":
        run_analysis(args.results_file)
    
    elif args.command == "list-results":
        list_results()
    
    else:
        print(f"{Fore.RED}Unknown command: {args.command}{Style.RESET_ALL}")
        parser.print_help()

if __name__ == "__main__":
    main()
