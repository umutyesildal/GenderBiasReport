"""
Main entry point for the Gender Bias in LLMs study
"""
import sys
import json
import argparse
from pathlib import Path
from colorama import Fore, Style, init
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils import validate_environment, setup_logging, save_json
from src.data_processing import CorpusProcessor, create_corpus_template
from src.llm_interface import LLMManager
from src.prompting import PromptManager
from src.experiment_runner import ExperimentRunner
from src.analysis import ComprehensiveAnalyzer
from src.config import RESULTS_DIR, EXPERIMENT_CONFIG

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
        
        # Automatically run analysis and export like test_experiment.py does
        print(f"\n{Fore.CYAN}Running post-experiment analysis and export...{Style.RESET_ALL}")
        
        try:
            # Run analysis
            analysis_success = run_analysis(runner.results_file)
            
            if analysis_success:
                print(f"{Fore.GREEN}✓ Analysis completed successfully{Style.RESET_ALL}")
            else:
                print(f"{Fore.YELLOW}Warning: Analysis had issues but experiment completed{Style.RESET_ALL}")
                
        except Exception as e:
            print(f"{Fore.YELLOW}Warning: Post-experiment analysis failed: {e}{Style.RESET_ALL}")
            print(f"You can run analysis manually with: python main.py analyze")
        
    else:
        print(f"\n{Fore.YELLOW}Experiment stopped or failed.{Style.RESET_ALL}")
        print("You can resume later by running the same command.")
        
        # Check if there are temp results that can be analyzed
        temp_files = list(RESULTS_DIR.glob("temp_results_*.json"))
        if temp_files:
            latest_temp = max(temp_files, key=lambda f: f.stat().st_mtime)
            print(f"\n{Fore.CYAN}Found incomplete results: {latest_temp.name}{Style.RESET_ALL}")
            response = input("Analyze incomplete results? (y/N): ").lower().strip()
            if response == 'y':
                try:
                    run_analysis(latest_temp)
                except Exception as e:
                    print(f"{Fore.YELLOW}Analysis of incomplete results failed: {e}{Style.RESET_ALL}")
    
    return success

def run_analysis(results_file: Path = None):
    """Run statistical analysis and generate visualizations"""
    print(f"{Fore.CYAN}Running analysis and generating visualizations...{Style.RESET_ALL}")
    
    if results_file is None:
        # Find the most recent results file - check both final and temp results
        results_files = list(RESULTS_DIR.glob("experiment_results_*.json"))
        temp_files = list(RESULTS_DIR.glob("temp_results_*.json"))
        
        # Prefer final results, but fall back to temp results if no final results exist
        if results_files:
            results_file = max(results_files, key=lambda f: f.stat().st_mtime)
            print(f"Using most recent final results: {results_file.name}")
        elif temp_files:
            results_file = max(temp_files, key=lambda f: f.stat().st_mtime)
            print(f"{Fore.YELLOW}No final results found, using temp results: {results_file.name}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Note: This may be from an incomplete experiment{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}No experiment results found in {RESULTS_DIR}{Style.RESET_ALL}")
            print("Run the experiment first using: python main.py run-experiment")
            return False
    
    try:
        analyzer = ComprehensiveAnalyzer(results_file)
        analysis_results = analyzer.run_complete_analysis()
        
        print(f"\n{Fore.GREEN}✓ Statistical analysis complete!{Style.RESET_ALL}")
        
        # Export data for easy viewing
        print(f"\n{Fore.CYAN}Exporting results for viewing...{Style.RESET_ALL}")
        try:
            from src.data_export import export_experiment_data
            exported_files = export_experiment_data(results_file)
            
            if exported_files:
                print(f"{Fore.GREEN}✓ Results exported:{Style.RESET_ALL}")
                for export_type, file_path in exported_files.items():
                    print(f"  {export_type}: {file_path}")
                
                # Open HTML viewer in browser if available
                if "html_viewer" in exported_files:
                    try:
                        import webbrowser
                        webbrowser.open(f"file://{exported_files['html_viewer']}")
                        print(f"{Fore.GREEN}✓ HTML viewer opened in browser{Style.RESET_ALL}")
                    except Exception as e:
                        print(f"{Fore.YELLOW}Note: Could not auto-open browser: {e}{Style.RESET_ALL}")
                        print(f"Manual open: file://{exported_files['html_viewer']}")
            
        except Exception as e:
            print(f"{Fore.YELLOW}Warning: Data export failed: {e}{Style.RESET_ALL}")
        
        print(f"\n{Fore.GREEN}✓ Analysis complete!{Style.RESET_ALL}")
        return True
        
    except Exception as e:
        print(f"{Fore.RED}Analysis failed: {e}{Style.RESET_ALL}")
        return False

def export_results(results_file: Path = None):
    """Export results to JSON and HTML viewer"""
    print(f"{Fore.CYAN}Exporting experiment results...{Style.RESET_ALL}")
    
    if results_file is None:
        # Find the most recent results file - check both final and temp results
        results_files = list(RESULTS_DIR.glob("experiment_results_*.json"))
        temp_files = list(RESULTS_DIR.glob("temp_results_*.json"))
        test_files = list(RESULTS_DIR.glob("test_results_*.json"))
        
        all_files = results_files + temp_files + test_files
        
        if not all_files:
            print(f"{Fore.RED}No experiment results found in {RESULTS_DIR}{Style.RESET_ALL}")
            print("Run the experiment first using: python main.py run-experiment")
            return False
        
        results_file = max(all_files, key=lambda f: f.stat().st_mtime)
        print(f"Using most recent results: {results_file.name}")
    
    try:
        from src.data_export import export_experiment_data
        exported_files = export_experiment_data(results_file)
        
        if exported_files:
            print(f"{Fore.GREEN}✓ Results exported:{Style.RESET_ALL}")
            for export_type, file_path in exported_files.items():
                print(f"  {export_type}: {file_path}")
            
            # Open HTML viewer in browser if available
            if "html_viewer" in exported_files:
                response = input(f"{Fore.YELLOW}Open HTML viewer in browser? (Y/n): {Style.RESET_ALL}").lower().strip()
                if response != 'n':
                    try:
                        import webbrowser
                        webbrowser.open(f"file://{exported_files['html_viewer']}")
                        print(f"{Fore.GREEN}✓ HTML viewer opened in browser{Style.RESET_ALL}")
                    except Exception as e:
                        print(f"{Fore.YELLOW}Could not auto-open browser: {e}{Style.RESET_ALL}")
                        print(f"Manual open: file://{exported_files['html_viewer']}")
            
            return True
        else:
            print(f"{Fore.RED}Export failed{Style.RESET_ALL}")
            return False
            
    except Exception as e:
        print(f"{Fore.RED}Export failed: {e}{Style.RESET_ALL}")
        return False

def list_results():
    """List all available result files"""
    print(f"{Fore.CYAN}Available Results Files:{Style.RESET_ALL}")
    print("=" * 50)
    
    # Get all result files
    experiment_files = list(RESULTS_DIR.glob("experiment_results_*.json"))
    temp_files = list(RESULTS_DIR.glob("temp_results_*.json"))
    test_files = list(RESULTS_DIR.glob("test_results_*.json"))
    state_files = list(RESULTS_DIR.glob("experiment_state_*.json"))
    
    if not any([experiment_files, temp_files, test_files]):
        print(f"{Fore.YELLOW}No result files found in {RESULTS_DIR}{Style.RESET_ALL}")
        return
    
    # Sort by modification time
    all_files = []
    
    for file in experiment_files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                status = data.get('status', 'unknown')
                total_experiments = data.get('total_experiments', 0)
                timestamp = data.get('timestamp', 'unknown')
            all_files.append({
                'file': file,
                'type': 'Final Results',
                'status': status,
                'experiments': total_experiments,
                'timestamp': timestamp,
                'mtime': file.stat().st_mtime
            })
        except Exception as e:
            all_files.append({
                'file': file,
                'type': 'Final Results',
                'status': f'Error: {e}',
                'experiments': '?',
                'timestamp': '?',
                'mtime': file.stat().st_mtime
            })
    
    for file in temp_files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                status = data.get('status', 'unknown')
                total_experiments = len(data.get('detailed_results', []))
                timestamp = data.get('timestamp', 'unknown')
            all_files.append({
                'file': file,
                'type': 'Temp Results',
                'status': status,
                'experiments': total_experiments,
                'timestamp': timestamp,
                'mtime': file.stat().st_mtime
            })
        except Exception as e:
            all_files.append({
                'file': file,
                'type': 'Temp Results',
                'status': f'Error: {e}',
                'experiments': '?',
                'timestamp': '?',
                'mtime': file.stat().st_mtime
            })
    
    for file in test_files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                status = data.get('status', 'unknown')
                total_experiments = data.get('total_experiments', 0)
                timestamp = data.get('timestamp', 'unknown')
            all_files.append({
                'file': file,
                'type': 'Test Results',
                'status': status,
                'experiments': total_experiments,
                'timestamp': timestamp,
                'mtime': file.stat().st_mtime
            })
        except Exception as e:
            all_files.append({
                'file': file,
                'type': 'Test Results',
                'status': f'Error: {e}',
                'experiments': '?',
                'timestamp': '?',
                'mtime': file.stat().st_mtime
            })
    
    # Sort by modification time (newest first)
    all_files.sort(key=lambda x: x['mtime'], reverse=True)
    
    for file_info in all_files:
        color = Fore.GREEN if file_info['status'] == 'completed' else Fore.YELLOW if file_info['status'] == 'in_progress' else Fore.RED
        print(f"{color}{file_info['type']}: {file_info['file'].name}{Style.RESET_ALL}")
        print(f"  Status: {file_info['status']}")
        print(f"  Experiments: {file_info['experiments']}")
        print(f"  Timestamp: {file_info['timestamp']}")
        print()
    
    # Show state files
    if state_files:
        print(f"{Fore.CYAN}Experiment State Files:{Style.RESET_ALL}")
        for file in sorted(state_files, key=lambda f: f.stat().st_mtime, reverse=True):
            print(f"  {file.name}")
    
    return True

def convert_temp_results():
    """Convert temp results to final results format"""
    print(f"{Fore.CYAN}Converting temp results to final format...{Style.RESET_ALL}")
    
    # Find temp result files
    temp_files = list(RESULTS_DIR.glob("temp_results_*.json"))
    
    if not temp_files:
        print(f"{Fore.YELLOW}No temp result files found{Style.RESET_ALL}")
        return False
    
    # Sort by modification time (newest first)
    temp_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    
    print(f"Found {len(temp_files)} temp result files:")
    for i, file in enumerate(temp_files):
        try:
            with open(file, 'r') as f:
                data = json.load(f)
            experiments = len(data.get('detailed_results', []))
            timestamp = data.get('timestamp', 'unknown')
            print(f"  {i+1}. {file.name} - {experiments} experiments ({timestamp})")
        except Exception as e:
            print(f"  {i+1}. {file.name} - Error reading file: {e}")
    
    # Ask user which file to convert
    try:
        choice = input(f"\n{Fore.YELLOW}Which file to convert? (1-{len(temp_files)}, or 'all'): {Style.RESET_ALL}").lower().strip()
        
        if choice == 'all':
            files_to_convert = temp_files
        else:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(temp_files):
                files_to_convert = [temp_files[choice_idx]]
            else:
                print(f"{Fore.RED}Invalid choice{Style.RESET_ALL}")
                return False
    except ValueError:
        print(f"{Fore.RED}Invalid input{Style.RESET_ALL}")
        return False
    
    # Convert selected files
    success_count = 0
    for temp_file in files_to_convert:
        try:
            # Load temp results
            with open(temp_file, 'r') as f:
                temp_data = json.load(f)
            
            detailed_results = temp_data.get('detailed_results', [])
            experiment_id = temp_data.get('experiment_id', 'unknown')
            
            print(f"\nConverting {temp_file.name} ({len(detailed_results)} experiments)...")
            
            # Calculate aggregate statistics
            from src.evaluation import ComprehensiveEvaluator
            evaluator = ComprehensiveEvaluator()
            aggregate_stats = None
            
            if detailed_results:
                try:
                    aggregate_stats = evaluator.calculate_aggregate_scores(detailed_results)
                    print(f"✓ Calculated aggregate statistics")
                except Exception as e:
                    print(f"{Fore.YELLOW}Warning: Could not calculate aggregate statistics: {e}{Style.RESET_ALL}")
            
            # Create final results structure
            final_results = {
                "experiment_id": experiment_id,
                "timestamp": datetime.now().isoformat(),
                "status": "completed",
                "configuration": EXPERIMENT_CONFIG,
                "total_experiments": len(detailed_results),
                "aggregate_statistics": aggregate_stats,
                "detailed_results": detailed_results,
                "final_progress": temp_data.get('progress', {})
            }
            
            # Save final results
            final_file = RESULTS_DIR / f"experiment_results_{experiment_id}.json"
            save_json(final_results, final_file)
            
            print(f"{Fore.GREEN}✓ Final results saved to: {final_file.name}{Style.RESET_ALL}")
            
            # Remove temp file
            temp_file.unlink()
            print(f"{Fore.GREEN}✓ Cleaned up temp file{Style.RESET_ALL}")
            
            success_count += 1
            
        except Exception as e:
            print(f"{Fore.RED}Error converting {temp_file.name}: {e}{Style.RESET_ALL}")
    
    print(f"\n{Fore.GREEN}Successfully converted {success_count}/{len(files_to_convert)} files{Style.RESET_ALL}")
    return success_count > 0

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
  python main.py export                # Export results to JSON/HTML viewer
  python main.py list-results          # List all available results
  python main.py convert-temp          # Convert temp results to final format
        """
    )
    
    parser.add_argument(
        "command",
        choices=["setup", "test", "run-experiment", "analyze", "export", "list-results", "convert-temp"],
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
    
    elif args.command == "export":
        export_results(args.results_file)
    
    elif args.command == "list-results":
        list_results()
    
    elif args.command == "convert-temp":
        convert_temp_results()
    
    else:
        print(f"{Fore.RED}Unknown command: {args.command}{Style.RESET_ALL}")
        parser.print_help()

if __name__ == "__main__":
    main()
