#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for Gender Bias study - Small scale validation
Runs 3 paragraphs x 4 strategies x 2 models x 1 repetition = 24 experiments
Perfect for testing before running the full study
"""
import sys
import time
from pathlib import Path
from datetime import datetime
from colorama import Fore, Style, init

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils import validate_environment, setup_logging, save_json, ExperimentState
from src.data_processing import CorpusProcessor
from src.llm_interface import LLMManager
from src.prompting import PromptManager
from src.evaluation import ComprehensiveEvaluator
from src.analysis import ComprehensiveAnalyzer
from src.config import EXPERIMENT_CONFIG, RESULTS_DIR

init(autoreset=True)

class TestExperimentRunner:
    """Test runner for small-scale validation"""
    
    def __init__(self, num_paragraphs: int = 3):
        self.num_paragraphs = num_paragraphs
        self.corpus_processor = CorpusProcessor()
        self.llm_manager = None
        self.prompt_manager = PromptManager()
        self.evaluator = ComprehensiveEvaluator()
        
        # Test experiment tracking
        self.experiment_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results_file = RESULTS_DIR / f"test_results_{self.experiment_id}.json"
        self.detailed_results = []
        
        print(f"{Fore.CYAN}Test Experiment Runner Initialized{Style.RESET_ALL}")
        print(f"Test ID: {self.experiment_id}")
        print(f"Testing with {num_paragraphs} paragraphs")
    
    def setup_test(self) -> bool:
        """Setup test environment"""
        print(f"{Fore.CYAN}Setting up test environment...{Style.RESET_ALL}")
        
        # Validate environment
        if not validate_environment():
            return False
        
        # Load corpus
        if not self.corpus_processor.load_corpus():
            return False
        
        # Take only first N paragraphs for testing
        all_paragraphs = self.corpus_processor.get_all_paragraphs()
        if len(all_paragraphs) < self.num_paragraphs:
            print(f"{Fore.RED}Not enough paragraphs in corpus. Need at least {self.num_paragraphs}{Style.RESET_ALL}")
            return False
        
        self.test_paragraphs = all_paragraphs[:self.num_paragraphs]
        print(f"{Fore.GREEN}✓ Selected {len(self.test_paragraphs)} paragraphs for testing{Style.RESET_ALL}")
        
        # Display selected paragraphs
        for i, p in enumerate(self.test_paragraphs):
            print(f"  {i+1}. {p['id']}: {p['text'][:100]}...")
        
        # Initialize LLM interfaces
        try:
            self.llm_manager = LLMManager()
            if not self.llm_manager.get_available_models():
                print(f"{Fore.RED}Error: No LLM interfaces available{Style.RESET_ALL}")
                return False
        except Exception as e:
            print(f"{Fore.RED}Error initializing LLM interfaces: {e}{Style.RESET_ALL}")
            return False
        
        # Calculate total test experiments
        strategies = self.prompt_manager.get_strategy_names()
        models = self.llm_manager.get_available_models()
        repetitions = 1  # Only 1 repetition for testing
        
        total_experiments = len(self.test_paragraphs) * len(strategies) * len(models) * repetitions
        
        print(f"{Fore.GREEN}✓ Test setup complete{Style.RESET_ALL}")
        print(f"Total test experiments: {total_experiments}")
        print(f"Strategies: {', '.join(strategies)}")
        print(f"Models: {', '.join(models)}")
        
        return True
    
    def run_single_test_experiment(self, paragraph_data: dict, strategy_name: str, 
                                 model_name: str, repetition: int = 1) -> dict:
        """Run a single test experiment"""
        paragraph_id = paragraph_data['id']
        paragraph_text = paragraph_data['text']
        
        experiment_id = f"test_{paragraph_id}_{strategy_name}_{model_name}_rep{repetition}"
        
        print(f"   Running: {experiment_id}")
        
        try:
            # Create prompt
            user_prompt, system_message = self.prompt_manager.create_prompt_for_strategy(
                strategy_name, paragraph_text
            )
            
            # Generate text with LLM
            start_time = time.time()
            generation_result = self.llm_manager.generate_with_model(
                model_name, user_prompt, system_message
            )
            generation_time = time.time() - start_time
            
            if not generation_result["success"]:
                error_msg = f"LLM generation failed: {generation_result['error']}"
                print(f"    {Fore.RED}X {error_msg}{Style.RESET_ALL}")
                return {
                    "experiment_id": experiment_id,
                    "success": False,
                    "error": error_msg
                }
            
            generated_text = generation_result["generated_text"]
            
            # Quick preview
            print(f"     Generated: {generated_text[:100]}...")
            
            # Evaluate the result
            evaluation_result = self.evaluator.evaluate_text_pair(
                paragraph_text, 
                generated_text,
                {
                    "paragraph_id": paragraph_id,
                    "strategy": strategy_name,
                    "model": model_name,
                    "repetition": repetition,
                    "experiment_id": experiment_id
                }
            )
            
            # Show quick evaluation results
            summary = evaluation_result["summary_scores"]
            print(f"     Bias reduction: {summary['bias_reduction_percentage']:.1f}%")
            print(f"     Gender neutral: {summary['is_gender_neutral']}")
            print(f"     Fluency: {summary['fluency_score']:.3f}")
            print(f"     BLEU-4: {summary['bleu_4_score']:.3f}")
            
            # Compile complete experiment result
            experiment_result = {
                "experiment_id": experiment_id,
                "timestamp": datetime.now().isoformat(),
                "paragraph_id": paragraph_id,
                "strategy": strategy_name,
                "model": model_name,
                "repetition": repetition,
                "generation_time": generation_time,
                "generation_result": generation_result,
                "evaluation": evaluation_result,
                "success": True
            }
            
            print(f"    {Fore.GREEN}OK Success{Style.RESET_ALL}")
            return experiment_result
            
        except Exception as e:
            error_msg = f"Experiment failed: {str(e)}"
            print(f"    {Fore.RED}X {error_msg}{Style.RESET_ALL}")
            return {
                "experiment_id": experiment_id,
                "success": False,
                "error": error_msg
            }
    
    def run_test_experiments(self) -> bool:
        """Run all test experiments"""
        if not self.setup_test():
            return False
        
        strategies = self.prompt_manager.get_strategy_names()
        models = self.llm_manager.get_available_models()
        
        print(f"\n{Fore.CYAN} Starting test experiments...{Style.RESET_ALL}")
        start_time = time.time()
        
        experiment_count = 0
        successful_experiments = 0
        
        try:
            for paragraph_data in self.test_paragraphs:
                print(f"\n Processing paragraph: {paragraph_data['id']}")
                
                for strategy_name in strategies:
                    print(f"   Strategy: {strategy_name}")
                    
                    for model_name in models:
                        print(f"     Model: {model_name}")
                        
                        experiment_count += 1
                        
                        # Run single experiment
                        result = self.run_single_test_experiment(
                            paragraph_data, strategy_name, model_name, 1
                        )
                        
                        if result and result.get("success", False):
                            self.detailed_results.append(result)
                            successful_experiments += 1
        
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Test interrupted by user{Style.RESET_ALL}")
            return False
        
        except Exception as e:
            print(f"\n{Fore.RED}Test failed with error: {e}{Style.RESET_ALL}")
            return False
        
        total_time = time.time() - start_time
        
        print(f"\n{Fore.GREEN}OK Test experiments completed!{Style.RESET_ALL}")
        print(f"Total time: {total_time:.1f} seconds")
        print(f"Successful experiments: {successful_experiments}/{experiment_count}")
        
        # Save test results
        self.save_test_results()
        
        return successful_experiments > 0
    
    def save_test_results(self):
        """Save test results"""
        try:
            test_results = {
                "experiment_id": self.experiment_id,
                "timestamp": datetime.now().isoformat(),
                "status": "completed",
                "test_configuration": {
                    "num_paragraphs": len(self.test_paragraphs),
                    "strategies": self.prompt_manager.get_strategy_names(),
                    "models": self.llm_manager.get_available_models(),
                    "repetitions": 1
                },
                "total_experiments": len(self.detailed_results),
                "detailed_results": self.detailed_results
            }
            
            save_json(test_results, self.results_file)
            print(f"{Fore.GREEN}✓ Test results saved to: {self.results_file}{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"{Fore.RED}Error saving test results: {e}{Style.RESET_ALL}")
    
    def run_test_analysis(self) -> bool:
        """Run analysis on test results"""
        if not self.results_file.exists():
            print(f"{Fore.RED}No test results found to analyze{Style.RESET_ALL}")
            return False
        
        try:
            print(f"\n{Fore.CYAN} Running test analysis...{Style.RESET_ALL}")
            
            # Use the comprehensive analyzer
            analyzer = ComprehensiveAnalyzer(self.results_file)
            analysis_results = analyzer.run_complete_analysis()
            
            # Print summary
            print(f"\n{Fore.CYAN} TEST ANALYSIS SUMMARY{Style.RESET_ALL}")
            print("=" * 50)
            
            # Show strategy performance
            strategy_stats = analysis_results["descriptive_statistics"]["by_strategy"]
            print(f"\n Strategy Performance:")
            for strategy, stats in strategy_stats.items():
                bias_mean = stats["bias_reduction_percentage"]["mean"]
                fluency_mean = stats["fluency_score"]["mean"]
                print(f"  {strategy}: Bias Reduction {bias_mean:.1f}%, Fluency {fluency_mean:.3f}")
            
            # Show ANOVA results
            print(f"\n Statistical Tests:")
            for metric, results in analysis_results["statistical_analysis"].items():
                if metric.startswith("anova_"):
                    significance = "OK Significant" if results["significant"] else "X Not significant"
                    print(f"  {metric.replace('anova_', '')}: {significance} (p={results['p_value']:.4f})")
            
            # Export data for easy viewing
            print(f"\n{Fore.CYAN}Exporting results for viewing...{Style.RESET_ALL}")
            try:
                from src.data_export import export_experiment_data
                exported_files = export_experiment_data(self.results_file)
                
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
            
            print(f"\nOK Test analysis complete!")
            return True
            
        except Exception as e:
            print(f"{Fore.RED}Test analysis failed: {e}{Style.RESET_ALL}")
            return False

def main():
    """Main function to run test experiments"""
    print(f"{Fore.CYAN} Gender Bias Study - Test Runner{Style.RESET_ALL}")
    print("=" * 60)
    print("This will run a small-scale test with 3 paragraphs to validate the system")
    print("before running the full 300-experiment study.")
    print("")
    
    # Ask for confirmation
    response = input(f"{Fore.YELLOW}Run test experiments? (Y/n): {Style.RESET_ALL}").lower().strip()
    if response == 'n':
        print("Test cancelled.")
        return
    
    # Initialize and run test
    test_runner = TestExperimentRunner(num_paragraphs=3)
    
    # Run test experiments
    test_success = test_runner.run_test_experiments()
    
    if test_success:
        print(f"\n{Fore.GREEN}🎉 Test experiments successful!{Style.RESET_ALL}")
        
        # Ask if user wants to run analysis
        response = input(f"{Fore.YELLOW}Run test analysis? (Y/n): {Style.RESET_ALL}").lower().strip()
        if response != 'n':
            analysis_success = test_runner.run_test_analysis()
            
            if analysis_success:
                print(f"\n{Fore.GREEN} Test complete! System is working correctly.{Style.RESET_ALL}")
                print(f"{Fore.CYAN}You can now run the full experiment with:{Style.RESET_ALL}")
                print(f"python main.py run-experiment")
            else:
                print(f"\n{Fore.YELLOW}Test experiments worked, but analysis had issues.{Style.RESET_ALL}")
    else:
        print(f"\n{Fore.RED}X Test experiments failed. Please check the setup.{Style.RESET_ALL}")
        print(f"Fix any issues and try again.")

if __name__ == "__main__":
    main()
