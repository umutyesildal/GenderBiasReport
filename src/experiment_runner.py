"""
Main experiment runner for the Gender Bias study
"""
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
from tqdm import tqdm

from .config import EXPERIMENT_CONFIG, OUTPUTS_DIR, RESULTS_DIR
from .utils import ExperimentState, save_json, print_progress, format_time_elapsed, validate_environment
from .data_processing import CorpusProcessor
from .llm_interface import LLMManager
from .prompting import PromptManager
from .evaluation import ComprehensiveEvaluator
from colorama import Fore, Style, init

init(autoreset=True)

class ExperimentRunner:
    """Main class to run the gender bias experiments"""
    
    def __init__(self, corpus_file: Optional[Path] = None):
        self.corpus_processor = CorpusProcessor(corpus_file)
        self.llm_manager = None
        self.prompt_manager = PromptManager()
        self.evaluator = ComprehensiveEvaluator()
        
        # Experiment tracking
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.state_file = RESULTS_DIR / f"experiment_state_{self.experiment_id}.json"
        self.state = ExperimentState(self.state_file)
        
        # Results storage
        self.results_file = RESULTS_DIR / f"experiment_results_{self.experiment_id}.json"
        self.detailed_results = []
        
        print(f"{Fore.CYAN}Initialized Experiment Runner{Style.RESET_ALL}")
        print(f"Experiment ID: {self.experiment_id}")
    
    def setup_experiment(self) -> bool:
        """Setup all components for the experiment"""
        print(f"{Fore.CYAN}Setting up experiment...{Style.RESET_ALL}")
        
        # Validate environment
        if not validate_environment():
            return False
        
        # Load corpus
        if not self.corpus_processor.load_corpus():
            return False
        
        # Validate corpus
        validation_results = self.corpus_processor.validate_paragraphs()
        if validation_results["paragraphs_with_gendered_terms"] == 0:
            print(f"{Fore.RED}Error: No paragraphs with gendered terms found in corpus{Style.RESET_ALL}")
            return False
        
        # Initialize LLM interfaces
        try:
            self.llm_manager = LLMManager()
            if not self.llm_manager.get_available_models():
                print(f"{Fore.RED}Error: No LLM interfaces available{Style.RESET_ALL}")
                return False
        except Exception as e:
            print(f"{Fore.RED}Error initializing LLM interfaces: {e}{Style.RESET_ALL}")
            return False
        
        # Calculate total experiments
        paragraphs = self.corpus_processor.get_all_paragraphs()
        strategies = self.prompt_manager.get_strategy_names()
        models = self.llm_manager.get_available_models()
        repetitions = EXPERIMENT_CONFIG["repetitions_per_paragraph"]
        
        total_experiments = len(paragraphs) * len(strategies) * len(models) * repetitions
        self.state.state["total_experiments"] = total_experiments
        self.state.save_state()
        
        print(f"{Fore.GREEN}✓ Experiment setup complete{Style.RESET_ALL}")
        print(f"Total experiments to run: {total_experiments}")
        print(f"Paragraphs: {len(paragraphs)}")
        print(f"Strategies: {len(strategies)} ({', '.join(strategies)})")
        print(f"Models: {len(models)} ({', '.join(models)})")
        print(f"Repetitions: {repetitions}")
        
        return True
    
    def run_single_experiment(self, paragraph_data: Dict, strategy_name: str, 
                            model_name: str, repetition: int) -> Dict[str, Any]:
        """Run a single experiment"""
        paragraph_id = paragraph_data['id']
        paragraph_text = paragraph_data['paragraph']
        
        # Create experiment ID
        experiment_id = f"{paragraph_id}_{strategy_name}_{model_name}_rep{repetition}"
        
        # Check if already completed
        if self.state.is_experiment_completed(experiment_id):
            print(f"{Fore.YELLOW}Skipping completed experiment: {experiment_id}{Style.RESET_ALL}")
            return None
        
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
                self.state.mark_experiment_failed(experiment_id, error_msg)
                return {
                    "experiment_id": experiment_id,
                    "success": False,
                    "error": error_msg
                }
            
            generated_text = generation_result["generated_text"]
            
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
            
            # Mark as completed
            self.state.mark_experiment_completed(experiment_id)
            
            return experiment_result
            
        except Exception as e:
            error_msg = f"Experiment failed: {str(e)}"
            self.state.mark_experiment_failed(experiment_id, error_msg)
            return {
                "experiment_id": experiment_id,
                "success": False,
                "error": error_msg
            }
    
    def run_all_experiments(self, resume: bool = True) -> bool:
        """Run all experiments"""
        if not self.setup_experiment():
            return False
        
        paragraphs = self.corpus_processor.get_all_paragraphs()
        strategies = self.prompt_manager.get_strategy_names()
        models = self.llm_manager.get_available_models()
        repetitions = EXPERIMENT_CONFIG["repetitions_per_paragraph"]
        
        total_experiments = len(paragraphs) * len(strategies) * len(models) * repetitions
        
        print(f"\n{Fore.CYAN}Starting experiment execution...{Style.RESET_ALL}")
        start_time = time.time()
        
        # Create progress tracking
        progress_bar = tqdm(
            total=total_experiments,
            desc="Running experiments",
            unit="exp",
            initial=self.state.state.get("completed_count", 0)
        )
        
        experiment_count = 0
        successful_experiments = 0
        
        try:
            for paragraph_data in paragraphs:
                for strategy_name in strategies:
                    for model_name in models:
                        for repetition in range(1, repetitions + 1):
                            experiment_count += 1
                            
                            # Run single experiment
                            result = self.run_single_experiment(
                                paragraph_data, strategy_name, model_name, repetition
                            )
                            
                            if result:
                                if result.get("success", False):
                                    self.detailed_results.append(result)
                                    successful_experiments += 1
                                    
                                    # Save intermediate results periodically
                                    if len(self.detailed_results) % 10 == 0:
                                        self.save_intermediate_results()
                            
                            progress_bar.update(1)
                            
                            # Show progress update
                            progress = self.state.get_progress()
                            progress_bar.set_postfix({
                                'Success': f"{successful_experiments}/{experiment_count}",
                                'Time': format_time_elapsed(start_time)
                            })
        
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Experiment interrupted by user{Style.RESET_ALL}")
            print(f"Progress saved. You can resume later.")
            return False
        
        finally:
            progress_bar.close()
            self.save_final_results()
        
        total_time = time.time() - start_time
        
        print(f"\n{Fore.GREEN}Experiment completed!{Style.RESET_ALL}")
        print(f"Total time: {format_time_elapsed(start_time)}")
        print(f"Successful experiments: {successful_experiments}/{total_experiments}")
        print(f"Results saved to: {self.results_file}")
        
        return True
    
    def save_intermediate_results(self):
        """Save intermediate results"""
        try:
            temp_results = {
                "experiment_id": self.experiment_id,
                "timestamp": datetime.now().isoformat(),
                "status": "in_progress",
                "detailed_results": self.detailed_results,
                "progress": self.state.get_progress()
            }
            
            temp_file = RESULTS_DIR / f"temp_results_{self.experiment_id}.json"
            save_json(temp_results, temp_file)
            
        except Exception as e:
            print(f"{Fore.YELLOW}Warning: Could not save intermediate results: {e}{Style.RESET_ALL}")
    
    def save_final_results(self):
        """Save final experiment results"""
        try:
            # Calculate aggregate statistics
            aggregate_stats = self.evaluator.calculate_aggregate_scores(self.detailed_results)
            
            final_results = {
                "experiment_id": self.experiment_id,
                "timestamp": datetime.now().isoformat(),
                "status": "completed",
                "configuration": EXPERIMENT_CONFIG,
                "total_experiments": len(self.detailed_results),
                "aggregate_statistics": aggregate_stats,
                "detailed_results": self.detailed_results,
                "final_progress": self.state.get_progress()
            }
            
            save_json(final_results, self.results_file)
            print(f"{Fore.GREEN}✓ Final results saved to: {self.results_file}{Style.RESET_ALL}")
            
            # Also create CSV summary for easy analysis
            self.create_csv_summary(final_results)
            
            # Clean up temporary files
            temp_file = RESULTS_DIR / f"temp_results_{self.experiment_id}.json"
            if temp_file.exists():
                temp_file.unlink()
                
        except Exception as e:
            print(f"{Fore.RED}Error saving final results: {e}{Style.RESET_ALL}")
    
    def create_csv_summary(self, results: Dict):
        """Create CSV summary of results"""
        try:
            summary_data = []
            
            for result in results["detailed_results"]:
                if result.get("success", False):
                    evaluation = result["evaluation"]
                    summary_scores = evaluation["summary_scores"]
                    
                    summary_data.append({
                        "experiment_id": result["experiment_id"],
                        "paragraph_id": result["paragraph_id"],
                        "strategy": result["strategy"],
                        "model": result["model"],
                        "repetition": result["repetition"],
                        "bias_reduction_percentage": summary_scores["bias_reduction_percentage"],
                        "is_gender_neutral": summary_scores["is_gender_neutral"],
                        "fluency_score": summary_scores["fluency_score"],
                        "bleu_4_score": summary_scores["bleu_4_score"],
                        "semantic_similarity": summary_scores["semantic_similarity"],
                        "generation_time": result["generation_time"]
                    })
            
            df = pd.DataFrame(summary_data)
            csv_file = RESULTS_DIR / f"experiment_summary_{self.experiment_id}.csv"
            df.to_csv(csv_file, index=False)
            
            print(f"{Fore.GREEN}✓ CSV summary saved to: {csv_file}{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"{Fore.YELLOW}Warning: Could not create CSV summary: {e}{Style.RESET_ALL}")
    
    def print_experiment_summary(self):
        """Print a summary of the experiment setup"""
        print(f"\n{Fore.CYAN}=== EXPERIMENT SUMMARY ==={Style.RESET_ALL}")
        
        # Corpus info
        paragraphs = self.corpus_processor.get_all_paragraphs()
        print(f"Corpus: {len(paragraphs)} paragraphs")
        
        # Strategy info
        strategies = self.prompt_manager.get_strategy_names()
        print(f"Strategies: {len(strategies)}")
        for strategy in strategies:
            desc = self.prompt_manager.get_strategy_description(strategy)
            print(f"  - {strategy}: {desc}")
        
        # Model info
        if self.llm_manager:
            models = self.llm_manager.get_available_models()
            print(f"Models: {len(models)} ({', '.join(models)})")
        
        # Experiment info
        repetitions = EXPERIMENT_CONFIG["repetitions_per_paragraph"]
        total = len(paragraphs) * len(strategies) * (len(self.llm_manager.get_available_models()) if self.llm_manager else 0) * repetitions
        print(f"Total experiments: {total}")
        print(f"Estimated time: {total * 5 / 60:.1f} minutes (assuming 5 sec per experiment)")

def main():
    """Main function to run experiments"""
    print(f"{Fore.CYAN}Gender Bias in LLMs - Experiment Runner{Style.RESET_ALL}")
    print("=" * 50)
    
    # Initialize experiment runner
    runner = ExperimentRunner()
    
    # Print experiment summary
    runner.print_experiment_summary()
    
    # Ask user confirmation
    print(f"\n{Fore.YELLOW}Ready to start experiments?{Style.RESET_ALL}")
    response = input("Type 'yes' to continue, 'setup' to just setup, or 'quit' to exit: ").lower().strip()
    
    if response == "quit":
        print("Exiting...")
        return
    elif response == "setup":
        runner.setup_experiment()
        print("Setup complete. Run the script again to start experiments.")
        return
    elif response == "yes":
        # Run all experiments
        success = runner.run_all_experiments()
        if success:
            print(f"{Fore.GREEN}All experiments completed successfully!{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}Experiments stopped. You can resume later.{Style.RESET_ALL}")
    else:
        print("Invalid response. Exiting...")

if __name__ == "__main__":
    main()
