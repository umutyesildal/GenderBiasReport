"""
Statistical analysis and visualization for the Gender Bias study
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from scipy.stats import f_oneway
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import json
from datetime import datetime

from .config import VIZ_CONFIG, RESULTS_DIR
from .utils import load_json, save_json
from colorama import Fore, Style

class StatisticalAnalyzer:
    """Perform statistical analysis on experiment results"""
    
    def __init__(self, results_file: Optional[Path] = None):
        self.results_file = results_file
        self.results_data = None
        self.df = None
        
        # Set up visualization style
        plt.style.use('default')  # Using default instead of seaborn-v0_8
        sns.set_palette(VIZ_CONFIG["color_palette"])
    
    def load_results(self, results_file: Path) -> bool:
        """Load experiment results from JSON file"""
        try:
            self.results_data = load_json(results_file)
            if not self.results_data:
                print(f"{Fore.RED}Could not load results from {results_file}{Style.RESET_ALL}")
                return False
            
            # Convert to DataFrame for easier analysis
            self.df = self._create_dataframe()
            print(f"{Fore.GREEN}✓ Loaded {len(self.df)} experiment results{Style.RESET_ALL}")
            return True
            
        except Exception as e:
            print(f"{Fore.RED}Error loading results: {e}{Style.RESET_ALL}")
            return False
    
    def _create_dataframe(self) -> pd.DataFrame:
        """Create pandas DataFrame from results data"""
        rows = []
        
        for result in self.results_data.get("detailed_results", []):
            if result.get("success", False):
                evaluation = result["evaluation"]
                summary_scores = evaluation["summary_scores"]
                
                row = {
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
                    "generation_time": result["generation_time"],
                    
                    # Additional detailed metrics
                    "original_bias_score": evaluation["bias_evaluation"]["original_bias"]["bias_score"],
                    "generated_bias_score": evaluation["bias_evaluation"]["generated_bias"]["bias_score"],
                    "total_gendered_terms_original": evaluation["bias_evaluation"]["original_bias"]["total_gendered_terms"],
                    "total_gendered_terms_generated": evaluation["bias_evaluation"]["generated_bias"]["total_gendered_terms"],
                }
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def perform_anova(self, dependent_var: str, independent_var: str = "strategy") -> Dict[str, Any]:
        """Perform ANOVA test"""
        if self.df is None:
            raise ValueError("No data loaded. Call load_results() first.")
        
        # Group data by independent variable
        groups = []
        group_names = []
        
        for group_name in self.df[independent_var].unique():
            group_data = self.df[self.df[independent_var] == group_name][dependent_var]
            groups.append(group_data)
            group_names.append(group_name)
        
        # Perform one-way ANOVA
        f_stat, p_value = f_oneway(*groups)
        
        # Calculate effect size (eta-squared)
        ss_between = sum(len(group) * (group.mean() - self.df[dependent_var].mean())**2 for group in groups)
        ss_total = ((self.df[dependent_var] - self.df[dependent_var].mean())**2).sum()
        eta_squared = ss_between / ss_total
        
        # Group statistics
        group_stats = {}
        for i, group_name in enumerate(group_names):
            group_stats[group_name] = {
                "mean": groups[i].mean(),
                "std": groups[i].std(),
                "count": len(groups[i])
            }
        
        return {
            "dependent_variable": dependent_var,
            "independent_variable": independent_var,
            "f_statistic": f_stat,
            "p_value": p_value,
            "eta_squared": eta_squared,
            "significant": p_value < 0.05,
            "group_statistics": group_stats,
            "group_names": group_names
        }
    
    def perform_post_hoc_tests(self, dependent_var: str, independent_var: str = "strategy") -> Dict[str, Any]:
        """Perform post-hoc pairwise comparisons"""
        from scipy.stats import ttest_ind
        
        groups = {}
        for group_name in self.df[independent_var].unique():
            groups[group_name] = self.df[self.df[independent_var] == group_name][dependent_var]
        
        pairwise_results = {}
        group_names = list(groups.keys())
        
        for i in range(len(group_names)):
            for j in range(i+1, len(group_names)):
                group1_name = group_names[i]
                group2_name = group_names[j]
                
                t_stat, p_val = ttest_ind(groups[group1_name], groups[group2_name])
                
                pairwise_results[f"{group1_name}_vs_{group2_name}"] = {
                    "t_statistic": t_stat,
                    "p_value": p_val,
                    "significant": p_val < 0.05,
                    "mean_diff": groups[group1_name].mean() - groups[group2_name].mean()
                }
        
        return pairwise_results
    
    def calculate_descriptive_statistics(self) -> Dict[str, Any]:
        """Calculate descriptive statistics for all metrics"""
        if self.df is None:
            raise ValueError("No data loaded. Call load_results() first.")
        
        metrics = ["bias_reduction_percentage", "fluency_score", "bleu_4_score", "semantic_similarity"]
        
        descriptive_stats = {}
        
        # Overall statistics
        descriptive_stats["overall"] = {}
        for metric in metrics:
            descriptive_stats["overall"][metric] = {
                "mean": self.df[metric].mean(),
                "std": self.df[metric].std(),
                "min": self.df[metric].min(),
                "max": self.df[metric].max(),
                "median": self.df[metric].median(),
                "q25": self.df[metric].quantile(0.25),
                "q75": self.df[metric].quantile(0.75)
            }
        
        # Statistics by strategy
        descriptive_stats["by_strategy"] = {}
        for strategy in self.df["strategy"].unique():
            strategy_df = self.df[self.df["strategy"] == strategy]
            descriptive_stats["by_strategy"][strategy] = {}
            
            for metric in metrics:
                descriptive_stats["by_strategy"][strategy][metric] = {
                    "mean": strategy_df[metric].mean(),
                    "std": strategy_df[metric].std(),
                    "count": len(strategy_df)
                }
        
        # Statistics by model
        descriptive_stats["by_model"] = {}
        for model in self.df["model"].unique():
            model_df = self.df[self.df["model"] == model]
            descriptive_stats["by_model"][model] = {}
            
            for metric in metrics:
                descriptive_stats["by_model"][model][metric] = {
                    "mean": model_df[metric].mean(),
                    "std": model_df[metric].std(),
                    "count": len(model_df)
                }
        
        return descriptive_stats

class VisualizationGenerator:
    """Generate comprehensive visualizations for the study"""
    
    def __init__(self, df: pd.DataFrame, output_dir: Path = None):
        self.df = df
        self.output_dir = output_dir or RESULTS_DIR / "visualizations"
        self.output_dir.mkdir(exist_ok=True)
        
        # Set up style
        plt.style.use('default')
        sns.set_palette(VIZ_CONFIG["color_palette"])
    
    def create_all_visualizations(self) -> List[str]:
        """Create all visualizations and return list of saved files"""
        saved_files = []
        
        print(f"{Fore.CYAN}Generating visualizations...{Style.RESET_ALL}")
        
        # 1. Strategy comparison plots
        saved_files.extend(self.create_strategy_comparison_plots())
        
        # 2. Model comparison plots
        saved_files.extend(self.create_model_comparison_plots())
        
        # 3. Correlation analysis
        saved_files.extend(self.create_correlation_plots())
        
        # 4. Distribution plots
        saved_files.extend(self.create_distribution_plots())
        
        # 5. Time series and progression plots
        saved_files.extend(self.create_progression_plots())
        
        # 6. Detailed metric analysis
        saved_files.extend(self.create_detailed_metric_plots())
        
        # 7. Interactive plots
        saved_files.extend(self.create_interactive_plots())
        
        # 8. Summary dashboard
        saved_files.extend(self.create_summary_dashboard())
        
        print(f"{Fore.GREEN}✓ Generated {len(saved_files)} visualizations{Style.RESET_ALL}")
        return saved_files
    
    def create_strategy_comparison_plots(self) -> List[str]:
        """Create plots comparing different prompting strategies"""
        saved_files = []
        
        # 1. Box plots for each metric by strategy
        metrics = ["bias_reduction_percentage", "fluency_score", "bleu_4_score", "semantic_similarity"]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            sns.boxplot(data=self.df, x="strategy", y=metric, ax=axes[i])
            axes[i].set_title(f'{metric.replace("_", " ").title()} by Strategy')
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        file_path = self.output_dir / "strategy_comparison_boxplots.png"
        plt.savefig(file_path, dpi=VIZ_CONFIG["dpi"], bbox_inches='tight')
        plt.close()
        saved_files.append(str(file_path))
        
        # 2. Mean scores comparison
        strategy_means = self.df.groupby("strategy")[metrics].mean()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        strategy_means.plot(kind='bar', ax=ax)
        ax.set_title('Mean Scores by Strategy')
        ax.set_ylabel('Score')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        file_path = self.output_dir / "strategy_mean_scores.png"
        plt.savefig(file_path, dpi=VIZ_CONFIG["dpi"], bbox_inches='tight')
        plt.close()
        saved_files.append(str(file_path))
        
        # 3. Success rate (gender neutralization) by strategy
        neutralization_rates = self.df.groupby("strategy")["is_gender_neutral"].mean()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        neutralization_rates.plot(kind='bar', ax=ax, color=sns.color_palette()[0])
        ax.set_title('Gender Neutralization Success Rate by Strategy')
        ax.set_ylabel('Success Rate')
        ax.set_ylim(0, 1)
        
        # Add percentage labels on bars
        for i, v in enumerate(neutralization_rates):
            ax.text(i, v + 0.02, f'{v:.1%}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        file_path = self.output_dir / "neutralization_success_by_strategy.png"
        plt.savefig(file_path, dpi=VIZ_CONFIG["dpi"], bbox_inches='tight')
        plt.close()
        saved_files.append(str(file_path))
        
        return saved_files
    
    def create_model_comparison_plots(self) -> List[str]:
        """Create plots comparing different models"""
        saved_files = []
        
        if self.df["model"].nunique() > 1:
            metrics = ["bias_reduction_percentage", "fluency_score", "bleu_4_score"]
            
            # Side-by-side comparison
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            for i, metric in enumerate(metrics):
                sns.boxplot(data=self.df, x="model", y=metric, ax=axes[i])
                axes[i].set_title(f'{metric.replace("_", " ").title()} by Model')
            
            plt.tight_layout()
            file_path = self.output_dir / "model_comparison.png"
            plt.savefig(file_path, dpi=VIZ_CONFIG["dpi"], bbox_inches='tight')
            plt.close()
            saved_files.append(str(file_path))
        
        return saved_files
    
    def create_correlation_plots(self) -> List[str]:
        """Create correlation analysis plots"""
        saved_files = []
        
        # Correlation heatmap
        metrics = ["bias_reduction_percentage", "fluency_score", "bleu_4_score", "semantic_similarity", "generation_time"]
        corr_matrix = self.df[metrics].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Correlation Matrix of Evaluation Metrics')
        
        plt.tight_layout()
        file_path = self.output_dir / "correlation_matrix.png"
        plt.savefig(file_path, dpi=VIZ_CONFIG["dpi"], bbox_inches='tight')
        plt.close()
        saved_files.append(str(file_path))
        
        # Scatter plots for key relationships
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Bias reduction vs Fluency
        sns.scatterplot(data=self.df, x="bias_reduction_percentage", y="fluency_score", 
                       hue="strategy", ax=axes[0, 0])
        axes[0, 0].set_title('Bias Reduction vs Fluency')
        
        # Bias reduction vs BLEU score
        sns.scatterplot(data=self.df, x="bias_reduction_percentage", y="bleu_4_score", 
                       hue="strategy", ax=axes[0, 1])
        axes[0, 1].set_title('Bias Reduction vs BLEU Score')
        
        # Fluency vs BLEU score
        sns.scatterplot(data=self.df, x="fluency_score", y="bleu_4_score", 
                       hue="strategy", ax=axes[1, 0])
        axes[1, 0].set_title('Fluency vs BLEU Score')
        
        # Generation time vs bias reduction
        sns.scatterplot(data=self.df, x="generation_time", y="bias_reduction_percentage", 
                       hue="strategy", ax=axes[1, 1])
        axes[1, 1].set_title('Generation Time vs Bias Reduction')
        
        plt.tight_layout()
        file_path = self.output_dir / "scatter_plot_relationships.png"
        plt.savefig(file_path, dpi=VIZ_CONFIG["dpi"], bbox_inches='tight')
        plt.close()
        saved_files.append(str(file_path))
        
        return saved_files
    
    def create_distribution_plots(self) -> List[str]:
        """Create distribution plots for key metrics"""
        saved_files = []
        
        metrics = ["bias_reduction_percentage", "fluency_score", "bleu_4_score"]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        for i, metric in enumerate(metrics):
            # Histogram
            self.df[metric].hist(bins=20, ax=axes[0, i], alpha=0.7)
            axes[0, i].set_title(f'Distribution of {metric.replace("_", " ").title()}')
            axes[0, i].set_xlabel(metric.replace("_", " ").title())
            axes[0, i].set_ylabel('Frequency')
            
            # Distribution by strategy
            for strategy in self.df["strategy"].unique():
                strategy_data = self.df[self.df["strategy"] == strategy][metric]
                axes[1, i].hist(strategy_data, alpha=0.5, label=strategy, bins=15)
            
            axes[1, i].set_title(f'{metric.replace("_", " ").title()} by Strategy')
            axes[1, i].set_xlabel(metric.replace("_", " ").title())
            axes[1, i].set_ylabel('Frequency')
            axes[1, i].legend()
        
        plt.tight_layout()
        file_path = self.output_dir / "distribution_plots.png"
        plt.savefig(file_path, dpi=VIZ_CONFIG["dpi"], bbox_inches='tight')
        plt.close()
        saved_files.append(str(file_path))
        
        return saved_files
    
    def create_progression_plots(self) -> List[str]:
        """Create progression and time-based plots"""
        saved_files = []
        
        # Performance by repetition (learning effect)
        if self.df["repetition"].nunique() > 1:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.ravel()
            
            metrics = ["bias_reduction_percentage", "fluency_score", "bleu_4_score", "generation_time"]
            
            for i, metric in enumerate(metrics):
                repetition_means = self.df.groupby(["strategy", "repetition"])[metric].mean().reset_index()
                
                for strategy in self.df["strategy"].unique():
                    strategy_data = repetition_means[repetition_means["strategy"] == strategy]
                    axes[i].plot(strategy_data["repetition"], strategy_data[metric], 
                               marker='o', label=strategy)
                
                axes[i].set_title(f'{metric.replace("_", " ").title()} by Repetition')
                axes[i].set_xlabel('Repetition')
                axes[i].set_ylabel(metric.replace("_", " ").title())
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            file_path = self.output_dir / "progression_by_repetition.png"
            plt.savefig(file_path, dpi=VIZ_CONFIG["dpi"], bbox_inches='tight')
            plt.close()
            saved_files.append(str(file_path))
        
        return saved_files
    
    def create_detailed_metric_plots(self) -> List[str]:
        """Create detailed analysis plots for specific metrics"""
        saved_files = []
        
        # Bias reduction effectiveness
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Original vs Generated bias scores
        axes[0, 0].scatter(self.df["original_bias_score"], self.df["generated_bias_score"], 
                          c=self.df["strategy"].astype('category').cat.codes, alpha=0.6)
        axes[0, 0].plot([0, self.df["original_bias_score"].max()], 
                       [0, self.df["original_bias_score"].max()], 'r--', alpha=0.5)
        axes[0, 0].set_xlabel('Original Bias Score')
        axes[0, 0].set_ylabel('Generated Bias Score')
        axes[0, 0].set_title('Bias Score: Original vs Generated')
        
        # Gendered terms reduction
        axes[0, 1].scatter(self.df["total_gendered_terms_original"], 
                          self.df["total_gendered_terms_generated"], alpha=0.6)
        axes[0, 1].plot([0, self.df["total_gendered_terms_original"].max()], 
                       [0, self.df["total_gendered_terms_original"].max()], 'r--', alpha=0.5)
        axes[0, 1].set_xlabel('Original Gendered Terms Count')
        axes[0, 1].set_ylabel('Generated Gendered Terms Count')
        axes[0, 1].set_title('Gendered Terms: Original vs Generated')
        
        # Strategy effectiveness
        strategy_effectiveness = self.df.groupby("strategy").agg({
            "bias_reduction_percentage": "mean",
            "is_gender_neutral": "mean",
            "fluency_score": "mean"
        })
        
        x = range(len(strategy_effectiveness))
        width = 0.25
        
        axes[1, 0].bar([i - width for i in x], strategy_effectiveness["bias_reduction_percentage"], 
                      width, label='Bias Reduction %', alpha=0.8)
        axes[1, 0].bar(x, strategy_effectiveness["is_gender_neutral"] * 100, 
                      width, label='Neutralization Success %', alpha=0.8)
        axes[1, 0].bar([i + width for i in x], strategy_effectiveness["fluency_score"] * 100, 
                      width, label='Fluency Score * 100', alpha=0.8)
        
        axes[1, 0].set_xlabel('Strategy')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Strategy Effectiveness Comparison')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(strategy_effectiveness.index, rotation=45)
        axes[1, 0].legend()
        
        # Error analysis
        failed_neutralizations = self.df[self.df["is_gender_neutral"] == False]
        if len(failed_neutralizations) > 0:
            failed_by_strategy = failed_neutralizations["strategy"].value_counts()
            axes[1, 1].pie(failed_by_strategy.values, labels=failed_by_strategy.index, autopct='%1.1f%%')
            axes[1, 1].set_title('Failed Neutralizations by Strategy')
        else:
            axes[1, 1].text(0.5, 0.5, 'All neutralizations successful!', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Neutralization Success Rate')
        
        plt.tight_layout()
        file_path = self.output_dir / "detailed_metric_analysis.png"
        plt.savefig(file_path, dpi=VIZ_CONFIG["dpi"], bbox_inches='tight')
        plt.close()
        saved_files.append(str(file_path))
        
        return saved_files
    
    def create_interactive_plots(self) -> List[str]:
        """Create interactive plots using Plotly"""
        saved_files = []
        
        # Interactive scatter plot
        fig = px.scatter(self.df, x="bias_reduction_percentage", y="fluency_score",
                        color="strategy", size="bleu_4_score", hover_data=["paragraph_id", "model"],
                        title="Interactive Analysis: Bias Reduction vs Fluency")
        
        file_path = self.output_dir / "interactive_scatter.html"
        fig.write_html(file_path)
        saved_files.append(str(file_path))
        
        # Interactive box plots
        fig = px.box(self.df, x="strategy", y="bias_reduction_percentage", 
                    color="model", title="Bias Reduction by Strategy and Model")
        
        file_path = self.output_dir / "interactive_boxplot.html"
        fig.write_html(file_path)
        saved_files.append(str(file_path))
        
        return saved_files
    
    def create_summary_dashboard(self) -> List[str]:
        """Create a comprehensive summary dashboard"""
        saved_files = []
        
        # Create a 2x3 subplot layout for dashboard
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # 1. Overall performance by strategy
        strategy_means = self.df.groupby("strategy")["bias_reduction_percentage"].mean()
        axes[0, 0].bar(strategy_means.index, strategy_means.values)
        axes[0, 0].set_title('Mean Bias Reduction by Strategy')
        axes[0, 0].set_ylabel('Bias Reduction %')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Success rates
        success_rates = self.df.groupby("strategy")["is_gender_neutral"].mean()
        axes[0, 1].bar(success_rates.index, success_rates.values, color='green', alpha=0.7)
        axes[0, 1].set_title('Gender Neutralization Success Rate')
        axes[0, 1].set_ylabel('Success Rate')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Quality metrics
        quality_metrics = self.df.groupby("strategy")[["fluency_score", "bleu_4_score"]].mean()
        quality_metrics.plot(kind='bar', ax=axes[0, 2])
        axes[0, 2].set_title('Quality Metrics by Strategy')
        axes[0, 2].set_ylabel('Score')
        axes[0, 2].tick_params(axis='x', rotation=45)
        axes[0, 2].legend()
        
        # 4. Trade-off analysis
        axes[1, 0].scatter(self.df["bias_reduction_percentage"], self.df["fluency_score"], 
                          c=self.df["strategy"].astype('category').cat.codes, alpha=0.6)
        axes[1, 0].set_xlabel('Bias Reduction %')
        axes[1, 0].set_ylabel('Fluency Score')
        axes[1, 0].set_title('Trade-off: Bias Reduction vs Fluency')
        
        # 5. Performance distribution
        self.df["bias_reduction_percentage"].hist(bins=20, ax=axes[1, 1], alpha=0.7)
        axes[1, 1].axvline(self.df["bias_reduction_percentage"].mean(), color='red', 
                          linestyle='--', label=f'Mean: {self.df["bias_reduction_percentage"].mean():.1f}%')
        axes[1, 1].set_title('Distribution of Bias Reduction')
        axes[1, 1].set_xlabel('Bias Reduction %')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        
        # 6. Summary statistics table
        axes[1, 2].axis('tight')
        axes[1, 2].axis('off')
        
        summary_stats = self.df.groupby("strategy").agg({
            "bias_reduction_percentage": ["mean", "std"],
            "is_gender_neutral": "mean",
            "fluency_score": "mean"
        }).round(2)
        
        # Flatten column names
        summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns]
        
        table_data = summary_stats.reset_index().values
        table = axes[1, 2].table(cellText=table_data, 
                                colLabels=['Strategy'] + list(summary_stats.columns),
                                cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        axes[1, 2].set_title('Summary Statistics by Strategy')
        
        plt.tight_layout()
        file_path = self.output_dir / "summary_dashboard.png"
        plt.savefig(file_path, dpi=VIZ_CONFIG["dpi"], bbox_inches='tight')
        plt.close()
        saved_files.append(str(file_path))
        
        return saved_files

class ComprehensiveAnalyzer:
    """Main class combining statistical analysis and visualization"""
    
    def __init__(self, results_file: Path):
        self.analyzer = StatisticalAnalyzer()
        self.results_file = results_file
        
        if not self.analyzer.load_results(results_file):
            raise ValueError(f"Could not load results from {results_file}")
        
        self.visualizer = VisualizationGenerator(self.analyzer.df)
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """Run complete statistical analysis and generate all visualizations"""
        print(f"{Fore.CYAN}Running comprehensive analysis...{Style.RESET_ALL}")
        
        analysis_results = {}
        
        # 1. Descriptive statistics
        print("Computing descriptive statistics...")
        analysis_results["descriptive_stats"] = self.analyzer.calculate_descriptive_statistics()
        
        # 2. ANOVA tests for each metric
        print("Performing ANOVA tests...")
        metrics = ["bias_reduction_percentage", "fluency_score", "bleu_4_score", "semantic_similarity"]
        analysis_results["anova_results"] = {}
        
        for metric in metrics:
            analysis_results["anova_results"][metric] = self.analyzer.perform_anova(metric)
            
            # Post-hoc tests if ANOVA is significant
            if analysis_results["anova_results"][metric]["significant"]:
                analysis_results["anova_results"][metric]["post_hoc"] = self.analyzer.perform_post_hoc_tests(metric)
        
        # 3. Generate all visualizations
        print("Generating visualizations...")
        saved_visualizations = self.visualizer.create_all_visualizations()
        analysis_results["visualization_files"] = saved_visualizations
        
        # 4. Save complete analysis results
        analysis_file = RESULTS_DIR / f"complete_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_json(analysis_results, analysis_file)
        
        print(f"{Fore.GREEN}✓ Complete analysis saved to: {analysis_file}{Style.RESET_ALL}")
        print(f"✓ Generated {len(saved_visualizations)} visualizations")
        
        return analysis_results
    
    def print_analysis_summary(self, analysis_results: Dict[str, Any]):
        """Print a summary of the analysis results"""
        print(f"\n{Fore.CYAN}=== ANALYSIS SUMMARY ==={Style.RESET_ALL}")
        
        # ANOVA results summary
        print(f"\n{Fore.YELLOW}Statistical Significance Tests (ANOVA):{Style.RESET_ALL}")
        for metric, results in analysis_results["anova_results"].items():
            significance = "✓ Significant" if results["significant"] else "✗ Not significant"
            color = Fore.GREEN if results["significant"] else Fore.RED
            print(f"{color}{metric}: {significance} (p={results['p_value']:.4f}){Style.RESET_ALL}")
        
        # Best performing strategies
        print(f"\n{Fore.YELLOW}Strategy Performance Summary:{Style.RESET_ALL}")
        strategy_stats = analysis_results["descriptive_stats"]["by_strategy"]
        
        for metric in ["bias_reduction_percentage", "fluency_score"]:
            best_strategy = max(strategy_stats.keys(), 
                              key=lambda s: strategy_stats[s][metric]["mean"])
            best_score = strategy_stats[best_strategy][metric]["mean"]
            print(f"Best {metric}: {best_strategy} ({best_score:.2f})")

if __name__ == "__main__":
    # Find the most recent results file
    results_files = list(RESULTS_DIR.glob("experiment_results_*.json"))
    
    if not results_files:
        print(f"{Fore.RED}No experiment results found in {RESULTS_DIR}{Style.RESET_ALL}")
        print("Run the experiment first using experiment_runner.py")
    else:
        # Use the most recent results file
        latest_results = max(results_files, key=lambda f: f.stat().st_mtime)
        print(f"Analyzing results from: {latest_results}")
        
        analyzer = ComprehensiveAnalyzer(latest_results)
        analysis_results = analyzer.run_complete_analysis()
        analyzer.print_analysis_summary(analysis_results)
