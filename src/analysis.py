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
        plt.style.use('default')
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
        overall_mean = self.df[dependent_var].mean()
        ss_between = sum(len(group) * (group.mean() - overall_mean)**2 for group in groups)
        ss_total = sum((self.df[dependent_var] - overall_mean)**2)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        result = {
            "dependent_variable": dependent_var,
            "independent_variable": independent_var,
            "f_statistic": float(f_stat) if not np.isnan(f_stat) else 0.0,
            "p_value": float(p_value) if not np.isnan(p_value) else 0.0,
            "eta_squared": float(eta_squared),
            "groups": group_names,
            "group_means": [float(group.mean()) for group in groups],
            "group_stds": [float(group.std()) for group in groups],
            "significant": bool(p_value < 0.05)
        }
        
        return result
    
    def perform_post_hoc_tests(self, dependent_var: str, independent_var: str = "strategy") -> Dict[str, Any]:
        """Perform simplified post-hoc tests after ANOVA"""
        if self.df is None:
            raise ValueError("No data loaded. Call load_results() first.")
        
        from scipy.stats import ttest_ind
        
        groups = []
        group_names = list(self.df[independent_var].unique())
        
        for group_name in group_names:
            group_data = self.df[self.df[independent_var] == group_name][dependent_var]
            groups.append(group_data)
        
        # Perform pairwise t-tests with Bonferroni correction
        significant_pairs = []
        all_comparisons = []
        
        for i in range(len(group_names)):
            for j in range(i+1, len(group_names)):
                try:
                    t_stat, p_val = ttest_ind(groups[i], groups[j])
                    
                    comparison = {
                        "group1": group_names[i],
                        "group2": group_names[j],
                        "t_statistic": float(t_stat),
                        "p_value": float(p_val),
                        "mean_diff": float(groups[i].mean() - groups[j].mean())
                    }
                    all_comparisons.append(comparison)
                    
                    # Apply Bonferroni correction
                    bonferroni_alpha = 0.05 / len(group_names)
                    if p_val < bonferroni_alpha:
                        significant_pairs.append(comparison)
                        
                except Exception as e:
                    print(f"{Fore.YELLOW}Warning: t-test failed for {group_names[i]} vs {group_names[j]}: {e}{Style.RESET_ALL}")
        
        result = {
            "test_type": "Pairwise t-tests with Bonferroni correction",
            "groups": group_names,
            "bonferroni_alpha": 0.05 / len(group_names) if len(group_names) > 1 else 0.05,
            "all_comparisons": all_comparisons,
            "significant_pairs": significant_pairs
        }
        
        return result
    
    def calculate_descriptive_statistics(self) -> Dict[str, Any]:
        """Calculate descriptive statistics for all metrics"""
        if self.df is None:
            raise ValueError("No data loaded. Call load_results() first.")
        
        metrics = ["bias_reduction_percentage", "fluency_score", "bleu_4_score", "semantic_similarity", "generation_time"]
        
        descriptive_stats = {}
        
        # Overall statistics
        descriptive_stats["overall"] = {}
        for metric in metrics:
            descriptive_stats["overall"][metric] = {
                "mean": float(self.df[metric].mean()),
                "std": float(self.df[metric].std()),
                "min": float(self.df[metric].min()),
                "max": float(self.df[metric].max()),
                "median": float(self.df[metric].median()),
                "q25": float(self.df[metric].quantile(0.25)),
                "q75": float(self.df[metric].quantile(0.75))
            }
        
        # By strategy
        descriptive_stats["by_strategy"] = {}
        for strategy in self.df["strategy"].unique():
            strategy_data = self.df[self.df["strategy"] == strategy]
            descriptive_stats["by_strategy"][strategy] = {}
            
            for metric in metrics:
                descriptive_stats["by_strategy"][strategy][metric] = {
                    "mean": float(strategy_data[metric].mean()),
                    "std": float(strategy_data[metric].std()),
                    "count": int(len(strategy_data))
                }
        
        # By model
        descriptive_stats["by_model"] = {}
        for model in self.df["model"].unique():
            model_data = self.df[self.df["model"] == model]
            descriptive_stats["by_model"][model] = {}
            
            for metric in metrics:
                descriptive_stats["by_model"][model][metric] = {
                    "mean": float(model_data[metric].mean()),
                    "std": float(model_data[metric].std()),
                    "count": int(len(model_data))
                }
        
        return descriptive_stats

class VisualizationGenerator:
    """Generate essential visualizations for the study"""
    
    def __init__(self, df: pd.DataFrame, output_dir: Path = None):
        self.df = df
        self.output_dir = output_dir or RESULTS_DIR / "visualizations"
        self.output_dir.mkdir(exist_ok=True)
        
        # Set up style
        plt.style.use('default')
        sns.set_palette(VIZ_CONFIG["color_palette"])
    
    def create_all_visualizations(self) -> List[str]:
        """Create essential visualizations and return list of saved files"""
        saved_files = []
        
        print(f"{Fore.CYAN}Generating essential visualizations...{Style.RESET_ALL}")
        
        # Only create essential visualizations to avoid clutter
        # 1. Strategy comparison plots (most important) - properly scaled
        saved_files.extend(self.create_strategy_comparison_plots())
        
        # 2. One key interactive plot for detailed exploration
        saved_files.extend(self.create_key_interactive_plot())
        
        # 3. Compact summary dashboard
        saved_files.extend(self.create_compact_summary())
        
        print(f"{Fore.GREEN}✓ Generated {len(saved_files)} essential visualizations{Style.RESET_ALL}")
        return saved_files
    
    def create_strategy_comparison_plots(self) -> List[str]:
        """Create properly scaled plots comparing different prompting strategies"""
        saved_files = []
        
        # Create a comprehensive comparison with appropriate scaling
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Bias Reduction Percentage (0-100 scale)
        sns.boxplot(data=self.df, x="strategy", y="bias_reduction_percentage", ax=axes[0,0])
        axes[0,0].set_title('Bias Reduction by Strategy', fontsize=14, fontweight='bold')
        axes[0,0].set_ylabel('Bias Reduction (%)')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].set_ylim(-10, 110)  # Allow for some outliers
        
        # Add mean values as text
        for i, strategy in enumerate(self.df['strategy'].unique()):
            mean_val = self.df[self.df['strategy'] == strategy]['bias_reduction_percentage'].mean()
            axes[0,0].text(i, mean_val + 5, f'{mean_val:.1f}%', ha='center', fontweight='bold')
        
        # 2. Gender Neutralization Success Rate
        neutralization_rates = self.df.groupby("strategy")["is_gender_neutral"].mean()
        bars = axes[0,1].bar(range(len(neutralization_rates)), neutralization_rates.values, 
                           color=sns.color_palette("viridis", len(neutralization_rates)))
        axes[0,1].set_title('Gender Neutralization Success Rate by Strategy', fontsize=14, fontweight='bold')
        axes[0,1].set_ylabel('Success Rate')
        axes[0,1].set_ylim(0, 1.1)
        axes[0,1].set_xticks(range(len(neutralization_rates)))
        axes[0,1].set_xticklabels(neutralization_rates.index, rotation=45)
        
        # Add percentage labels on bars
        for i, v in enumerate(neutralization_rates.values):
            axes[0,1].text(i, v + 0.05, f'{v:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Fluency Score (0-1 scale)
        sns.boxplot(data=self.df, x="strategy", y="fluency_score", ax=axes[1,0])
        axes[1,0].set_title('Fluency Score by Strategy', fontsize=14, fontweight='bold')
        axes[1,0].set_ylabel('Fluency Score (0-1)')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].set_ylim(0, 1.05)
        
        # 4. Semantic Similarity (0-1 scale)
        sns.boxplot(data=self.df, x="strategy", y="semantic_similarity", ax=axes[1,1])
        axes[1,1].set_title('Semantic Similarity by Strategy', fontsize=14, fontweight='bold')
        axes[1,1].set_ylabel('Semantic Similarity (0-1)')
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].set_ylim(0, 1.05)
        
        plt.tight_layout()
        file_path = self.output_dir / "strategy_comparison_comprehensive.png"
        plt.savefig(file_path, dpi=VIZ_CONFIG["dpi"], bbox_inches='tight')
        plt.close()
        saved_files.append(str(file_path))
        
        return saved_files
    
    def create_key_interactive_plot(self) -> List[str]:
        """Create one key interactive plot for detailed exploration"""
        saved_files = []
        
        # Interactive scatter plot: Bias Reduction vs Quality metrics
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Bias Reduction vs Fluency", "Bias Reduction vs Semantic Similarity"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        strategies = self.df['strategy'].unique()
        colors = px.colors.qualitative.Set1[:len(strategies)]
        
        for i, strategy in enumerate(strategies):
            strategy_data = self.df[self.df['strategy'] == strategy]
            
            # Plot 1: Bias Reduction vs Fluency
            fig.add_trace(
                go.Scatter(
                    x=strategy_data['bias_reduction_percentage'],
                    y=strategy_data['fluency_score'],
                    mode='markers',
                    name=f'{strategy} (Fluency)',
                    marker=dict(color=colors[i], size=8),
                    text=[f"ID: {id}<br>Fluency: {fluency:.3f}<br>Bias Reduction: {bias:.1f}%" 
                          for id, fluency, bias in zip(strategy_data['experiment_id'], 
                                                     strategy_data['fluency_score'], 
                                                     strategy_data['bias_reduction_percentage'])],
                    hovertemplate='<b>%{text}</b><extra></extra>',
                    showlegend=True
                ),
                row=1, col=1
            )
            
            # Plot 2: Bias Reduction vs Semantic Similarity
            fig.add_trace(
                go.Scatter(
                    x=strategy_data['bias_reduction_percentage'],
                    y=strategy_data['semantic_similarity'],
                    mode='markers',
                    name=f'{strategy} (Similarity)',
                    marker=dict(color=colors[i], size=8),
                    text=[f"ID: {id}<br>Similarity: {sim:.3f}<br>Bias Reduction: {bias:.1f}%" 
                          for id, sim, bias in zip(strategy_data['experiment_id'], 
                                                 strategy_data['semantic_similarity'], 
                                                 strategy_data['bias_reduction_percentage'])],
                    hovertemplate='<b>%{text}</b><extra></extra>',
                    showlegend=False
                ),
                row=1, col=2
            )
        
        fig.update_xaxes(title_text="Bias Reduction (%)", row=1, col=1)
        fig.update_xaxes(title_text="Bias Reduction (%)", row=1, col=2)
        fig.update_yaxes(title_text="Fluency Score", row=1, col=1)
        fig.update_yaxes(title_text="Semantic Similarity", row=1, col=2)
        
        fig.update_layout(
            title="Interactive Analysis: Bias Reduction vs Quality Metrics",
            height=500,
            hovermode='closest'
        )
        
        file_path = self.output_dir / "interactive_bias_quality_analysis.html"
        fig.write_html(str(file_path))
        saved_files.append(str(file_path))
        
        return saved_files
    
    def create_compact_summary(self) -> List[str]:
        """Create a compact summary dashboard"""
        saved_files = []
        
        # Calculate key statistics
        overall_stats = {
            'total_experiments': len(self.df),
            'strategies_tested': self.df['strategy'].nunique(),
            'avg_bias_reduction': self.df['bias_reduction_percentage'].mean(),
            'neutralization_success_rate': self.df['is_gender_neutral'].mean(),
            'avg_fluency': self.df['fluency_score'].mean(),
            'avg_semantic_similarity': self.df['semantic_similarity'].mean()
        }
        
        strategy_stats = self.df.groupby('strategy').agg({
            'bias_reduction_percentage': ['mean', 'std'],
            'is_gender_neutral': 'mean',
            'fluency_score': 'mean',
            'semantic_similarity': 'mean'
        }).round(3)
        
        # Create summary plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Gender Bias Study - Summary Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Overall metrics bar chart
        metrics = ['Avg Bias Reduction (%)', 'Neutralization Rate (%)', 'Avg Fluency', 'Avg Similarity']
        values = [overall_stats['avg_bias_reduction'], 
                 overall_stats['neutralization_success_rate'] * 100,
                 overall_stats['avg_fluency'] * 100,
                 overall_stats['avg_semantic_similarity'] * 100]
        
        bars = axes[0,0].bar(metrics, values, color=['#e74c3c', '#27ae60', '#3498db', '#f39c12'])
        axes[0,0].set_title('Overall Performance Metrics')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[0,0].text(bar.get_x() + bar.get_width()/2., height + 1,
                          f'{value:.1f}%' if 'Rate' in metrics[values.index(value)] or 'Reduction' in metrics[values.index(value)] 
                          else f'{value:.1f}',
                          ha='center', va='bottom', fontweight='bold')
        
        # 2. Strategy performance heatmap
        heatmap_data = strategy_stats.xs('mean', level=1, axis=1)
        heatmap_data.columns = ['Bias Reduction %', 'Neutralization Rate', 'Fluency', 'Similarity']
        
        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn', 
                   ax=axes[0,1], cbar_kws={'label': 'Score'})
        axes[0,1].set_title('Strategy Performance Heatmap')
        axes[0,1].set_xlabel('')
        
        # 3. Distribution of bias reduction
        axes[1,0].hist(self.df['bias_reduction_percentage'], bins=10, alpha=0.7, color='#3498db', edgecolor='black')
        axes[1,0].axvline(overall_stats['avg_bias_reduction'], color='red', linestyle='--', linewidth=2, label=f'Mean: {overall_stats["avg_bias_reduction"]:.1f}%')
        axes[1,0].set_title('Distribution of Bias Reduction')
        axes[1,0].set_xlabel('Bias Reduction (%)')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].legend()
        
        # 4. Key statistics text
        axes[1,1].axis('off')
        stats_text = f"""
KEY FINDINGS:

• Total Experiments: {overall_stats['total_experiments']}
• Strategies Tested: {overall_stats['strategies_tested']}

• Average Bias Reduction: {overall_stats['avg_bias_reduction']:.1f}%
• Gender Neutralization Success: {overall_stats['neutralization_success_rate']:.1%}

• Average Fluency Score: {overall_stats['avg_fluency']:.3f}
• Average Semantic Similarity: {overall_stats['avg_semantic_similarity']:.3f}

Best Strategy: {strategy_stats.xs('mean', level=1, axis=1)['bias_reduction_percentage'].idxmax()}
"""
        axes[1,1].text(0.1, 0.9, stats_text, transform=axes[1,1].transAxes, 
                      fontsize=12, verticalalignment='top', 
                      bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        file_path = self.output_dir / "summary_dashboard.png"
        plt.savefig(file_path, dpi=VIZ_CONFIG["dpi"], bbox_inches='tight')
        plt.close()
        saved_files.append(str(file_path))
        
        return saved_files

class ComprehensiveAnalyzer:
    """Main analyzer class that combines statistical analysis and visualization"""
    
    def __init__(self, results_file: Path):
        self.results_file = results_file
        self.analyzer = StatisticalAnalyzer(results_file)
        self.visualizer = None
        
    def run_complete_analysis(self) -> Dict[str, Any]:
        """Run complete statistical analysis and generate visualizations"""
        print(f"{Fore.CYAN}Running comprehensive analysis...{Style.RESET_ALL}")
        
        # Load results
        if not self.analyzer.load_results(self.results_file):
            return None
        
        # Initialize visualizer
        self.visualizer = VisualizationGenerator(self.analyzer.df)
        
        analysis_results = {
            "file_path": str(self.results_file),
            "timestamp": datetime.now().isoformat(),
            "statistical_analysis": {},
            "descriptive_statistics": {},
            "visualizations": []
        }
        
        # Descriptive statistics
        print(f"{Fore.CYAN}Computing descriptive statistics...{Style.RESET_ALL}")
        analysis_results["descriptive_statistics"] = self.analyzer.calculate_descriptive_statistics()
        
        # ANOVA tests for key metrics
        print(f"{Fore.CYAN}Performing ANOVA tests...{Style.RESET_ALL}")
        key_metrics = ["bias_reduction_percentage", "fluency_score", "semantic_similarity"]
        
        for metric in key_metrics:
            try:
                anova_result = self.analyzer.perform_anova(metric, "strategy")
                analysis_results["statistical_analysis"][f"anova_{metric}"] = anova_result
                
                # Post-hoc tests if ANOVA is significant
                if anova_result["significant"]:
                    posthoc_result = self.analyzer.perform_post_hoc_tests(metric, "strategy")
                    analysis_results["statistical_analysis"][f"posthoc_{metric}"] = posthoc_result
                    
            except Exception as e:
                print(f"{Fore.YELLOW}Warning: Analysis failed for {metric}: {e}{Style.RESET_ALL}")
        
        # Generate visualizations
        print(f"{Fore.CYAN}Generating visualizations...{Style.RESET_ALL}")
        viz_files = self.visualizer.create_all_visualizations()
        analysis_results["visualizations"] = viz_files
        
        # Save analysis results
        output_file = self.results_file.parent / f"complete_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_json(analysis_results, output_file)
        
        print(f"{Fore.GREEN}✓ Complete analysis saved to: {output_file}{Style.RESET_ALL}")
        return analysis_results

def analyze_results(results_file: Path) -> Dict[str, Any]:
    """Main function to analyze experiment results"""
    analyzer = ComprehensiveAnalyzer(results_file)
    return analyzer.run_complete_analysis()
