"""
Improved Statistical Analysis and Visualization for Gender Bias Study
Focus: Bias Reduction, BLEU-4 Scores, ANOVA Analysis of Repetitions
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
from typing import Dict, List, Any, Tuple, Optional, Union
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from config import VIZ_CONFIG, RESULTS_DIR
from utils import load_json, save_json
from colorama import Fore, Style

class ImprovedAnalyzer:
    """Improved analyzer focusing on key metrics and proper ANOVA analysis"""
    
    def __init__(self, results_file=None):
        self.results_file = results_file
        self.results_data = None
        self.df = None
        
        # Set up beautiful visualization style
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except:
            plt.style.use('seaborn-whitegrid')
        sns.set_palette("husl")
        
        # Custom color scheme for strategies
        self.strategy_colors = {
            'raw': '#e74c3c',           # Red - worst performance
            'system': '#f39c12',        # Orange - medium
            'few_shot': '#3498db',      # Blue - good 
            'few_shot_verification': '#27ae60'  # Green - best performance
        }
    
    def load_and_prepare_data(self, results_file) -> bool:
        """Load and prepare data with focus on key metrics"""
        try:
            self.results_data = load_json(results_file)
            if not self.results_data:
                print(f"{Fore.RED}Could not load results from {results_file}{Style.RESET_ALL}")
                return False
            
            # Create focused DataFrame
            self.df = self._create_focused_dataframe()
            print(f"{Fore.GREEN}✓ Loaded {len(self.df)} experiments for analysis{Style.RESET_ALL}")
            print(f"  - Strategies: {', '.join(self.df['strategy'].unique())}")
            print(f"  - Models: {', '.join(self.df['model'].unique())}")
            print(f"  - Repetitions per strategy: {self.df['repetition'].nunique()}")
            return True
            
        except Exception as e:
            print(f"{Fore.RED}Error loading results: {e}{Style.RESET_ALL}")
            return False
    
    def _create_focused_dataframe(self) -> pd.DataFrame:
        """Create DataFrame focused on bias reduction and semantic preservation"""
        rows = []
        
        for result in self.results_data.get("detailed_results", []):
            if result.get("success", False):
                evaluation = result["evaluation"]
                summary_scores = evaluation["summary_scores"]
                
                # Extract key information
                row = {
                    "experiment_id": result["experiment_id"],
                    "paragraph_id": result["paragraph_id"],
                    "strategy": result["strategy"],
                    "model": result["model"],
                    "repetition": result["repetition"],
                    
                    # PRIMARY METRICS (most important)
                    "bias_reduction_percentage": summary_scores["bias_reduction_percentage"],
                    "is_gender_neutral": summary_scores["is_gender_neutral"],
                    "bleu_4_score": summary_scores["bleu_4_score"],
                    
                    # SECONDARY METRICS
                    "semantic_similarity": summary_scores["semantic_similarity"],
                    "fluency_score": summary_scores["fluency_score"],
                    
                    # DETAILED BIAS METRICS
                    "original_bias_score": evaluation["bias_evaluation"]["original_bias"]["bias_score"],
                    "generated_bias_score": evaluation["bias_evaluation"]["generated_bias"]["bias_score"],
                    "original_gendered_terms": evaluation["bias_evaluation"]["original_bias"]["total_gendered_terms"],
                    "generated_gendered_terms": evaluation["bias_evaluation"]["generated_bias"]["total_gendered_terms"],
                    
                    # PERFORMANCE METRICS
                    "generation_time": result["generation_time"],
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Create additional derived metrics
        df['bias_reduction_absolute'] = df['original_bias_score'] - df['generated_bias_score']
        df['terms_eliminated'] = df['original_gendered_terms'] - df['generated_gendered_terms']
        df['neutralization_success'] = df['is_gender_neutral'].astype(int)
        
        return df
    
    def perform_comprehensive_anova(self) -> Dict[str, Any]:
        """Perform comprehensive ANOVA analysis focusing on repetitions"""
        if self.df is None:
            raise ValueError("No data loaded. Call load_and_prepare_data() first.")
        
        print(f"{Fore.CYAN}Performing comprehensive ANOVA analysis...{Style.RESET_ALL}")
        
        results = {}
        key_metrics = ["bias_reduction_percentage", "bleu_4_score", "neutralization_success"]
        
        for metric in key_metrics:
            print(f"  Analyzing {metric}...")
            
            # 1. Strategy comparison (main effect)
            strategy_groups = [
                self.df[self.df['strategy'] == strategy][metric].values 
                for strategy in self.df['strategy'].unique()
            ]
            
            f_stat, p_value = f_oneway(*strategy_groups)
            
            # Calculate effect size (eta-squared)
            overall_mean = self.df[metric].mean()
            ss_between = sum(len(group) * (np.mean(group) - overall_mean)**2 for group in strategy_groups)
            ss_total = sum((self.df[metric] - overall_mean)**2)
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            
            # 2. Repetition consistency analysis
            repetition_consistency = self._analyze_repetition_consistency(metric)
            
            # 3. Strategy performance ranking
            strategy_means = self.df.groupby('strategy')[metric].agg(['mean', 'std', 'count']).round(4)
            strategy_ranking = strategy_means.sort_values('mean', ascending=False)
            
            results[metric] = {
                "anova": {
                    "f_statistic": float(f_stat),
                    "p_value": float(p_value),
                    "eta_squared": float(eta_squared),
                    "significant": bool(p_value < 0.05),
                    "effect_size": self._interpret_effect_size(eta_squared)
                },
                "strategy_ranking": strategy_ranking.to_dict(),
                "repetition_consistency": repetition_consistency,
                "summary": {
                    "best_strategy": strategy_ranking.index[0],
                    "worst_strategy": strategy_ranking.index[-1],
                    "performance_range": float(strategy_ranking['mean'].iloc[0] - strategy_ranking['mean'].iloc[-1])
                }
            }
        
        return results
    
    def _analyze_repetition_consistency(self, metric: str) -> Dict[str, Any]:
        """Analyze consistency across the 3 repetitions"""
        consistency_results = {}
        
        for strategy in self.df['strategy'].unique():
            strategy_data = self.df[self.df['strategy'] == strategy]
            
            # Group by paragraph and calculate variance across repetitions
            paragraph_variances = []
            for paragraph_id in strategy_data['paragraph_id'].unique():
                paragraph_reps = strategy_data[strategy_data['paragraph_id'] == paragraph_id][metric]
                if len(paragraph_reps) >= 2:  # Need at least 2 repetitions
                    paragraph_variances.append(paragraph_reps.var())
            
            consistency_results[strategy] = {
                "mean_variance": float(np.mean(paragraph_variances)) if paragraph_variances else 0.0,
                "consistency_score": float(1 / (1 + np.mean(paragraph_variances))) if paragraph_variances else 0.0,
                "repetition_reliability": "High" if np.mean(paragraph_variances) < 0.01 else "Medium" if np.mean(paragraph_variances) < 0.05 else "Low"
            }
        
        return consistency_results
    
    def _interpret_effect_size(self, eta_squared: float) -> str:
        """Interpret eta-squared effect size"""
        if eta_squared >= 0.14:
            return "Large"
        elif eta_squared >= 0.06:
            return "Medium"
        elif eta_squared >= 0.01:
            return "Small"
        else:
            return "Negligible"

class ImprovedVisualizer:
    """Create focused, publication-ready visualizations"""
    
    def __init__(self, df: pd.DataFrame, output_dir: Path = None):
        self.df = df
        self.output_dir = output_dir or RESULTS_DIR / "improved_visualizations"
        self.output_dir.mkdir(exist_ok=True)
        
        # Beautiful color scheme
        self.strategy_colors = {
            'raw': '#e74c3c',           # Red - worst
            'system': '#f39c12',        # Orange - medium
            'few_shot': '#3498db',      # Blue - good 
            'few_shot_verification': '#27ae60'  # Green - best
        }
        
        # Set style
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except:
            plt.style.use('seaborn-whitegrid')
        sns.set_context("paper", font_scale=1.2)
    
    def create_bias_reduction_masterplot(self) -> str:
        """Create the main bias reduction visualization - the star of the show"""
        print(f"{Fore.CYAN}Creating bias reduction masterplot...{Style.RESET_ALL}")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Gender Bias Reduction Analysis - Comprehensive Overview', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        # 1. Main plot: Bias Reduction by Strategy (with individual points + stats)
        ax1 = axes[0, 0]
        
        # Box plot with individual points
        box_parts = ax1.boxplot([self.df[self.df['strategy'] == s]['bias_reduction_percentage'].values 
                                for s in ['raw', 'system', 'few_shot', 'few_shot_verification']], 
                               labels=['Raw', 'System', 'Few-Shot', 'FS + Verification'],
                               patch_artist=True, showfliers=False)
        
        # Color the boxes
        colors = ['#e74c3c', '#f39c12', '#3498db', '#27ae60']
        for patch, color in zip(box_parts['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Add scatter points
        for i, strategy in enumerate(['raw', 'system', 'few_shot', 'few_shot_verification']):
            strategy_data = self.df[self.df['strategy'] == strategy]['bias_reduction_percentage']
            y = strategy_data.values
            x = np.random.normal(i + 1, 0.04, size=len(y))  # Add jitter
            ax1.scatter(x, y, alpha=0.6, color=colors[i], s=30)
        
        # Add mean values as text
        for i, strategy in enumerate(['raw', 'system', 'few_shot', 'few_shot_verification']):
            mean_val = self.df[self.df['strategy'] == strategy]['bias_reduction_percentage'].mean()
            ax1.text(i + 1, mean_val + 2, f'{mean_val:.1f}%', 
                    ha='center', va='bottom', fontweight='bold', fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        ax1.set_title('Bias Reduction Performance by Strategy', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Bias Reduction Percentage (%)', fontsize=14)
        ax1.set_xlabel('Prompting Strategy', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-5, 105)
        
        # 2. Success Rate (Gender Neutralization)
        ax2 = axes[0, 1]
        success_rates = self.df.groupby('strategy')['is_gender_neutral'].mean() * 100
        strategies = ['raw', 'system', 'few_shot', 'few_shot_verification']
        strategy_labels = ['Raw', 'System', 'Few-Shot', 'FS + Verification']
        
        bars = ax2.bar(strategy_labels, [success_rates[s] for s in strategies], 
                      color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add percentage labels on bars
        for bar, strategy in zip(bars, strategies):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{success_rates[strategy]:.1f}%',
                    ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        ax2.set_title('Gender Neutralization Success Rate', fontsize=16, fontweight='bold')
        ax2.set_ylabel('Success Rate (%)', fontsize=14)
        ax2.set_xlabel('Prompting Strategy', fontsize=14)
        ax2.set_ylim(0, 105)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. BLEU-4 Score Analysis
        ax3 = axes[1, 0]
        
        # Violin plot for BLEU-4 scores
        violin_parts = ax3.violinplot([self.df[self.df['strategy'] == s]['bleu_4_score'].values 
                                      for s in strategies], 
                                     positions=range(1, 5), showmeans=True, showmedians=True)
        
        # Color the violins
        for i, pc in enumerate(violin_parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
        
        ax3.set_title('Semantic Preservation (BLEU-4 Score)', fontsize=16, fontweight='bold')
        ax3.set_ylabel('BLEU-4 Score', fontsize=14)
        ax3.set_xlabel('Prompting Strategy', fontsize=14)
        ax3.set_xticks(range(1, 5))
        ax3.set_xticklabels(strategy_labels)
        ax3.grid(True, alpha=0.3)
        
        # Add mean values
        for i, strategy in enumerate(strategies):
            mean_val = self.df[self.df['strategy'] == strategy]['bleu_4_score'].mean()
            ax3.text(i + 1, mean_val + 0.02, f'{mean_val:.3f}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=11,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        
        # 4. Repetition Consistency Analysis
        ax4 = axes[1, 1]
        
        # Calculate consistency across repetitions
        consistency_data = []
        consistency_labels = []
        
        for strategy in strategies:
            strategy_data = self.df[self.df['strategy'] == strategy]
            # Calculate standard deviation across repetitions for each paragraph
            paragraph_stds = []
            for paragraph_id in strategy_data['paragraph_id'].unique():
                paragraph_reps = strategy_data[strategy_data['paragraph_id'] == paragraph_id]['bias_reduction_percentage']
                if len(paragraph_reps) >= 2:
                    paragraph_stds.append(paragraph_reps.std())
            
            if paragraph_stds:
                consistency_data.append(np.mean(paragraph_stds))
                consistency_labels.append(['Raw', 'System', 'Few-Shot', 'FS + Verification'][strategies.index(strategy)])
        
        bars = ax4.bar(range(len(consistency_data)), consistency_data, 
                      color=colors[:len(consistency_data)], alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add values on bars
        for i, (bar, value) in enumerate(zip(bars, consistency_data)):
            ax4.text(bar.get_x() + bar.get_width()/2., value + 0.2,
                    f'{value:.2f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        ax4.set_title('Repetition Consistency (Lower = More Consistent)', fontsize=16, fontweight='bold')
        ax4.set_ylabel('Average Standard Deviation', fontsize=14)
        ax4.set_xlabel('Prompting Strategy', fontsize=14)
        ax4.set_xticks(range(len(consistency_labels)))
        ax4.set_xticklabels(consistency_labels)
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        file_path = self.output_dir / "bias_reduction_masterplot.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"{Fore.GREEN}✓ Created bias reduction masterplot: {file_path}{Style.RESET_ALL}")
        return str(file_path)
    
    def create_anova_visualization(self, anova_results: Dict[str, Any]) -> str:
        """Create simplified ANOVA visualization with bias-BLEU scatterplot and key statistics"""
        print(f"{Fore.CYAN}Creating simplified ANOVA analysis visualization...{Style.RESET_ALL}")
        
        # Create figure with custom layout
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, height_ratios=[0.8, 1.5, 0.7], width_ratios=[1, 1, 1])
        
        # Main title
        fig.suptitle('Statistical Analysis: ANOVA Results & Performance Summary', 
                    fontsize=20, fontweight='bold', y=0.95)
        
        # Color scheme
        colors = ['#ff6b9d', '#4ecdc4', '#45b7d1', '#96ceb4']
        strategies = ['raw', 'system', 'few_shot', 'few_shot_verification']
        strategy_labels = ['Raw', 'System Prompt', 'Few-Shot', 'Few-Shot + Verification']
        
        # 1. ANOVA Results Bar Chart (Top row, spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        
        # Get F-statistics for bias reduction and BLEU-4
        f_bias = anova_results["bias_reduction_percentage"]["anova"]["f_statistic"]
        f_bleu = anova_results["bleu_4_score"]["anova"]["f_statistic"]
        p_bias = anova_results["bias_reduction_percentage"]["anova"]["p_value"]
        p_bleu = anova_results["bleu_4_score"]["anova"]["p_value"]
        
        metrics = ['Bias Reduction', 'BLEU-4']
        f_stats = [f_bias, f_bleu]
        p_values = [p_bias, p_bleu]
        
        bars = ax1.bar(metrics, f_stats, color=['#3498db', '#e74c3c'], alpha=0.8, width=0.6)
        
        # Add F-statistics and significance
        for i, (bar, f_stat, p_val) in enumerate(zip(bars, f_stats, p_values)):
            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'F = {f_stat:.1f}\\n{significance}',
                    ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        ax1.set_title('ANOVA Results (F-Statistics)', fontsize=16, fontweight='bold', pad=20)
        ax1.set_ylabel('F-Statistic', fontsize=12)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0, max(f_stats) * 1.2)
        
        # 2. Key Statistics Summary (Top right)
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis('off')
        
        # Calculate key statistics
        best_strategy = self.df.groupby('strategy')['bias_reduction_percentage'].mean().idxmax()
        best_bias_reduction = self.df.groupby('strategy')['bias_reduction_percentage'].mean().max()
        median_bias_overall = self.df['bias_reduction_percentage'].median()
        median_bleu_overall = self.df['bleu_4_score'].median()
        
        # Strategy-specific medians
        strategy_medians = self.df.groupby('strategy')['bias_reduction_percentage'].median()
        
        stats_text = f"""
        📊 KEY STATISTICS
        
        🏆 Best Strategy: {best_strategy.replace('_', ' ').title()}
        📈 Best Performance: {best_bias_reduction:.1f}%
        
        📊 Overall Medians:
        • Bias Reduction: {median_bias_overall:.1f}%
        • BLEU-4 Score: {median_bleu_overall:.3f}
        
        📋 Median by Strategy:
        • Raw: {strategy_medians.get('raw', 0):.1f}%
        • System: {strategy_medians.get('system', 0):.1f}%
        • Few-Shot: {strategy_medians.get('few_shot', 0):.1f}%
        • FS+Ver: {strategy_medians.get('few_shot_verification', 0):.1f}%
        """
        
        ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, fontsize=11,
                verticalalignment='top', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
        
        # 3. Main Bias Reduction vs BLEU-4 Scatterplot (Middle row, spans all columns)
        ax3 = fig.add_subplot(gs[1, :])
        
        # Create scatterplot for each strategy
        for i, (strategy, color, label) in enumerate(zip(strategies, colors, strategy_labels)):
            strategy_data = self.df[self.df['strategy'] == strategy]
            
            scatter = ax3.scatter(strategy_data['bias_reduction_percentage'], 
                                strategy_data['bleu_4_score'],
                                c=color, label=label, alpha=0.7, s=60,
                                edgecolors='white', linewidth=0.5)
            
            # Add strategy median markers
            median_bias = strategy_data['bias_reduction_percentage'].median()
            median_bleu = strategy_data['bleu_4_score'].median()
            ax3.scatter(median_bias, median_bleu, c=color, s=200, marker='D', 
                       edgecolors='black', linewidth=2, alpha=0.9,
                       label=f'{label} Median')
        
        # Add overall median lines
        ax3.axvline(x=median_bias_overall, color='gray', linestyle='--', alpha=0.6, 
                   label=f'Overall Median Bias: {median_bias_overall:.1f}%')
        ax3.axhline(y=median_bleu_overall, color='gray', linestyle='--', alpha=0.6,
                   label=f'Overall Median BLEU: {median_bleu_overall:.3f}')
        
        ax3.set_xlabel('Bias Reduction (%)', fontsize=14, fontweight='bold')
        ax3.set_ylabel('BLEU-4 Score (Semantic Preservation)', fontsize=14, fontweight='bold')
        ax3.set_title('Bias Reduction vs Semantic Preservation', fontsize=16, fontweight='bold', pad=20)
        ax3.grid(True, alpha=0.3)
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        # 4. Strategy Performance Summary (Bottom row)
        ax4 = fig.add_subplot(gs[2, :])
        
        # Create performance comparison
        strategy_stats = []
        for strategy, label in zip(strategies, strategy_labels):
            data = self.df[self.df['strategy'] == strategy]
            stats = {
                'Strategy': label,
                'Mean Bias Reduction': f"{data['bias_reduction_percentage'].mean():.1f}%",
                'Median Bias Reduction': f"{data['bias_reduction_percentage'].median():.1f}%",
                'Mean BLEU-4': f"{data['bleu_4_score'].mean():.3f}",
                'Success Rate': f"{(data['bias_reduction_percentage'] > 50).mean()*100:.0f}%"
            }
            strategy_stats.append(stats)
        
        # Create table
        df_table = pd.DataFrame(strategy_stats)
        ax4.axis('tight')
        ax4.axis('off')
        
        table = ax4.table(cellText=df_table.values, colLabels=df_table.columns,
                         cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(df_table.columns)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color code the best performers
        for i in range(1, len(strategy_stats) + 1):
            if 'Few-Shot + Verification' in df_table.iloc[i-1]['Strategy']:
                for j in range(len(df_table.columns)):
                    table[(i, j)].set_facecolor('#d4edda')
        
        plt.tight_layout()
        
        file_path = self.output_dir / "anova_statistical_analysis.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"{Fore.GREEN}✓ Created simplified ANOVA visualization: {file_path}{Style.RESET_ALL}")
        return str(file_path)
    
    def create_interactive_dashboard(self) -> str:
        """Create comprehensive interactive dashboard"""
        print(f"{Fore.CYAN}Creating interactive dashboard...{Style.RESET_ALL}")
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Bias Reduction by Strategy", "Strategy Performance Radar", 
                           "Repetition Analysis", "Bias vs Semantic Preservation"),
            specs=[[{"type": "box"}, {"type": "scatterpolar"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        strategies = ['raw', 'system', 'few_shot', 'few_shot_verification']
        strategy_labels = ['Raw', 'System', 'Few-Shot', 'FS + Verification']
        colors = ['#e74c3c', '#f39c12', '#3498db', '#27ae60']
        
        # 1. Box plot for bias reduction
        for i, (strategy, color, label) in enumerate(zip(strategies, colors, strategy_labels)):
            strategy_data = self.df[self.df['strategy'] == strategy]['bias_reduction_percentage']
            fig.add_trace(
                go.Box(y=strategy_data, name=label, marker_color=color,
                      boxpoints='all', jitter=0.3, pointpos=-1.8),
                row=1, col=1
            )
        
        # 2. Radar chart for overall performance
        metrics = ['bias_reduction_percentage', 'bleu_4_score', 'neutralization_success']
        metric_labels = ['Bias Reduction', 'BLEU-4', 'Neutralization']
        
        for i, (strategy, color, label) in enumerate(zip(strategies, colors, strategy_labels)):
            strategy_data = self.df[self.df['strategy'] == strategy]
            
            # Normalize metrics to 0-1 scale for radar chart
            values = []
            for metric in metrics:
                if metric == 'neutralization_success':
                    val = strategy_data['is_gender_neutral'].mean()
                else:
                    val = strategy_data[metric].mean()
                
                # Normalize based on metric type
                if metric == 'bias_reduction_percentage':
                    val = val / 100  # Already percentage
                values.append(val)
            
            fig.add_trace(
                go.Scatterpolar(
                    r=values + [values[0]],  # Close the polygon
                    theta=metric_labels + [metric_labels[0]],
                    fill='toself',
                    name=label,
                    line_color=color,
                    opacity=0.6
                ),
                row=1, col=2
            )
        
        # 3. Repetition consistency scatter
        for i, (strategy, color, label) in enumerate(zip(strategies, colors, strategy_labels)):
            strategy_data = self.df[self.df['strategy'] == strategy]
            
            # Calculate mean and std for each paragraph
            paragraph_stats = strategy_data.groupby('paragraph_id')['bias_reduction_percentage'].agg(['mean', 'std']).reset_index()
            
            fig.add_trace(
                go.Scatter(
                    x=paragraph_stats['mean'],
                    y=paragraph_stats['std'],
                    mode='markers',
                    name=f'{label} Consistency',
                    marker=dict(color=color, size=8, opacity=0.7),
                    text=[f"Paragraph {pid}<br>Mean: {mean:.1f}%<br>Std: {std:.2f}" 
                          for pid, mean, std in zip(paragraph_stats['paragraph_id'], 
                                                   paragraph_stats['mean'], 
                                                   paragraph_stats['std'])],
                    hovertemplate='<b>%{text}</b><extra></extra>'
                ),
                row=2, col=1
            )
        
        # 4. Bias reduction vs BLEU-4
        for i, (strategy, color, label) in enumerate(zip(strategies, colors, strategy_labels)):
            strategy_data = self.df[self.df['strategy'] == strategy]
            
            fig.add_trace(
                go.Scatter(
                    x=strategy_data['bias_reduction_percentage'],
                    y=strategy_data['bleu_4_score'],
                    mode='markers',
                    name=f'{label} Quality',
                    marker=dict(color=color, size=6, opacity=0.7),
                    text=[f"ID: {eid}<br>Bias Reduction: {br:.1f}%<br>BLEU-4: {bleu:.3f}" 
                          for eid, br, bleu in zip(strategy_data['experiment_id'], 
                                                  strategy_data['bias_reduction_percentage'], 
                                                  strategy_data['bleu_4_score'])],
                    hovertemplate='<b>%{text}</b><extra></extra>'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Gender Bias Analysis - Interactive Dashboard",
            title_font_size=20,
            height=800,
            showlegend=True,
            hovermode='closest'
        )
        
        # Update axes
        fig.update_xaxes(title_text="Bias Reduction (%)", row=2, col=1)
        fig.update_yaxes(title_text="Standard Deviation", row=2, col=1)
        fig.update_xaxes(title_text="Bias Reduction (%)", row=2, col=2)
        fig.update_yaxes(title_text="BLEU-4 Score", row=2, col=2)
        
        file_path = self.output_dir / "interactive_comprehensive_dashboard.html"
        fig.write_html(str(file_path))
        
        print(f"{Fore.GREEN}✓ Created interactive dashboard: {file_path}{Style.RESET_ALL}")
        return str(file_path)
    
    def create_all_improved_visualizations(self, anova_results: Dict[str, Any]) -> List[str]:
        """Create all improved visualizations"""
        saved_files = []
        
        print(f"{Fore.CYAN}Creating improved visualization suite...{Style.RESET_ALL}")
        
        # 1. Main bias reduction plot (the star)
        saved_files.append(self.create_bias_reduction_masterplot())
        
        # 2. ANOVA statistical analysis
        saved_files.append(self.create_anova_visualization(anova_results))
        
        # 3. Interactive dashboard
        saved_files.append(self.create_interactive_dashboard())
        
        print(f"{Fore.GREEN}✓ Created {len(saved_files)} improved visualizations{Style.RESET_ALL}")
        return saved_files

def run_improved_analysis(results_file: Path) -> Dict[str, Any]:
    """Run complete improved analysis pipeline"""
    print(f"{Fore.CYAN}=== IMPROVED GENDER BIAS ANALYSIS ==={Style.RESET_ALL}")
    
    # Initialize analyzer
    analyzer = ImprovedAnalyzer()
    
    # Load data
    if not analyzer.load_and_prepare_data(results_file):
        return {"error": "Failed to load data"}
    
    # Perform comprehensive ANOVA
    anova_results = analyzer.perform_comprehensive_anova()
    
    # Create visualizations
    visualizer = ImprovedVisualizer(analyzer.df)
    visualization_files = visualizer.create_all_improved_visualizations(anova_results)
    
    # Create summary report
    summary = {
        "analysis_timestamp": datetime.now().isoformat(),
        "data_summary": {
            "total_experiments": len(analyzer.df),
            "strategies": list(analyzer.df['strategy'].unique()),
            "models": list(analyzer.df['model'].unique()),
            "repetitions": analyzer.df['repetition'].nunique()
        },
        "anova_results": anova_results,
        "visualization_files": visualization_files,
        "key_findings": {
            "best_strategy": anova_results["bias_reduction_percentage"]["summary"]["best_strategy"],
            "best_bias_reduction": float(analyzer.df.groupby('strategy')['bias_reduction_percentage'].mean().max()),
            "statistical_significance": all(anova_results[metric]["anova"]["significant"] 
                                          for metric in ["bias_reduction_percentage", "bleu_4_score"]),
            "effect_sizes": {metric: anova_results[metric]["anova"]["effect_size"] 
                           for metric in ["bias_reduction_percentage", "bleu_4_score", "neutralization_success"]}
        }
    }
    
    # Save analysis results
    analysis_file = RESULTS_DIR / f"improved_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_json(summary, analysis_file)
    
    print(f"{Fore.GREEN}✓ Improved analysis complete! Results saved to {analysis_file}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}✓ Visualizations saved to: {visualizer.output_dir}{Style.RESET_ALL}")
    
    return summary
