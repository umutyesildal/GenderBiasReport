"""
Data export and visualization utilities for Gender Bias study results
"""
import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from .utils import load_json, save_json
from .evaluation import clean_verification_text
from colorama import Fore, Style

class ResultsExporter:
    """Export and visualize experiment results"""
    
    def __init__(self, results_file: Path):
        self.results_file = results_file
        self.results_data = load_json(results_file)
        self.output_dir = results_file.parent
    
    def export_paragraph_comparisons(self) -> str:
        """Export clean JSON with original vs generated text comparisons"""
        if not self.results_data:
            return None
        
        comparisons = []
        
        for result in self.results_data.get("detailed_results", []):
            # Clean verification text from generated content for display
            original_text = result["evaluation"]["original_text"]
            generated_text = result["evaluation"]["generated_text"]
            cleaned_generated_text = clean_verification_text(generated_text)
            
            comparison = {
                "experiment_id": result["experiment_id"],
                "paragraph_id": result["paragraph_id"],
                "strategy": result["strategy"],
                "model": result["model"],
                "original_text": original_text,
                "generated_text": cleaned_generated_text,  # Use cleaned text
                "generated_text_with_verification": generated_text,  # Keep full text for reference
                "bias_scores": {
                    "original_bias_score": result["evaluation"]["bias_evaluation"]["original_bias"]["bias_score"],
                    "generated_bias_score": result["evaluation"]["bias_evaluation"]["generated_bias"]["bias_score"],
                    "bias_reduction_percentage": result["evaluation"]["bias_evaluation"]["bias_reduction_percentage"],
                    "gender_neutral": result["evaluation"]["bias_evaluation"]["successful_neutralization"]
                },
                "quality_scores": {
                    "fluency_score": result["evaluation"]["fluency_evaluation"]["fluency_score"],
                    "bleu_4_score": result["evaluation"]["meaning_preservation"]["bleu_scores"]["bleu_4"],
                    "semantic_similarity": result["evaluation"]["meaning_preservation"]["semantic_similarity"]["jaccard_similarity"]
                },
                "generation_time": result["generation_time"],
                "tokens_used": result["generation_result"]["tokens_used"]
            }
            comparisons.append(comparison)
        
        # Save clean comparison data
        export_file = self.output_dir / f"paragraph_comparisons_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_json(comparisons, export_file)
        
        print(f"{Fore.GREEN}✓ Exported paragraph comparisons to: {export_file}{Style.RESET_ALL}")
        return str(export_file)
    
    def create_html_viewer(self) -> str:
        """Create an HTML viewer for all generated paragraphs"""
        if not self.results_data:
            return None
        
        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gender Bias Study Results Viewer</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            padding: 30px;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
        }
        .filters {
            margin-bottom: 30px;
            padding: 20px;
            background: #ecf0f1;
            border-radius: 8px;
        }
        .filter-group {
            display: inline-block;
            margin-right: 20px;
        }
        select, input {
            padding: 8px 12px;
            border: 1px solid #bdc3c7;
            border-radius: 4px;
            font-size: 14px;
        }
        .experiment {
            border: 1px solid #e1e8ed;
            border-radius: 8px;
            margin-bottom: 20px;
            padding: 20px;
            background: #fdfdfd;
        }
        .experiment-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #3498db;
        }
        .experiment-id {
            font-weight: bold;
            color: #2c3e50;
            font-size: 16px;
        }
        .strategy-badge {
            padding: 4px 12px;
            border-radius: 20px;
            color: white;
            font-size: 12px;
            font-weight: bold;
        }
        .strategy-raw { background-color: #e74c3c; }
        .strategy-system { background-color: #27ae60; }
        .strategy-few-shot { background-color: #3498db; }
        .strategy-few-shot-verification { background-color: #9b59b6; }
        
        .text-comparison {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        .text-block {
            padding: 15px;
            border-radius: 6px;
            line-height: 1.6;
        }
        .original-text {
            background-color: #fff5f5;
            border-left: 4px solid #e74c3c;
        }
        .generated-text {
            background-color: #f0fff4;
            border-left: 4px solid #27ae60;
        }
        .text-label {
            font-weight: bold;
            margin-bottom: 10px;
            text-transform: uppercase;
            font-size: 12px;
            letter-spacing: 0.5px;
        }
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 6px;
        }
        .metric {
            text-align: center;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }
        .metric-label {
            font-size: 12px;
            color: #7f8c8d;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .bias-reduction {
            color: #27ae60;
        }
        .bias-increase {
            color: #e74c3c;
        }
        .neutral-badge {
            background: #27ae60;
            color: white;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 10px;
            margin-left: 10px;
        }
        .summary {
            background: #34495e;
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }
        .summary h2 {
            margin-top: 0;
        }
        .summary-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Gender Bias Study Results Viewer</h1>
        
        <div class="summary">
            <h2>Experiment Summary</h2>
            <div class="summary-stats">
                <div>
                    <div class="metric-value">""" + str(self.results_data.get("total_experiments", 0)) + """</div>
                    <div class="metric-label">Total Experiments</div>
                </div>
                <div>
                    <div class="metric-value">""" + str(len(set(r["strategy"] for r in self.results_data.get("detailed_results", [])))) + """</div>
                    <div class="metric-label">Strategies Tested</div>
                </div>
                <div>
                    <div class="metric-value">""" + str(len(set(r["paragraph_id"] for r in self.results_data.get("detailed_results", [])))) + """</div>
                    <div class="metric-label">Paragraphs Processed</div>
                </div>
            </div>
        </div>
        
        <div class="filters">
            <div class="filter-group">
                <label>Strategy:</label>
                <select id="strategyFilter">
                    <option value="">All Strategies</option>
                    <option value="raw">Raw</option>
                    <option value="system">System</option>
                    <option value="few_shot">Few-shot</option>
                    <option value="few_shot_verification">Few-shot + Verification</option>
                </select>
            </div>
            <div class="filter-group">
                <label>Gender Neutral Only:</label>
                <input type="checkbox" id="neutralFilter">
            </div>
            <div class="filter-group">
                <label>Min Bias Reduction (%):</label>
                <input type="number" id="biasFilter" min="0" max="100" placeholder="0">
            </div>
        </div>
        
        <div id="experiments">
"""
        
        # Add each experiment
        for result in self.results_data.get("detailed_results", []):
            bias_eval = result["evaluation"]["bias_evaluation"]
            quality_eval = result["evaluation"]
            
            # Clean verification text for display
            original_text = result['evaluation']['original_text']
            generated_text = result['evaluation']['generated_text']
            cleaned_generated_text = clean_verification_text(generated_text)
            
            bias_reduction = bias_eval["bias_reduction_percentage"]
            is_neutral = bias_eval["successful_neutralization"]
            
            html_content += f"""
            <div class="experiment" data-strategy="{result['strategy']}" data-neutral="{str(is_neutral).lower()}" data-bias-reduction="{bias_reduction}">
                <div class="experiment-header">
                    <div class="experiment-id">{result['experiment_id']}</div>
                    <div>
                        <span class="strategy-badge strategy-{result['strategy'].replace('_', '-')}">{result['strategy'].replace('_', ' ').title()}</span>
                        {f'<span class="neutral-badge">Gender Neutral</span>' if is_neutral else ''}
                    </div>
                </div>
                
                <div class="text-comparison">
                    <div class="text-block original-text">
                        <div class="text-label">Original Text</div>
                        {original_text}
                    </div>
                    <div class="text-block generated-text">
                        <div class="text-label">Generated Text (Cleaned)</div>
                        {cleaned_generated_text}
                    </div>
                </div>
                
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-value {'bias-reduction' if bias_reduction >= 0 else 'bias-increase'}">{bias_reduction:.1f}%</div>
                        <div class="metric-label">Bias Reduction</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{quality_eval['fluency_evaluation']['fluency_score']:.3f}</div>
                        <div class="metric-label">Fluency</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{quality_eval['meaning_preservation']['bleu_scores']['bleu_4']:.3f}</div>
                        <div class="metric-label">BLEU-4</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{quality_eval['meaning_preservation']['semantic_similarity']['jaccard_similarity']:.3f}</div>
                        <div class="metric-label">Semantic Similarity</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{result['generation_time']:.1f}s</div>
                        <div class="metric-label">Generation Time</div>
                    </div>
                </div>
            </div>
            """
        
        html_content += """
        </div>
    </div>

    <script>
        function filterExperiments() {
            const strategyFilter = document.getElementById('strategyFilter').value;
            const neutralFilter = document.getElementById('neutralFilter').checked;
            const biasFilter = parseFloat(document.getElementById('biasFilter').value) || 0;
            
            const experiments = document.querySelectorAll('.experiment');
            
            experiments.forEach(experiment => {
                let show = true;
                
                if (strategyFilter && experiment.dataset.strategy !== strategyFilter) {
                    show = false;
                }
                
                if (neutralFilter && experiment.dataset.neutral !== 'true') {
                    show = false;
                }
                
                if (parseFloat(experiment.dataset.biasReduction) < biasFilter) {
                    show = false;
                }
                
                experiment.style.display = show ? 'block' : 'none';
            });
        }
        
        document.getElementById('strategyFilter').addEventListener('change', filterExperiments);
        document.getElementById('neutralFilter').addEventListener('change', filterExperiments);
        document.getElementById('biasFilter').addEventListener('input', filterExperiments);
    </script>
</body>
</html>
"""
        
        # Save HTML file
        html_file = self.output_dir / f"results_viewer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"{Fore.GREEN}✓ Created HTML viewer: {html_file}{Style.RESET_ALL}")
        return str(html_file)

def export_experiment_data(results_file: Path) -> Dict[str, str]:
    """Export experiment data in multiple formats"""
    exporter = ResultsExporter(results_file)
    
    exported_files = {}
    
    # Export clean JSON comparisons
    json_file = exporter.export_paragraph_comparisons()
    if json_file:
        exported_files["json_comparisons"] = json_file
    
    # Create HTML viewer
    html_file = exporter.create_html_viewer()
    if html_file:
        exported_files["html_viewer"] = html_file
    
    return exported_files
