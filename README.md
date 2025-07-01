# Gender Bias in Large Language Models Study

A comprehensive research framework for investigating how different prompting strategies affect gender bias in Large Language Model outputs.

## 🎯 Study Overview

This project implements a controlled experiment to evaluate four different prompting strategies:

1. **Raw Prompt** (Control) - Basic rewrite request
2. **System Prompt** - Explicit gender-neutral instructions  
3. **Few-Shot** - Examples + instructions
4. **Few-Shot + Verification** - Examples + self-verification

### Evaluation Metrics

- **Gender Bias Score** - Automated detection and quantification of gendered terms
- **Fluency Score** - Text quality and readability assessment
- **BLEU-4 Score** - Meaning preservation compared to original text
- **Semantic Similarity** - Content preservation analysis

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project
cd Gender-Bias

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup Environment

```bash
# Setup project and create templates
python main.py setup

# Copy the example environment file
cp .env.example .env

# Edit .env and add your API keys
# OPENAI_API_KEY=your_openai_key_here
# GEMINI_API_KEY=your_gemini_key_here
```

### 3. Prepare Your Corpus

1. Open `data/corpus/corpus_template.csv`
2. Replace the template data with your 25 paragraphs
3. Ensure each paragraph has gendered terms
4. Save as `data/corpus/paragraphs.csv`

### 4. Run the Experiment

```bash
# Test all components first
python main.py test

# Run the full experiment (takes 30-60 minutes)
python main.py run-experiment

# Analyze results and generate visualizations
python main.py analyze
```

## 📊 Features

### Comprehensive Experiment Framework
- **Multi-LLM Support**: OpenAI GPT and Google Gemini
- **Automated Pipeline**: End-to-end experiment execution
- **Resume Capability**: Robust handling of interruptions
- **Progress Tracking**: Real-time experiment monitoring

### Advanced Evaluation System
- **Automated Gender Bias Detection**: Regex-based pattern matching
- **Quality Assessment**: Fluency and meaning preservation
- **Statistical Analysis**: ANOVA and post-hoc tests
- **Interactive Visualizations**: 20+ different analysis plots

### Data Management
- **Structured Storage**: JSON and CSV output formats
- **Reproducible Results**: Complete experiment logging
- **Version Control Ready**: Proper .gitignore and structure

## 📁 Project Structure

```
Gender-Bias/
├── src/                     # Core source code
│   ├── config.py           # Configuration settings
│   ├── utils.py            # Utility functions
│   ├── data_processing.py  # Corpus handling
│   ├── llm_interface.py    # LLM API interfaces
│   ├── prompting.py        # Prompting strategies
│   ├── evaluation.py       # Evaluation metrics
│   ├── experiment_runner.py # Main experiment execution
│   └── analysis.py         # Statistical analysis & visualization
├── data/
│   ├── corpus/             # Input paragraphs
│   ├── outputs/            # Generated texts
│   └── results/            # Experiment results & analysis
├── notebooks/              # Jupyter notebooks (optional)
├── main.py                 # CLI entry point
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🛠️ Usage Examples

### Command Line Interface

```bash
# Setup and test
python main.py setup
python main.py test

# Run experiments
python main.py run-experiment
python main.py run-experiment --no-resume  # Start fresh

# Analysis
python main.py analyze
python main.py analyze --results-file path/to/specific/results.json
python main.py list-results
```

### Programmatic Usage

```python
from src import ExperimentRunner, ComprehensiveAnalyzer

# Run experiment
runner = ExperimentRunner()
success = runner.run_all_experiments()

# Analyze results
analyzer = ComprehensiveAnalyzer(runner.results_file)
analysis_results = analyzer.run_complete_analysis()
```

## 📈 Generated Visualizations

The analysis generates 20+ visualizations including:

1. **Strategy Comparison**
   - Box plots for each metric by strategy
   - Mean scores comparison
   - Success rate analysis

2. **Model Performance**
   - Cross-model comparisons
   - Performance consistency analysis

3. **Correlation Analysis**
   - Metric correlation heatmaps
   - Trade-off scatter plots

4. **Distribution Analysis**
   - Performance distributions
   - Strategy effectiveness breakdown

5. **Interactive Plots**
   - Plotly-based interactive visualizations
   - Drill-down analysis capabilities

6. **Summary Dashboard**
   - Comprehensive overview
   - Key findings highlight

## 🔧 Configuration

### Experiment Settings (`src/config.py`)

```python
EXPERIMENT_CONFIG = {
    "repetitions_per_paragraph": 3,
    "prompt_strategies": ["raw", "system", "few_shot", "few_shot_verification"],
    "llm_models": ["openai", "gemini"],
    "temperature": 0.7,
}
```

### Rate Limiting

```python
RATE_LIMITS = {
    "openai": 30,  # requests per minute
    "gemini": 60,
}
```

### Gendered Terms Detection

The system uses regex patterns to detect:
- Pronouns: he, she, him, her, his, hers
- General terms: man, woman, boy, girl, male, female
- Professional terms: actor/actress, waiter/waitress, etc.
- Family terms: father, mother, son, daughter, etc.

## 📊 Statistical Analysis

### ANOVA Testing
- Tests for significant differences between prompting strategies
- Separate analysis for each evaluation metric
- Effect size calculation (eta-squared)

### Post-hoc Analysis
- Pairwise comparisons between strategies
- Bonferroni correction for multiple comparisons
- Detailed significance reporting

### Descriptive Statistics
- Complete summary statistics by strategy and model
- Distribution analysis
- Performance consistency measures

## 🚨 Troubleshooting

### Common Issues

1. **API Key Errors**
   ```bash
   # Make sure .env file exists and has correct keys
   cp .env.example .env
   # Edit .env with your actual API keys
   ```

2. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Corpus Format Errors**
   - Ensure CSV has columns: id, paragraph, source
   - Check that paragraphs contain gendered terms
   - Verify CSV encoding (UTF-8 recommended)

4. **Rate Limiting Issues**
   - The system automatically handles rate limits
   - Adjust limits in `src/config.py` if needed

### Debug Mode

```bash
python main.py test --verbose
```

## 📝 Requirements

### Python Packages
- openai>=1.0.0
- google-generativeai>=0.3.0
- pandas>=2.0.0
- matplotlib>=3.7.0
- seaborn>=0.12.0
- plotly>=5.15.0
- scipy>=1.10.0
- nltk>=3.8.0
- And more (see requirements.txt)

### API Access
- OpenAI API key (GPT models)
- Google Gemini API key
- Optional: Additional LLM APIs for extended analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is intended for academic research purposes. Please cite appropriately if used in publications.

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the generated logs
3. Test individual components with `python main.py test`

## 🔬 Research Context

This framework is designed for academic research on gender bias in LLMs. It provides:

- **Methodological Rigor**: Controlled experimental design
- **Reproducibility**: Complete logging and version control
- **Statistical Validity**: Proper statistical testing procedures  
- **Comprehensive Analysis**: Multiple evaluation perspectives

Perfect for semester projects, thesis research, or academic publications investigating AI fairness and bias mitigation strategies.

---

**Ready to start?** Run `python main.py setup` to begin! 🚀
