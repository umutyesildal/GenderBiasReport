"""
Gender Bias in LLMs Study

A comprehensive study investigating how different prompting strategies affect 
gender bias in Large Language Model outputs.
"""

__version__ = "1.0.0"
__author__ = "Gender Bias Study Team"

from .config import EXPERIMENT_CONFIG
from .utils import validate_environment
from .data_processing import CorpusProcessor
from .llm_interface import LLMManager
from .prompting import PromptManager
from .evaluation import ComprehensiveEvaluator
from .experiment_runner import ExperimentRunner
from .analysis import ComprehensiveAnalyzer

__all__ = [
    "EXPERIMENT_CONFIG",
    "validate_environment", 
    "CorpusProcessor",
    "LLMManager",
    "PromptManager", 
    "ComprehensiveEvaluator",
    "ExperimentRunner",
    "ComprehensiveAnalyzer"
]
