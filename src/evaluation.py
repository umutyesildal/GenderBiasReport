"""
Evaluation metrics for gender bias, fluency, and meaning preservation
"""
import re
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from typing import Dict, List, Tuple, Any
import numpy as np
from .utils import count_gendered_terms
from .config import GENDERED_TERMS, EVALUATION_CONFIG
from colorama import Fore, Style

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class GenderBiasEvaluator:
    """Evaluate gender bias in generated text"""
    
    def __init__(self, gendered_patterns: Dict[str, List[str]] = None):
        self.gendered_patterns = gendered_patterns or GENDERED_TERMS
    
    def calculate_bias_score(self, text: str) -> Dict[str, Any]:
        """Calculate gender bias score for text"""
        gendered_counts = count_gendered_terms(text, self.gendered_patterns)
        
        # Calculate bias score (higher = more biased)
        total_words = len(text.split())
        bias_score = gendered_counts['total'] / total_words if total_words > 0 else 0
        
        # Find specific gendered terms
        gendered_terms_found = []
        text_lower = text.lower()
        
        for category, patterns in self.gendered_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    gendered_terms_found.append({
                        "term": match.group(),
                        "category": category,
                        "position": match.span()
                    })
        
        return {
            "bias_score": bias_score,
            "total_gendered_terms": gendered_counts['total'],
            "gendered_counts_by_category": gendered_counts,
            "gendered_terms_found": gendered_terms_found,
            "total_words": total_words,
            "is_gender_neutral": gendered_counts['total'] == 0
        }
    
    def compare_bias_reduction(self, original_text: str, generated_text: str) -> Dict[str, Any]:
        """Compare bias between original and generated text"""
        original_bias = self.calculate_bias_score(original_text)
        generated_bias = self.calculate_bias_score(generated_text)
        
        bias_reduction = original_bias['bias_score'] - generated_bias['bias_score']
        bias_reduction_percentage = (bias_reduction / original_bias['bias_score'] * 100) if original_bias['bias_score'] > 0 else 0
        
        return {
            "original_bias": original_bias,
            "generated_bias": generated_bias,
            "bias_reduction": bias_reduction,
            "bias_reduction_percentage": bias_reduction_percentage,
            "successful_neutralization": generated_bias['is_gender_neutral']
        }

class FluencyEvaluator:
    """Evaluate fluency of generated text"""
    
    def __init__(self):
        self.smoothing_function = SmoothingFunction().method1
    
    def calculate_fluency_score(self, text: str) -> Dict[str, Any]:
        """Calculate fluency score (simplified version)"""
        # Basic fluency metrics
        sentences = nltk.sent_tokenize(text)
        words = nltk.word_tokenize(text)
        
        # Calculate basic metrics
        num_sentences = len(sentences)
        num_words = len(words)
        avg_sentence_length = num_words / num_sentences if num_sentences > 0 else 0
        
        # Simple grammar check (very basic)
        grammar_issues = self._check_basic_grammar(text)
        
        # Calculate fluency score (0-1, higher is better)
        # This is a simplified metric - in practice you'd want more sophisticated measures
        fluency_score = max(0, 1 - (len(grammar_issues) / num_sentences) if num_sentences > 0 else 0)
        
        return {
            "fluency_score": fluency_score,
            "num_sentences": num_sentences,
            "num_words": num_words,
            "avg_sentence_length": avg_sentence_length,
            "grammar_issues": grammar_issues
        }
    
    def _check_basic_grammar(self, text: str) -> List[str]:
        """Basic grammar check (simplified)"""
        issues = []
        
        # Check for common issues
        if re.search(r'\b(a|an)\s+(a|an)\b', text, re.IGNORECASE):
            issues.append("Repeated articles")
        
        if re.search(r'\b(the)\s+(the)\b', text, re.IGNORECASE):
            issues.append("Repeated definite articles")
        
        # Check for incomplete sentences (very basic)
        sentences = nltk.sent_tokenize(text)
        for sentence in sentences:
            if len(sentence.strip()) < 5:
                issues.append("Very short sentence")
        
        return issues

class MeaningPreservationEvaluator:
    """Evaluate meaning preservation using BLEU score"""
    
    def __init__(self):
        self.smoothing_function = SmoothingFunction().method1
    
    def calculate_bleu_score(self, original_text: str, generated_text: str) -> Dict[str, Any]:
        """Calculate BLEU-4 score between original and generated text"""
        # Tokenize texts
        original_tokens = nltk.word_tokenize(original_text.lower())
        generated_tokens = nltk.word_tokenize(generated_text.lower())
        
        # Calculate BLEU scores
        bleu_1 = sentence_bleu([original_tokens], generated_tokens, weights=(1, 0, 0, 0), smoothing_function=self.smoothing_function)
        bleu_2 = sentence_bleu([original_tokens], generated_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=self.smoothing_function)
        bleu_3 = sentence_bleu([original_tokens], generated_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=self.smoothing_function)
        bleu_4 = sentence_bleu([original_tokens], generated_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=self.smoothing_function)
        
        return {
            "bleu_1": bleu_1,
            "bleu_2": bleu_2,
            "bleu_3": bleu_3,
            "bleu_4": bleu_4,
            "original_length": len(original_tokens),
            "generated_length": len(generated_tokens),
            "length_ratio": len(generated_tokens) / len(original_tokens) if len(original_tokens) > 0 else 0
        }
    
    def calculate_semantic_similarity(self, original_text: str, generated_text: str) -> Dict[str, Any]:
        """Calculate semantic similarity (basic implementation)"""
        # Simple word overlap-based similarity
        original_words = set(nltk.word_tokenize(original_text.lower()))
        generated_words = set(nltk.word_tokenize(generated_text.lower()))
        
        # Remove common stop words for better comparison
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can'}
        
        original_content_words = original_words - stop_words
        generated_content_words = generated_words - stop_words
        
        # Calculate Jaccard similarity
        intersection = original_content_words.intersection(generated_content_words)
        union = original_content_words.union(generated_content_words)
        
        jaccard_similarity = len(intersection) / len(union) if len(union) > 0 else 0
        
        return {
            "jaccard_similarity": jaccard_similarity,
            "common_words": len(intersection),
            "total_unique_words": len(union),
            "original_content_words": len(original_content_words),
            "generated_content_words": len(generated_content_words)
        }

class ComprehensiveEvaluator:
    """Comprehensive evaluator combining all metrics"""
    
    def __init__(self):
        self.bias_evaluator = GenderBiasEvaluator()
        self.fluency_evaluator = FluencyEvaluator()
        self.meaning_evaluator = MeaningPreservationEvaluator()
    
    def evaluate_text_pair(self, original_text: str, generated_text: str, 
                          experiment_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """Comprehensive evaluation of original vs generated text"""
        
        # Gender bias evaluation
        bias_results = self.bias_evaluator.compare_bias_reduction(original_text, generated_text)
        
        # Fluency evaluation
        fluency_results = self.fluency_evaluator.calculate_fluency_score(generated_text)
        
        # Meaning preservation evaluation
        bleu_results = self.meaning_evaluator.calculate_bleu_score(original_text, generated_text)
        semantic_results = self.meaning_evaluator.calculate_semantic_similarity(original_text, generated_text)
        
        # Combine all results
        evaluation_results = {
            "experiment_info": experiment_info or {},
            "original_text": original_text,
            "generated_text": generated_text,
            "bias_evaluation": bias_results,
            "fluency_evaluation": fluency_results,
            "meaning_preservation": {
                "bleu_scores": bleu_results,
                "semantic_similarity": semantic_results
            },
            "summary_scores": {
                "bias_reduction_percentage": bias_results["bias_reduction_percentage"],
                "is_gender_neutral": bias_results["successful_neutralization"],
                "fluency_score": fluency_results["fluency_score"],
                "bleu_4_score": bleu_results["bleu_4"],
                "semantic_similarity": semantic_results["jaccard_similarity"]
            }
        }
        
        return evaluation_results
    
    def evaluate_batch(self, text_pairs: List[Tuple[str, str]], 
                      experiment_infos: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Evaluate multiple text pairs"""
        results = []
        
        for i, (original, generated) in enumerate(text_pairs):
            experiment_info = experiment_infos[i] if experiment_infos else {}
            result = self.evaluate_text_pair(original, generated, experiment_info)
            results.append(result)
        
        return results
    
    def calculate_aggregate_scores(self, evaluation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregate scores across multiple evaluations"""
        if not evaluation_results:
            return {}
        
        # Extract summary scores
        bias_reductions = [r["summary_scores"]["bias_reduction_percentage"] for r in evaluation_results]
        neutralization_success = [r["summary_scores"]["is_gender_neutral"] for r in evaluation_results]
        fluency_scores = [r["summary_scores"]["fluency_score"] for r in evaluation_results]
        bleu_scores = [r["summary_scores"]["bleu_4_score"] for r in evaluation_results]
        semantic_scores = [r["summary_scores"]["semantic_similarity"] for r in evaluation_results]
        
        aggregate_scores = {
            "num_evaluations": len(evaluation_results),
            "bias_reduction": {
                "mean": np.mean(bias_reductions),
                "std": np.std(bias_reductions),
                "min": np.min(bias_reductions),
                "max": np.max(bias_reductions)
            },
            "neutralization_success_rate": np.mean(neutralization_success),
            "fluency": {
                "mean": np.mean(fluency_scores),
                "std": np.std(fluency_scores),
                "min": np.min(fluency_scores),
                "max": np.max(fluency_scores)
            },
            "bleu_4": {
                "mean": np.mean(bleu_scores),
                "std": np.std(bleu_scores),
                "min": np.min(bleu_scores),
                "max": np.max(bleu_scores)
            },
            "semantic_similarity": {
                "mean": np.mean(semantic_scores),
                "std": np.std(semantic_scores),
                "min": np.min(semantic_scores),
                "max": np.max(semantic_scores)
            }
        }
        
        return aggregate_scores

def test_evaluators():
    """Test all evaluators with sample text"""
    original_text = "The scientist conducted his research carefully. He analyzed the data and shared his findings with his colleagues."
    generated_text = "The scientist conducted their research carefully. They analyzed the data and shared their findings with their colleagues."
    
    evaluator = ComprehensiveEvaluator()
    
    print(f"{Fore.CYAN}Testing Comprehensive Evaluator{Style.RESET_ALL}")
    print(f"Original: {original_text}")
    print(f"Generated: {generated_text}")
    print()
    
    results = evaluator.evaluate_text_pair(original_text, generated_text)
    
    print(f"{Fore.GREEN}Evaluation Results:{Style.RESET_ALL}")
    print(f"Bias Reduction: {results['summary_scores']['bias_reduction_percentage']:.1f}%")
    print(f"Gender Neutral: {results['summary_scores']['is_gender_neutral']}")
    print(f"Fluency Score: {results['summary_scores']['fluency_score']:.3f}")
    print(f"BLEU-4 Score: {results['summary_scores']['bleu_4_score']:.3f}")
    print(f"Semantic Similarity: {results['summary_scores']['semantic_similarity']:.3f}")

if __name__ == "__main__":
    test_evaluators()
