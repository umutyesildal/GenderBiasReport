"""
Data processing and corpus validation for the Gender Bias study
"""
import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from .utils import count_gendered_terms, print_progress
from .config import GENDERED_TERMS, CORPUS_DIR
from colorama import Fore, Style

class CorpusProcessor:
    """Handle corpus loading, validation, and preprocessing"""
    
    def __init__(self, corpus_file: Optional[Path] = None):
        self.corpus_file = corpus_file or CORPUS_DIR / "paragraphs.csv"
        self.paragraphs = []
        self.gendered_patterns = GENDERED_TERMS
    
    def load_corpus(self) -> bool:
        """Load corpus from CSV file"""
        try:
            if not self.corpus_file.exists():
                print(f"{Fore.RED}Corpus file not found: {self.corpus_file}{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}Please create {self.corpus_file} with columns: id, paragraph, source{Style.RESET_ALL}")
                return False
            
            df = pd.read_csv(self.corpus_file)
            
            # Validate required columns
            required_columns = ['id', 'paragraph', 'source']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"{Fore.RED}Missing required columns: {missing_columns}{Style.RESET_ALL}")
                return False
            
            self.paragraphs = df.to_dict('records')
            print(f"{Fore.GREEN}✓ Loaded {len(self.paragraphs)} paragraphs from corpus{Style.RESET_ALL}")
            return True
            
        except Exception as e:
            print(f"{Fore.RED}Error loading corpus: {e}{Style.RESET_ALL}")
            return False
    
    def validate_paragraphs(self) -> Dict[str, any]:
        """Validate paragraphs for gendered terms and other criteria"""
        validation_results = {
            "total_paragraphs": len(self.paragraphs),
            "valid_paragraphs": 0,
            "paragraphs_with_gendered_terms": 0,
            "paragraphs_without_gendered_terms": [],
            "gendered_term_counts": {},
            "validation_details": []
        }
        
        print(f"{Fore.CYAN}Validating {len(self.paragraphs)} paragraphs...{Style.RESET_ALL}")
        
        for i, paragraph_data in enumerate(self.paragraphs):
            paragraph_id = paragraph_data['id']
            paragraph_text = paragraph_data['paragraph']
            
            # Count gendered terms
            gendered_counts = count_gendered_terms(paragraph_text, self.gendered_patterns)
            
            # Check if paragraph has gendered terms
            has_gendered_terms = gendered_counts['total'] > 0
            
            paragraph_validation = {
                "id": paragraph_id,
                "has_gendered_terms": has_gendered_terms,
                "gendered_counts": gendered_counts,
                "word_count": len(paragraph_text.split()),
                "character_count": len(paragraph_text)
            }
            
            validation_results["validation_details"].append(paragraph_validation)
            
            if has_gendered_terms:
                validation_results["paragraphs_with_gendered_terms"] += 1
                validation_results["valid_paragraphs"] += 1
            else:
                validation_results["paragraphs_without_gendered_terms"].append(paragraph_id)
            
            # Track gendered term counts
            for category, count in gendered_counts.items():
                if category not in validation_results["gendered_term_counts"]:
                    validation_results["gendered_term_counts"][category] = 0
                validation_results["gendered_term_counts"][category] += count
            
            # Progress update
            if (i + 1) % 5 == 0 or i == len(self.paragraphs) - 1:
                print_progress(len(self.paragraphs), i + 1, "Validating paragraphs")
        
        # Print validation summary
        self.print_validation_summary(validation_results)
        
        return validation_results
    
    def print_validation_summary(self, results: Dict):
        """Print a summary of validation results"""
        print(f"\n{Fore.CYAN}=== CORPUS VALIDATION SUMMARY ==={Style.RESET_ALL}")
        print(f"Total paragraphs: {results['total_paragraphs']}")
        print(f"Paragraphs with gendered terms: {results['paragraphs_with_gendered_terms']}")
        print(f"Paragraphs without gendered terms: {len(results['paragraphs_without_gendered_terms'])}")
        
        if results['paragraphs_without_gendered_terms']:
            print(f"{Fore.YELLOW}Warning: These paragraphs have no gendered terms:{Style.RESET_ALL}")
            for pid in results['paragraphs_without_gendered_terms']:
                print(f"  - {pid}")
        
        print(f"\n{Fore.CYAN}Gendered Term Distribution:{Style.RESET_ALL}")
        for category, count in results['gendered_term_counts'].items():
            if category != 'total':
                print(f"  {category}: {count} occurrences")
        print(f"  {Fore.GREEN}Total gendered terms: {results['gendered_term_counts'].get('total', 0)}{Style.RESET_ALL}")
    
    def get_paragraph_by_id(self, paragraph_id: str) -> Optional[Dict]:
        """Get a specific paragraph by ID"""
        for paragraph in self.paragraphs:
            if str(paragraph['id']) == str(paragraph_id):
                return paragraph
        return None
    
    def get_all_paragraphs(self) -> List[Dict]:
        """Get all loaded paragraphs"""
        return self.paragraphs
    
    def analyze_gendered_terms_in_text(self, text: str) -> Dict:
        """Analyze gendered terms in a specific text"""
        counts = count_gendered_terms(text, self.gendered_patterns)
        
        # Find specific matches for detailed analysis
        detailed_matches = {}
        text_lower = text.lower()
        
        for category, patterns in self.gendered_patterns.items():
            matches = []
            for pattern in patterns:
                found_matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in found_matches:
                    matches.append({
                        "term": match.group(),
                        "position": match.span(),
                        "pattern": pattern
                    })
            detailed_matches[category] = matches
        
        return {
            "counts": counts,
            "detailed_matches": detailed_matches,
            "text_length": len(text),
            "word_count": len(text.split())
        }
    
    def create_sample_corpus_file(self) -> bool:
        """Create a sample corpus file for testing"""
        sample_paragraphs = [
            {
                "id": "sample_1",
                "paragraph": "The scientist conducted his research carefully. He analyzed the data and shared his findings with his colleagues. The professor encouraged her students to think critically about the results.",
                "source": "Sample Science Textbook"
            },
            {
                "id": "sample_2", 
                "paragraph": "Every student must submit his or her assignment on time. The teacher will review each paper and provide feedback. She expects all students to demonstrate their understanding of the material.",
                "source": "Sample Education Manual"
            },
            {
                "id": "sample_3",
                "paragraph": "The businessman presented his proposal to the board. The chairwoman listened carefully and asked several questions. Each member of the committee must cast his or her vote.",
                "source": "Sample Business Text"
            }
        ]
        
        try:
            df = pd.DataFrame(sample_paragraphs)
            sample_file = CORPUS_DIR / "sample_paragraphs.csv"
            df.to_csv(sample_file, index=False)
            
            print(f"{Fore.GREEN}✓ Created sample corpus file: {sample_file}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}You can use this as a template for your actual corpus file{Style.RESET_ALL}")
            return True
            
        except Exception as e:
            print(f"{Fore.RED}Error creating sample corpus: {e}{Style.RESET_ALL}")
            return False

def create_corpus_template():
    """Create a template corpus file for users to fill in"""
    template_data = {
        "id": ["para_001", "para_002", "para_003", "..."],
        "paragraph": [
            "Your first paragraph with gendered terms here...",
            "Your second paragraph with gendered terms here...", 
            "Your third paragraph with gendered terms here...",
            "..."
        ],
        "source": [
            "Source textbook/paper name",
            "Source textbook/paper name",
            "Source textbook/paper name", 
            "..."
        ]
    }
    
    df = pd.DataFrame(template_data)
    template_file = CORPUS_DIR / "corpus_template.csv"
    df.to_csv(template_file, index=False)
    
    print(f"{Fore.GREEN}✓ Created corpus template: {template_file}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Fill in your 25 paragraphs in this file and rename to 'paragraphs.csv'{Style.RESET_ALL}")

if __name__ == "__main__":
    # Test the corpus processor
    processor = CorpusProcessor()
    
    # Create sample corpus if none exists
    if not processor.corpus_file.exists():
        processor.create_sample_corpus_file()
        create_corpus_template()
    
    # Load and validate corpus
    if processor.load_corpus():
        results = processor.validate_paragraphs()
