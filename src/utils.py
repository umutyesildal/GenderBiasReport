"""
Utility functions for the Gender Bias study
"""
import os
import json
import re
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import pandas as pd
from colorama import init, Fore, Style

# Initialize colorama for colored terminal output
init(autoreset=True)

# Load environment variables
load_dotenv()

def setup_logging(log_level=logging.INFO):
    """Setup logging configuration"""
    from .config import LOGS_DIR
    
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file = LOGS_DIR / f"gender_bias_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def get_api_key(service: str) -> Optional[str]:
    """Get API key for a specific service"""
    key_mapping = {
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "huggingface": "HUGGINGFACE_API_KEY",
        "groq": "GROQ_API_KEY"
    }
    
    env_var = key_mapping.get(service.lower())
    if not env_var:
        raise ValueError(f"Unknown service: {service}")
    
    api_key = os.getenv(env_var)
    if not api_key:
        print(f"{Fore.YELLOW}Warning: No API key found for {service}. Set {env_var} in your .env file.{Style.RESET_ALL}")
    
    return api_key

def validate_corpus_file(file_path: Path) -> bool:
    """Validate that corpus file exists and has correct format"""
    if not file_path.exists():
        print(f"{Fore.RED}Error: Corpus file not found: {file_path}{Style.RESET_ALL}")
        return False
    
    try:
        df = pd.read_csv(file_path)
        required_columns = ['id', 'paragraph', 'source']
        
        if not all(col in df.columns for col in required_columns):
            print(f"{Fore.RED}Error: Corpus file must have columns: {required_columns}{Style.RESET_ALL}")
            return False
            
        print(f"{Fore.GREEN}✓ Corpus file validated: {len(df)} paragraphs found{Style.RESET_ALL}")
        return True
        
    except Exception as e:
        print(f"{Fore.RED}Error reading corpus file: {e}{Style.RESET_ALL}")
        return False

def save_json(data: Any, file_path: Path, indent: int = 2):
    """Save data to JSON file"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"{Fore.RED}Error saving JSON file {file_path}: {e}{Style.RESET_ALL}")
        return False

def load_json(file_path: Path) -> Optional[Any]:
    """Load data from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"{Fore.RED}Error loading JSON file {file_path}: {e}{Style.RESET_ALL}")
        return None

def count_gendered_terms(text: str, gendered_patterns: Dict[str, List[str]]) -> Dict[str, int]:
    """Count gendered terms in text using regex patterns"""
    counts = {}
    text_lower = text.lower()
    
    for category, patterns in gendered_patterns.items():
        category_count = 0
        for pattern in patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            category_count += len(matches)
        counts[category] = category_count
    
    counts['total'] = sum(counts.values())
    return counts

def create_progress_bar_simple(total: int, current: int, description: str = "") -> str:
    """Create a simple text-based progress bar"""
    bar_length = 50
    filled_length = int(bar_length * current // total)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    percent = round(100.0 * current / total, 1)
    return f"{description} |{bar}| {percent}% ({current}/{total})"

def print_progress(total: int, current: int, description: str = ""):
    """Print progress with colored output"""
    progress_bar = create_progress_bar_simple(total, current, description)
    if current == total:
        print(f"{Fore.GREEN}{progress_bar}{Style.RESET_ALL}")
    else:
        print(f"{Fore.CYAN}{progress_bar}{Style.RESET_ALL}")

def rate_limiter(last_request_time: float, min_interval: float) -> float:
    """Simple rate limiter - returns the time to wait"""
    current_time = time.time()
    time_passed = current_time - last_request_time
    
    if time_passed < min_interval:
        wait_time = min_interval - time_passed
        return wait_time
    
    return 0.0

def safe_filename(text: str, max_length: int = 50) -> str:
    """Create a safe filename from text"""
    # Remove or replace unsafe characters
    safe_text = re.sub(r'[<>:"/\\|?*]', '_', text)
    safe_text = re.sub(r'\s+', '_', safe_text)
    
    if len(safe_text) > max_length:
        safe_text = safe_text[:max_length]
    
    return safe_text.strip('_')

def format_time_elapsed(start_time: float) -> str:
    """Format elapsed time in human-readable format"""
    elapsed = time.time() - start_time
    
    if elapsed < 60:
        return f"{elapsed:.1f} seconds"
    elif elapsed < 3600:
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        return f"{minutes}m {seconds}s"
    else:
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        return f"{hours}h {minutes}m"

def validate_environment() -> bool:
    """Validate that the environment is properly set up"""
    issues = []
    
    # Check for .env file
    if not Path('.env').exists():
        issues.append("No .env file found. Copy .env.example to .env and add your API keys.")
    
    # Check for required API keys
    required_keys = ['OPENAI_API_KEY', 'GEMINI_API_KEY']
    for key in required_keys:
        if not os.getenv(key):
            issues.append(f"Missing {key} in .env file")
    
    if issues:
        print(f"{Fore.RED}Environment validation failed:{Style.RESET_ALL}")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    print(f"{Fore.GREEN}✓ Environment validation passed{Style.RESET_ALL}")
    return True

class ExperimentState:
    """Manage experiment state for resumability"""
    
    def __init__(self, state_file: Path):
        self.state_file = state_file
        self.state = self.load_state()
    
    def load_state(self) -> Dict:
        """Load experiment state from file"""
        if self.state_file.exists():
            return load_json(self.state_file) or {}
        return {
            "completed_experiments": [],
            "failed_experiments": [],
            "last_update": None,
            "total_experiments": 0,
            "completed_count": 0
        }
    
    def save_state(self):
        """Save current state to file"""
        self.state["last_update"] = datetime.now().isoformat()
        save_json(self.state, self.state_file)
    
    def is_experiment_completed(self, experiment_id: str) -> bool:
        """Check if an experiment is already completed"""
        return experiment_id in self.state.get("completed_experiments", [])
    
    def mark_experiment_completed(self, experiment_id: str):
        """Mark an experiment as completed"""
        if experiment_id not in self.state.get("completed_experiments", []):
            self.state.setdefault("completed_experiments", []).append(experiment_id)
            self.state["completed_count"] = len(self.state["completed_experiments"])
            self.save_state()
    
    def mark_experiment_failed(self, experiment_id: str, error: str):
        """Mark an experiment as failed"""
        self.state.setdefault("failed_experiments", []).append({
            "experiment_id": experiment_id,
            "error": error,
            "timestamp": datetime.now().isoformat()
        })
        self.save_state()
    
    def get_progress(self) -> Dict:
        """Get current progress statistics"""
        completed = len(self.state.get("completed_experiments", []))
        failed = len(self.state.get("failed_experiments", []))
        total = self.state.get("total_experiments", 0)
        
        return {
            "completed": completed,
            "failed": failed,
            "remaining": max(0, total - completed - failed),
            "total": total,
            "completion_rate": (completed / total * 100) if total > 0 else 0
        }
