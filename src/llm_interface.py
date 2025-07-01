"""
LLM interfaces for different AI models
"""
import time
import openai
import google.generativeai as genai
from typing import Dict, List, Optional, Any
from .utils import get_api_key, rate_limiter, setup_logging
from .config import RATE_LIMITS, EXPERIMENT_CONFIG
from colorama import Fore, Style

logger = setup_logging()

class BaseLLMInterface:
    """Base class for LLM interfaces"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.last_request_time = 0.0
        self.rate_limit = RATE_LIMITS.get(model_name, 30)  # Default 30 requests per minute
        self.min_interval = 60.0 / self.rate_limit  # Seconds between requests
    
    def _wait_for_rate_limit(self):
        """Wait if necessary to respect rate limits"""
        wait_time = rate_limiter(self.last_request_time, self.min_interval)
        if wait_time > 0:
            logger.debug(f"Rate limiting: waiting {wait_time:.1f} seconds")
            time.sleep(wait_time)
        self.last_request_time = time.time()
    
    def generate_text(self, prompt: str, system_message: Optional[str] = None) -> Dict[str, Any]:
        """Generate text using the LLM - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement generate_text method")

class OpenAIInterface(BaseLLMInterface):
    """Interface for OpenAI GPT models"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        super().__init__("openai")
        self.model_name = model_name
        
        # Initialize OpenAI client
        api_key = get_api_key("openai")
        if not api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
        
        self.client = openai.OpenAI(api_key=api_key)
        logger.info(f"Initialized OpenAI interface with model: {model_name}")
    
    def generate_text(self, prompt: str, system_message: Optional[str] = None) -> Dict[str, Any]:
        """Generate text using OpenAI GPT"""
        self._wait_for_rate_limit()
        
        try:
            messages = []
            
            if system_message:
                messages.append({"role": "system", "content": system_message})
            
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=EXPERIMENT_CONFIG["temperature"],
                max_tokens=1000,  # Sufficient for paragraph rewrites
            )
            
            generated_text = response.choices[0].message.content
            
            return {
                "generated_text": generated_text,
                "success": True,
                "model": self.model_name,
                "tokens_used": response.usage.total_tokens if response.usage else None,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return {
                "generated_text": None,
                "success": False,
                "model": self.model_name,
                "tokens_used": None,
                "error": str(e)
            }

class GeminiInterface(BaseLLMInterface):
    """Interface for Google Gemini models"""
    
    def __init__(self, model_name: str = "gemini-pro"):
        super().__init__("gemini")
        self.model_name = model_name
        
        # Initialize Gemini
        api_key = get_api_key("gemini")
        if not api_key:
            raise ValueError("Gemini API key not found. Please set GEMINI_API_KEY in your .env file.")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        logger.info(f"Initialized Gemini interface with model: {model_name}")
    
    def generate_text(self, prompt: str, system_message: Optional[str] = None) -> Dict[str, Any]:
        """Generate text using Google Gemini"""
        self._wait_for_rate_limit()
        
        try:
            # Combine system message with prompt if provided
            full_prompt = prompt
            if system_message:
                full_prompt = f"{system_message}\n\n{prompt}"
            
            response = self.model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=EXPERIMENT_CONFIG["temperature"],
                    max_output_tokens=1000,
                )
            )
            
            generated_text = response.text
            
            return {
                "generated_text": generated_text,
                "success": True,
                "model": self.model_name,
                "tokens_used": None,  # Gemini doesn't always provide token count
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return {
                "generated_text": None,
                "success": False,
                "model": self.model_name,
                "tokens_used": None,
                "error": str(e)
            }

class LLMManager:
    """Manage multiple LLM interfaces"""
    
    def __init__(self):
        self.interfaces = {}
        self.initialize_interfaces()
    
    def initialize_interfaces(self):
        """Initialize all available LLM interfaces"""
        # Try to initialize OpenAI
        try:
            self.interfaces["openai"] = OpenAIInterface()
            print(f"{Fore.GREEN}✓ OpenAI interface initialized{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.YELLOW}⚠ OpenAI interface failed to initialize: {e}{Style.RESET_ALL}")
        
        # Try to initialize Gemini
        try:
            self.interfaces["gemini"] = GeminiInterface()
            print(f"{Fore.GREEN}✓ Gemini interface initialized{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.YELLOW}⚠ Gemini interface failed to initialize: {e}{Style.RESET_ALL}")
        
        if not self.interfaces:
            raise RuntimeError("No LLM interfaces could be initialized. Please check your API keys.")
        
        print(f"{Fore.CYAN}Available LLM interfaces: {list(self.interfaces.keys())}{Style.RESET_ALL}")
    
    def get_interface(self, model_name: str) -> Optional[BaseLLMInterface]:
        """Get a specific LLM interface"""
        return self.interfaces.get(model_name)
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names"""
        return list(self.interfaces.keys())
    
    def generate_with_model(self, model_name: str, prompt: str, system_message: Optional[str] = None) -> Dict[str, Any]:
        """Generate text with a specific model"""
        interface = self.get_interface(model_name)
        if not interface:
            return {
                "generated_text": None,
                "success": False,
                "model": model_name,
                "tokens_used": None,
                "error": f"Model {model_name} not available"
            }
        
        return interface.generate_text(prompt, system_message)
    
    def test_all_interfaces(self) -> Dict[str, bool]:
        """Test all initialized interfaces with a simple prompt"""
        test_prompt = "Rewrite this sentence: The student submitted his assignment."
        results = {}
        
        print(f"{Fore.CYAN}Testing all LLM interfaces...{Style.RESET_ALL}")
        
        for model_name, interface in self.interfaces.items():
            try:
                result = interface.generate_text(test_prompt)
                results[model_name] = result["success"]
                
                if result["success"]:
                    print(f"{Fore.GREEN}✓ {model_name}: {result['generated_text'][:100]}...{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}✗ {model_name}: {result['error']}{Style.RESET_ALL}")
                    
            except Exception as e:
                results[model_name] = False
                print(f"{Fore.RED}✗ {model_name}: {e}{Style.RESET_ALL}")
        
        return results

if __name__ == "__main__":
    # Test the LLM interfaces
    try:
        manager = LLMManager()
        test_results = manager.test_all_interfaces()
        
        print(f"\n{Fore.CYAN}=== LLM INTERFACE TEST RESULTS ==={Style.RESET_ALL}")
        for model, success in test_results.items():
            status = "✓ Working" if success else "✗ Failed"
            color = Fore.GREEN if success else Fore.RED
            print(f"{color}{model}: {status}{Style.RESET_ALL}")
            
    except Exception as e:
        print(f"{Fore.RED}Error testing LLM interfaces: {e}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Make sure you have set up your .env file with API keys{Style.RESET_ALL}")
