"""
Prompting strategies for gender bias evaluation
"""
from typing import Dict, Optional, Tuple
from .config import EXPERIMENT_CONFIG

class PromptStrategy:
    """Base class for prompting strategies"""
    
    def __init__(self, name: str):
        self.name = name
    
    def create_prompt(self, paragraph: str) -> Tuple[str, Optional[str]]:
        """Create prompt and system message for the given paragraph
        
        Returns:
            Tuple of (user_prompt, system_message)
        """
        raise NotImplementedError("Subclasses must implement create_prompt method")
    
    def get_description(self) -> str:
        """Get a description of this prompting strategy"""
        raise NotImplementedError("Subclasses must implement get_description method")

class RawPromptStrategy(PromptStrategy):
    """Raw prompt strategy - minimal instruction (control group)"""
    
    def __init__(self):
        super().__init__("raw")
    
    def create_prompt(self, paragraph: str) -> Tuple[str, Optional[str]]:
        user_prompt = f"Rewrite the following paragraph clearly:\n\n{paragraph}"
        return user_prompt, None
    
    def get_description(self) -> str:
        return "Basic rewrite request without any specific gender-neutral instructions (control group)"

class SystemPromptStrategy(PromptStrategy):
    """System prompt strategy - explicit system-level instruction"""
    
    def __init__(self):
        super().__init__("system")
    
    def create_prompt(self, paragraph: str) -> Tuple[str, Optional[str]]:
        system_message = "You are an inclusive writing assistant. Rewrite the following text using gender-neutral language."
        user_prompt = paragraph
        return user_prompt, system_message
    
    def get_description(self) -> str:
        return "Uses system-level instruction to encourage gender-neutral language"

class FewShotPromptStrategy(PromptStrategy):
    """Few-shot prompt strategy - examples of gender-neutral rewrites"""
    
    def __init__(self):
        super().__init__("few_shot")
    
    def create_prompt(self, paragraph: str) -> Tuple[str, Optional[str]]:
        system_message = "You are an inclusive writing assistant. Rewrite the text using gender-neutral language."
        
        user_prompt = f"""Here are two examples of gender-neutral rewrites:

Original: "Every student must submit his paper."
Neutral: "All students must submit their papers."

Original: "A professor should encourage his students."
Neutral: "Professors should encourage their students."

Now rewrite this paragraph clearly in gender-neutral language:

{paragraph}"""
        
        return user_prompt, system_message
    
    def get_description(self) -> str:
        return "Provides examples of gender-neutral rewrites before the main task"

class FewShotVerificationPromptStrategy(PromptStrategy):
    """Few-shot + verification strategy - examples plus self-verification"""
    
    def __init__(self):
        super().__init__("few_shot_verification")
    
    def create_prompt(self, paragraph: str) -> Tuple[str, Optional[str]]:
        system_message = "You are an inclusive writing assistant. Rewrite the text using gender-neutral language."
        
        user_prompt = f"""Here are two examples of gender-neutral rewrites:

Original: "Every student must submit his paper."
Neutral: "All students must submit their papers."

Original: "A professor should encourage his students."
Neutral: "Professors should encourage their students."

Now rewrite this paragraph clearly in gender-neutral language:

{paragraph}

After your initial rewrite, please verify: Are there still gendered terms (he/she/him/her/his/hers/man/woman, etc.) in your rewrite? If yes, rewrite again to be fully gender-neutral."""
        
        return user_prompt, system_message
    
    def get_description(self) -> str:
        return "Provides examples plus asks the model to self-verify and correct any remaining gendered terms"

class PromptManager:
    """Manage all prompting strategies"""
    
    def __init__(self):
        self.strategies = {
            "raw": RawPromptStrategy(),
            "system": SystemPromptStrategy(),
            "few_shot": FewShotPromptStrategy(),
            "few_shot_verification": FewShotVerificationPromptStrategy()
        }
    
    def get_strategy(self, strategy_name: str) -> Optional[PromptStrategy]:
        """Get a specific prompting strategy"""
        return self.strategies.get(strategy_name)
    
    def get_all_strategies(self) -> Dict[str, PromptStrategy]:
        """Get all available strategies"""
        return self.strategies
    
    def get_strategy_names(self) -> list:
        """Get list of available strategy names"""
        return list(self.strategies.keys())
    
    def create_prompt_for_strategy(self, strategy_name: str, paragraph: str) -> Tuple[str, Optional[str]]:
        """Create prompt for a specific strategy and paragraph"""
        strategy = self.get_strategy(strategy_name)
        if not strategy:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        return strategy.create_prompt(paragraph)
    
    def get_strategy_description(self, strategy_name: str) -> str:
        """Get description of a specific strategy"""
        strategy = self.get_strategy(strategy_name)
        if not strategy:
            return f"Unknown strategy: {strategy_name}"
        
        return strategy.get_description()
    
    def print_all_strategies(self):
        """Print information about all strategies"""
        print("Available Prompting Strategies:")
        print("=" * 50)
        
        for name, strategy in self.strategies.items():
            print(f"\n{name.upper()}:")
            print(f"  Description: {strategy.get_description()}")
            
            # Show example prompt
            sample_text = "The scientist conducted his research carefully."
            user_prompt, system_message = strategy.create_prompt(sample_text)
            
            if system_message:
                print(f"  System Message: {system_message}")
            print(f"  User Prompt Preview: {user_prompt[:100]}...")

def test_prompts():
    """Test all prompting strategies with a sample paragraph"""
    sample_paragraph = """The scientist conducted his research carefully in the laboratory. 
    He analyzed the data and shared his findings with his colleagues. The professor 
    encouraged her students to think critically about the results and each student 
    was expected to submit his or her own analysis."""
    
    manager = PromptManager()
    
    print("Testing Prompting Strategies")
    print("=" * 50)
    
    for strategy_name in manager.get_strategy_names():
        print(f"\n--- {strategy_name.upper()} STRATEGY ---")
        print(f"Description: {manager.get_strategy_description(strategy_name)}")
        
        user_prompt, system_message = manager.create_prompt_for_strategy(strategy_name, sample_paragraph)
        
        if system_message:
            print(f"\nSystem Message:")
            print(system_message)
        
        print(f"\nUser Prompt:")
        print(user_prompt)
        print("-" * 50)

if __name__ == "__main__":
    test_prompts()
