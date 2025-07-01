#!/usr/bin/env python3
"""
Quick setup and test script for the Gender Bias study
Run this script to verify everything is working correctly
"""

import subprocess
import sys
from pathlib import Path
import os

def run_command(command, description):
    """Run a command and report success/failure"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - SUCCESS")
            return True
        else:
            print(f"❌ {description} - FAILED")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False

def check_file_exists(file_path, description):
    """Check if a file exists"""
    if Path(file_path).exists():
        print(f"✅ {description} - EXISTS")
        return True
    else:
        print(f"❌ {description} - MISSING")
        return False

def main():
    print("🚀 Gender Bias Study - Quick Setup & Test")
    print("=" * 50)
    
    # Check Python version
    print(f"🐍 Python version: {sys.version}")
    
    # Check if we're in the right directory
    if not Path("main.py").exists():
        print("❌ Please run this script from the project root directory")
        sys.exit(1)
    
    # 1. Install dependencies
    print("\n📦 INSTALLING DEPENDENCIES")
    if not run_command("pip install -r requirements.txt", "Installing Python packages"):
        print("⚠️  Installation failed. You may need to activate your virtual environment first.")
        print("   Try: python -m venv venv && source venv/bin/activate")
    
    # 2. Check environment setup
    print("\n🔧 CHECKING ENVIRONMENT")
    
    # Check for .env file
    if not check_file_exists(".env", ".env file"):
        print("   Copy .env.example to .env and add your API keys")
        if check_file_exists(".env.example", ".env.example template"):
            run_command("cp .env.example .env", "Creating .env file from template")
    
    # 3. Test project setup
    print("\n🧪 TESTING PROJECT SETUP")
    
    # Test imports
    test_script = '''
import sys
sys.path.append("src")
try:
    from src.config import EXPERIMENT_CONFIG
    from src.utils import validate_environment
    from src.data_processing import CorpusProcessor
    print("✅ All core imports successful")
except Exception as e:
    print(f"❌ Import error: {e}")
    '''
    
    if not run_command(f'python -c "{test_script}"', "Testing core imports"):
        print("   Check that all dependencies are installed correctly")
    
    # 4. Test main CLI
    print("\n🎯 TESTING MAIN CLI")
    if not run_command("python main.py --help", "Testing main CLI help"):
        print("   There may be an issue with the main.py file")
    
    # 5. Setup project structure
    print("\n📁 SETTING UP PROJECT STRUCTURE")
    run_command("python main.py setup", "Running project setup")
    
    # 6. Check directory structure
    print("\n📂 CHECKING DIRECTORY STRUCTURE")
    directories = [
        "src",
        "data/corpus", 
        "data/outputs",
        "data/results",
        "notebooks"
    ]
    
    for directory in directories:
        check_file_exists(directory, f"Directory: {directory}")
    
    # 7. Test components
    print("\n🔍 TESTING COMPONENTS")
    run_command("python main.py test", "Testing all components")
    
    # Final recommendations
    print("\n" + "=" * 50)
    print("🎉 SETUP COMPLETE!")
    print("\n📋 NEXT STEPS:")
    print("1. Add your API keys to the .env file:")
    print("   - OPENAI_API_KEY=your_key_here")
    print("   - GEMINI_API_KEY=your_key_here")
    print("")
    print("2. Prepare your corpus:")
    print("   - Edit data/corpus/corpus_template.csv")
    print("   - Add your 25 paragraphs with gendered terms")
    print("   - Save as data/corpus/paragraphs.csv")
    print("")
    print("3. Run the experiment:")
    print("   python main.py run-experiment")
    print("")
    print("4. Analyze results:")
    print("   python main.py analyze")
    print("   # OR use the Jupyter notebook")
    print("   jupyter notebook notebooks/Gender_Bias_Analysis.ipynb")
    print("")
    print("🚀 Ready to start your gender bias research!")

if __name__ == "__main__":
    main()
