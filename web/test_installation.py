#!/usr/bin/env python3
"""
AI Safety Models POC - Installation Test

Quick test to verify that all components are properly installed and working.
"""

import sys
import importlib

def test_import(module_name, description):
    """Test if a module can be imported"""
    try:
        importlib.import_module(module_name)
        print(f"✅ {description}: OK")
        return True
    except ImportError as e:
        print(f"❌ {description}: FAILED - {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of the system"""
    try:
        from sys.ai_safety_integration import AISafetySystem
        system = AISafetySystem()
        print("✅ AI Safety System initialization: OK")
        return True
    except Exception as e:
        print(f"❌ AI Safety System initialization: FAILED - {e}")
        return False

def main():
    """Run installation tests"""

    print("🧪 AI Safety Models POC - Installation Test")
    print("=" * 50)

    # Test required packages
    tests = [
        ("torch", "PyTorch"),
        ("transformers", "Hugging Face Transformers"),
        ("sklearn", "Scikit-learn"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("flask", "Flask"),
        ("nltk", "NLTK"),
        ("textblob", "TextBlob"),
        ("vaderSentiment", "VADER Sentiment"),
    ]

    print("\nTesting package imports...")
    passed_tests = 0

    for module, description in tests:
        if test_import(module, description):
            passed_tests += 1

    print(f"\nPackage tests: {passed_tests}/{len(tests)} passed")

    # Test NLTK data
    print("\nTesting NLTK data...")
    try:
        import nltk
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('vader_lexicon')
        print("✅ NLTK data: OK")
        nltk_ok = True
    except Exception as e:
        print(f"❌ NLTK data: FAILED - {e}")
        print("Run: python -c \"import nltk; nltk.download('punkt'); nltk.download('vader_lexicon')\"")
        nltk_ok = False

    # Test system functionality
    print("\nTesting system functionality...")
    system_ok = test_basic_functionality()

    # Test file structure
    print("\nTesting file structure...")
    import os
    required_files = [
        'config.py',
        'sample_data_generator.py',
        'abuse_detection_model.py',
        'escalation_detection_model.py',
        'crisis_intervention_model.py',
        'content_filtering_model.py',
        'ai_safety_integration.py',
        'main.py',
        'app.py',
        'templates/index.html'
    ]

    files_ok = 0
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}: OK")
            files_ok += 1
        else:
            print(f"❌ {file_path}: MISSING")

    print(f"\nFile structure tests: {files_ok}/{len(required_files)} passed")

    # Overall result
    print("\n" + "=" * 50)

    all_passed = (passed_tests == len(tests) and 
                  nltk_ok and 
                  system_ok and 
                  files_ok == len(required_files))

    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("\nThe AI Safety Models POC is ready to use!")
        print("\nNext steps:")
        print("• Run demo: python demo.py")
        print("• Start web app: python app.py")
        print("• Interactive CLI: python main.py --mode interactive")
        return 0
    else:
        print("❌ SOME TESTS FAILED!")
        print("\nPlease fix the issues above before proceeding.")
        print("You may need to:")
        print("• Install missing packages: pip install -r requirements.txt")
        print("• Download NLTK data: python -c \"import nltk; nltk.download('punkt'); nltk.download('vader_lexicon')\"")
        print("• Ensure all files are in the correct location")
        return 1

if __name__ == "__main__":
    sys.exit(main())
