#!/bin/bash

# AI Safety Models POC - Setup Script
# This script automates the setup process for the AI Safety Models system

echo "🛡️  AI Safety Models POC - Setup Script"
echo "========================================"

# Check Python version
python_version=$(python3 --version 2>/dev/null)
if [ $? -ne 0 ]; then
    echo "❌ Error: Python 3 is not installed or not accessible as 'python3'"
    echo "Please install Python 3.8 or higher before running this script."
    exit 1
fi

echo "✅ Found Python: $python_version"

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: No virtual environment detected."
    echo "It's recommended to use a virtual environment."

    read -p "Would you like to create and activate a virtual environment? (y/n): " create_venv

    if [ "$create_venv" = "y" ] || [ "$create_venv" = "Y" ]; then
        echo "📦 Creating virtual environment..."
        python3 -m venv ai_safety_env

        echo "🔧 Activating virtual environment..."
        source ai_safety_env/bin/activate

        echo "✅ Virtual environment activated"
        echo "Note: To activate this environment in future sessions, run:"
        echo "source ai_safety_env/bin/activate"
    fi
fi

# Install pip dependencies
echo "📥 Installing Python dependencies..."
pip install --upgrade pip

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    if [ $? -eq 0 ]; then
        echo "✅ Dependencies installed successfully"
    else
        echo "❌ Error installing dependencies"
        exit 1
    fi
else
    echo "❌ Error: requirements.txt not found"
    echo "Please ensure you're running this script from the project directory."
    exit 1
fi

# Download NLTK data
echo "🔤 Downloading NLTK data..."
python3 -c "
import nltk
try:
    nltk.download('punkt', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('stopwords', quiet=True)
    print('✅ NLTK data downloaded successfully')
except Exception as e:
    print(f'⚠️ Warning: Error downloading NLTK data: {e}')
    print('You may need to download NLTK data manually later')
"

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p models/saved
mkdir -p logs
mkdir -p static/css
mkdir -p static/js

echo "✅ Directory structure created"

# Test the installation
echo "🧪 Testing installation..."
python3 -c "
try:
    import torch
    import transformers
    import flask
    import pandas
    import numpy
    import sklearn
    print('✅ All required packages imported successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

# Generate sample data
echo "📊 Generating sample data..."
python3 -c "
from sample_data_generator import SampleDataGenerator
import pandas as pd

try:
    generator = SampleDataGenerator()

    # Generate sample datasets
    abuse_data = generator.generate_abuse_detection_data(100)
    crisis_data = generator.generate_crisis_data(50)
    content_data = generator.generate_content_filtering_data(30)

    # Save sample data
    abuse_data.to_csv('data/raw/sample_abuse_data.csv', index=False)
    crisis_data.to_csv('data/raw/sample_crisis_data.csv', index=False)
    content_data.to_csv('data/raw/sample_content_data.csv', index=False)

    print('✅ Sample data generated and saved')
except Exception as e:
    print(f'⚠️ Warning: Error generating sample data: {e}')
    print('You can generate sample data manually later')
"

# Check if everything is working
echo "✅ Running quick system test..."
python3 -c "
try:
    from ai_safety_integration import AISafetySystem
    system = AISafetySystem()
    print('✅ AI Safety System initialized successfully')
except Exception as e:
    print(f'❌ System initialization error: {e}')
    print('Please check the installation and try running setup again')
"

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Start the web application: python app.py"
echo "2. Or run the CLI demo: python main.py --mode demo"
echo "3. Or run the interactive demo: python main.py --mode interactive"
echo ""
echo "The web interface will be available at: http://localhost:5000"
echo ""
echo "For more information, see README.md"
echo ""
