#!/bin/bash

# Deployment script for LLM-Powered Intelligent Query-Retrieval System

echo "🚀 Starting deployment of LLM-Powered Query-Retrieval System..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip first."
    exit 1
fi

echo "✅ Python and pip found"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Check if all required packages are installed
echo "🔍 Verifying installation..."
python3 -c "
try:
    import fastapi
    import pydantic
    import uvicorn
    import httpx
    import numpy
    import sentence_transformers
    import faiss
    import PyPDF2
    import docx
    print('✅ All required packages installed successfully')
except ImportError as e:
    print(f'❌ Missing package: {e}')
    exit(1)
"

# Set environment variables
echo "🔧 Setting up environment..."
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Start the application
echo "🌟 Starting the application..."
echo "API will be available at: http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"
echo "Health Check: http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run the application
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
