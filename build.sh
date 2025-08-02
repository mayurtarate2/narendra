#!/usr/bin/env bash
# Render.com build script

set -o errexit  # exit on error

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download required NLTK data
python -c "
import nltk
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    print('NLTK data downloaded successfully')
except Exception as e:
    print(f'NLTK download warning: {e}')
"

# Create necessary directories
mkdir -p logs
mkdir -p temp

echo "Build completed successfully!"
