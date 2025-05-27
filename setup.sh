#!/bin/bash
set -e

echo "🚀 Setting up rows_vision..."

# Check if .env exists
if [ ! -f .env ]; then
    cp .env.example .env
    echo "⚠️  Please edit .env with your API keys before continuing"
    exit 1
fi

# Check if at least one API key is set
if ! grep -q "^API_KEY_" .env | grep -v "your_.*_key_here"; then
    echo "❌ No API keys found in .env file"
    echo "Please add at least one API key to .env"
    exit 1
fi

echo "✅ Setup complete! You can now run:"
echo "  Docker: docker build -t rows-vision . && docker run -d -p 8080:8080 --env-file .env rows-vision"
echo "  Local:  python3 main.py"