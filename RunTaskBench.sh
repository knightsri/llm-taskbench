#!/bin/bash
# LLM TaskBench - One-command startup script

set -e

echo "🚀 Starting LLM TaskBench..."
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  No .env file found. Creating from .env.example..."
    cp .env.example .env
    echo "✅ Created .env file"
    echo ""
    echo "📝 Please edit .env and add your API keys:"
    echo "   - OPENROUTER_API_KEY"
    echo "   - ANTHROPIC_API_KEY (optional, for quality check generation)"
    echo ""
    read -p "Press Enter after updating .env file..."
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

echo "📦 Building and starting services..."
docker-compose up --build -d

echo ""
echo "⏳ Waiting for services to be healthy..."
sleep 5

# Check service health
echo ""
echo "🔍 Checking service status..."
docker-compose ps

echo ""
echo "✅ LLM TaskBench is running!"
echo ""
echo "🌐 Access points:"
echo "   - Frontend UI: http://localhost:9999"
echo "   - Backend API: http://localhost:8000"
echo "   - API Docs: http://localhost:8000/docs"
echo ""
echo "📝 Useful commands:"
echo "   - View logs: docker-compose logs -f"
echo "   - Stop services: docker-compose down"
echo "   - Restart: docker-compose restart"
echo ""
echo "🎉 Happy benchmarking!"
