# Quick Start Guide

Get the LLM Analysis Quiz Solver running in 5 minutes.

## Prerequisites

- Python 3.11 or higher
- Google Gemini API key (free at https://ai.google.dev)

## Step 1: Clone and Setup

```bash
git clone https://github.com/yourusername/llm-analysis-quiz.git
cd llm-analysis-quiz
```

## Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
python -m playwright install chromium
```

## Step 4: Configure

```bash
cp .env.example .env
```

Edit `.env` and add:
- Your email
- A secure secret (any string)
- Your Gemini API key from https://ai.google.dev

## Step 5: Run

```bash
uvicorn app.main:app --reload
```

Open browser: http://localhost:8000/docs

## Testing

```bash
# Run all tests
pytest tests/ -v

# Test specific endpoint
curl -X GET http://localhost:8000/health
```

## Docker Setup

```bash
docker-compose up --build
```

Visit: http://localhost:8000/docs

## Useful Commands

```bash
# Start development server
uvicorn app.main:app --reload --log-level debug

# Run tests with coverage
pytest tests/ --cov=app --cov-report=html

# Build Docker image
docker build -t llm-quiz:latest .

# View logs
tail -f app.log

# Check Gemini connection
curl -X POST http://localhost:8000/test-connection
```

## Common Issues

**"ModuleNotFoundError: No module named 'playwright'"**
```bash
python -m playwright install chromium
```

**"GEMINI_API_KEY not set"**
- Check `.env` file exists
- Verify key is set correctly
- Restart server after changing `.env`

**Port 8000 already in use**
```bash
uvicorn app.main:app --port 8001
```

**Playwright browser won't start**
```bash
python -m playwright install-deps chromium
```

## Next Steps

- Customize system/user prompts in `app/security.py`
- Adjust timeout values in `app/config.py`
- Configure logging level
- Set up CI/CD with GitHub Actions
- Deploy to production
- Submit to quiz evaluation
