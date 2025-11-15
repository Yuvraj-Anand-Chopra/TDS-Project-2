# LLM Analysis Quiz Solver

A robust, production-ready FastAPI application that automatically solves data analysis quizzes using Google Gemini API, Playwright web scraping, and intelligent data processing.

## Features

- FastAPI Backend: Modern async Python web framework with automatic OpenAPI documentation
- Google Gemini Integration: LLM-powered analysis and data extraction
- Headless Browser Automation: Playwright-based JavaScript rendering and DOM parsing
- Data Processing Pipeline: PDF extraction, API integration, data analysis, and visualization
- Prompt Injection Defense: Security mechanisms to protect against LLM attacks
- Robust Error Handling: Exponential backoff retry logic, timeout management, comprehensive logging
- Docker Support: Containerized deployment for consistent environments
- CI/CD Pipeline: GitHub Actions workflow for automated testing and deployment
- Comprehensive Testing: Unit tests with pytest and high code coverage

## Requirements

- Python 3.11+
- Google Gemini API Key
- Docker (optional, for containerization)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/llm-analysis-quiz.git
cd llm-analysis-quiz
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
python -m playwright install chromium
```

### 4. Configure Environment Variables

```bash
cp .env.example .env
# Edit .env with your configuration
```

Fill in the `.env` file:

```
EMAIL=your-email@vit.ac.in
SECRET=your-secure-secret-key
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL=gemini-2.0-flash
LOG_LEVEL=INFO
DEBUG=False
```

## Running the Application

### Local Development

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

**API Documentation**: Visit `http://localhost:8000/docs` for interactive Swagger UI

### Docker Deployment

```bash
# Build and run with docker-compose
docker-compose up --build

# Or build image manually
docker build -t llm-quiz:latest .
docker run -p 8000:8000 --env-file .env llm-quiz:latest
```

## API Endpoints

### Health Check
```
GET /health
```

### Solve Quiz
```
POST /solve-quiz
```

Request:
```json
{
  "email": "student@vit.ac.in",
  "secret": "your-secret",
  "url": "https://quiz.example.com/quiz-123"
}
```

## Testing

Run all tests:

```bash
pytest tests/ -v
```

Run with coverage:

```bash
pytest tests/ -v --cov=app --cov-report=html
```

## Project Structure

```
llm-analysis-quiz/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration management
│   ├── models.py            # Pydantic models
│   ├── security.py          # Authentication & defense
│   ├── scrapers.py          # Browser automation
│   ├── analyzer.py          # Data processing
│   ├── processor.py         # Quiz processing logic
│   ├── request_handler.py   # Request utilities
│   ├── llm_helper.py        # Gemini API wrapper
│   └── utils.py             # Helper functions
├── tests/
│   ├── test_api.py
│   ├── test_analyzer.py
│   └── __init__.py
├── .github/
│   └── workflows/
│       └── ci-cd.yml        # GitHub Actions pipeline
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, email your-email@vit.ac.in or create an issue on GitHub.
