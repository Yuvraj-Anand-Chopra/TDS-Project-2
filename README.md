# TDS Project 2 - LLM Quiz Solver

This is my project for the Tools in Data Science course (Sep 2025). It's an API that can automatically solve data analysis quizzes using AI!

## What This Does

I built a FastAPI application that:
- Receives quiz URLs through a POST request
- Visits those URLs and reads the questions
- Uses Google's Gemini AI to solve them
- Submits the answers automatically
- Can handle multiple questions in a chain

The cool part is it can handle different types of questions like downloading files, parsing data, doing calculations, etc.

## Technologies Used

- **FastAPI** - Web API framework
- **Google Gemini AI** - To understand and solve quiz questions
- **BeautifulSoup** - HTML parsing and content extraction
- **Render** - Cloud deployment platform
- **Python 3.11**

## Project Structure

```
TDS-Project-2/
├── app/
│   ├── main.py          # Main API code
│   └── __init__.py
├── requirements.txt     # Python dependencies
├── .gitignore          
├── LICENSE             # MIT License
└── README.md           
```

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Yuvraj-Anand-Chopra/TDS-Project-2.git
cd TDS-Project-2
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root directory:
```
EMAIL=24f2002642@ds.study.iitm.ac.in
SECRET=Secret
GEMINI_API_KEY=your-gemini-api-key-here
```

### 4. Run Locally
```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Health Check
```
GET /health
```

**Response:**
```json
{
  "status": "ok",
  "model": "gemini-2.0-flash"
}
```

### Solve Quiz
```
POST /solve-quiz
```

**Request:**
```json
{
  "email": "24f2002642@ds.study.iitm.ac.in",
  "secret": "Secret",
  "url": "https://tds-llm-analysis.s-anand.net/demo"
}
```

**Response:**
```json
{
  "correct": true,
  "url": "https://next-question-url",
  "reason": null
}
```

## Testing

### Using PowerShell
```powershell
Invoke-RestMethod -Uri "https://tds-project-2-yuvraj.onrender.com/solve-quiz" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"email":"24f2002642@ds.study.iitm.ac.in","secret":"Secret","url":"https://tds-llm-analysis.s-anand.net/demo"}'
```

### Using Browser
Visit the interactive API docs: https://tds-project-2-yuvraj.onrender.com/docs

## How It Works

1. **Receives Request** - API gets email, secret, and quiz URL
2. **Validates Secret** - Checks if secret matches (returns 403 if invalid)
3. **Fetches Quiz Page** - Downloads the HTML content
4. **Decodes Content** - Handles JavaScript obfuscation (base64 decoding)
5. **Extracts Question** - Parses HTML to get the actual question
6. **AI Processing** - Sends question to Gemini AI
7. **Executes Code** - If needed, runs Python code to solve data problems
8. **Submits Answer** - Posts answer to the quiz system
9. **Handles Chain** - Automatically processes next question if available

## Key Features

- **Secret Validation** - Secure API access with environment-based secrets
- **Base64 Decoding** - Handles JavaScript obfuscated content
- **LLM Integration** - Uses Google Gemini 2.0 Flash for intelligent problem solving
- **Dynamic Code Execution** - Can run Python code for data processing tasks
- **Multi-step Support** - Automatically chains through multiple quiz questions
- **Type Flexibility** - Handles int, float, and string answer types

## Deployment

**Live URL:** https://tds-project-2-yuvraj.onrender.com

Deployed on Render with:
- Automatic deployments from GitHub main branch
- Environment variables configured in Render dashboard
- Free tier hosting

## Challenges Solved

- **JavaScript Rendering**: Implemented base64 decoding for obfuscated content
- **Secret Security**: Proper environment variable validation
- **Type Handling**: Automatic conversion of numeric strings to int/float
- **Multi-step Quizzes**: Recursive handling of question chains

## Requirements

See `requirements.txt` for full list. Main dependencies:
- fastapi==0.104.1
- uvicorn==0.24.0
- google-generativeai==0.3.0
- beautifulsoup4==4.12.2
- requests==2.31.0
- pandas==2.1.3

## Submission Details

- **Course:** Tools in Data Science (Sep 2025)
- **Student:** Yuvraj Anand Chopra
- **Email:** 24f2002642@ds.study.iitm.ac.in
- **GitHub:** https://github.com/Yuvraj-Anand-Chopra/TDS-Project-2
- **API Endpoint:** https://tds-project-2-yuvraj.onrender.com/solve-quiz
- **Evaluation Date:** November 29, 2025

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- TDS course team for the project specification
- Google Gemini AI for LLM capabilities
- FastAPI community for excellent documentation
- Render for free hosting platform

---

**Note:** This is a student project for educational purposes. The API runs on Render's free tier and may take a few seconds to wake up on first request.
