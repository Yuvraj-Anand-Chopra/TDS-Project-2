import os
import logging
import asyncio
import json
import re
import base64
import time
from typing import Optional, Dict, Any, Callable
from urllib.parse import urlparse

import requests
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from pydantic_settings import BaseSettings
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
class Settings(BaseSettings):
    API_NAME: str = "LLM Quiz Solver"
    API_VERSION: str = "1.0.0"
    
    # Credentials
    GEMINI_API_KEY: str
    GEMINI_MODEL: str = "gemini-2.0-flash"
    
    # Defaults (can be overridden by env vars)
    EMAIL: str = "24f2002642@ds.study.iitm.ac.in"
    SECRET: str = "YuvrajChopra2024Secret"
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

# --- MODELS ---
class QuizRequest(BaseModel):
    email: str
    secret: str
    url: HttpUrl

class QuizResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    message: str = ""

# --- HELPERS ---

class GeminiHelper:
    def __init__(self):
        try:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self.model = genai.GenerativeModel(settings.GEMINI_MODEL)
            logger.info(f"Gemini initialized with model: {settings.GEMINI_MODEL}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {str(e)}")
            raise

    async def generate_response(self, prompt: str, max_retries: int = 3) -> str:
        for attempt in range(max_retries):
            try:
                logger.info(f"Generating response from Gemini (attempt {attempt + 1}/{max_retries})")
                response = self.model.generate_content(
                    prompt,
                    generation_config={"max_output_tokens": 1024, "temperature": 0.7}
                )
                if response.text:
                    return response.text.strip()
                return "No response"
            except Exception as e:
                logger.warning(f"Gemini attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(1)

class RequestHandler:
    @staticmethod
    async def submit_json(url: str, payload: dict, timeout: int = 30) -> dict:
        """Submit JSON payload to URL safely"""
        logger.info(f"Submitting to {url}")
        # Run blocking request in a thread
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, 
            lambda: requests.post(url, json=payload, timeout=timeout)
        )
        response.raise_for_status()
        return response.json()

class WebScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    async def fetch_quiz_page(self, url: str) -> Dict:
        """Fetch quiz page and decode content"""
        try:
            logger.info(f"Fetching quiz page: {url}")
            
            # Run blocking get in thread
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: self.session.get(url, timeout=30)
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Decode Base64 content if present (common in these TDS quizzes)
            scripts = soup.find_all('script')
            for script in scripts:
                if script.string and 'atob' in script.string:
                    matches = re.findall(r'atob\([`"\']([A-Za-z0-9+/=\\s]+)[`"\']\)', script.string)
                    for match in matches:
                        try:
                            clean_match = match.replace('\n', '').replace(' ', '').replace('\r', '')
                            decoded = base64.b64decode(clean_match).decode('utf-8')
                            decoded_soup = BeautifulSoup(decoded, 'html.parser')
                            
                            # Replace result div or append
                            result_div = soup.find(id='result')
                            if result_div:
                                result_div.clear()
                                result_div.append(decoded_soup)
                        except Exception as e:
                            logger.warning(f"Failed to decode base64: {e}")

            # Extract text
            result_div = soup.find(id='result')
            instructions = result_div.get_text('\n').strip() if result_div else soup.get_text('\n').strip()
            
            return {
                'url': url,
                'instructions': instructions
            }
        except Exception as e:
            logger.error(f"Error fetching quiz page: {e}")
            raise

# --- PROCESSOR ---

class QuizProcessor:
    def __init__(self):
        self.gemini = GeminiHelper()
        self.scraper = WebScraper()

    async def solve_quiz(self, url: str) -> Dict:
        try:
            # 1. Fetch and Parse
            quiz_data = await self.scraper.fetch_quiz_page(url)
            instructions = quiz_data['instructions']
            logger.info(f"Extracted instructions length: {len(instructions)}")

            # 2. Determine Submit URL
            parsed = urlparse(url)
            submit_url = f"{parsed.scheme}://{parsed.netloc}/submit"
            
            # 3. Solve with LLM
            prompt = f"""
            You are a quiz solver. Analyze this quiz question and provide the answer.
            
            Question:
            {instructions}
            
            Provide ONLY the answer value. If it is a math problem, solve it. 
            If it asks for a specific word, provide that word.
            """
            answer = await self.gemini.generate_response(prompt)
            logger.info(f"Generated answer: {answer}")

            # 4. Submit
            payload = {
                "email": settings.EMAIL,
                "secret": settings.SECRET,
                "url": url,
                "answer": answer
            }
            
            response = await RequestHandler.submit_json(submit_url, payload)
            logger.info(f"Submission response: {response}")
            
            return {
                "success": True,
                "answer": answer,
                "api_response": response
            }

        except Exception as e:
            logger.error(f"Quiz processing failed: {e}")
            return {"success": False, "error": str(e)}

# --- API APP ---

app = FastAPI(title=settings.API_NAME, version=settings.API_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

processor = QuizProcessor()

@app.get("/")
async def root():
    return {"message": "LLM Quiz Solver API is running", "docs": "/docs"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/solve-quiz")
async def solve_quiz_endpoint(request: QuizRequest):
    logger.info(f"Received quiz request for: {request.url}")
    
    # Ensure email/secret from request match our config if needed, 
    # or just use the ones passed in the request if you prefer dynamic auth.
    # For this project, we usually override with our secure env vars for submission
    # but passing them through is fine too.
    
    result = await processor.solve_quiz(str(request.url))
    
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))
        
    return result
