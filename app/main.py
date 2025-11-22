import os
import logging
import asyncio
import json
import re
import base64
import time
from typing import Optional, Dict, Any
from urllib.parse import urlparse

import requests
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, field_validator
from pydantic_settings import BaseSettings
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# --- CONFIGURATION & ENV VARS ---
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    API_NAME: str = "LLM Quiz Solver"
    API_VERSION: str = "1.0.0"
    
    # API Credentials (set these in Render Dashboard)
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = "gemini-2.0-flash"
    
    # User Defaults
    EMAIL: str = "24f2002642@ds.study.iitm.ac.in"
    SECRET: str = "YuvrajChopra2024Secret"
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

# --- PYDANTIC MODELS (V2 COMPATIBLE) ---
class QuizRequest(BaseModel):
    email: str
    secret: str
    url: HttpUrl

    # V2 Validator Syntax
    @field_validator('url')
    @classmethod
    def validate_url(cls, v):
        return v

class QuizResponse(BaseModel):
    success: bool
    answer: Optional[Any] = None
    api_response: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# --- CORE LOGIC ---

class GeminiHelper:
    def __init__(self):
        if not settings.GEMINI_API_KEY:
            logger.warning("GEMINI_API_KEY not set! API will fail.")
            return
        try:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self.model = genai.GenerativeModel(settings.GEMINI_MODEL)
            logger.info(f"Gemini initialized: {settings.GEMINI_MODEL}")
        except Exception as e:
            logger.error(f"Gemini init failed: {e}")

    async def generate_answer(self, question_text: str) -> str:
        try:
            prompt = f"""
            Solve this quiz question directly. Return ONLY the answer value.
            
            Question:
            {question_text}
            
            If it's a math problem, return the number.
            If it's a text question, return the word/phrase.
            Do not add explanations or markdown.
            """
            response = self.model.generate_content(prompt)
            return response.text.strip() if response.text else "Error: No text"
        except Exception as e:
            logger.error(f"LLM Generation failed: {e}")
            return f"Error: {str(e)}"

class WebScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    async def fetch_and_parse(self, url: str) -> str:
        """Fetches page, handles Base64 obfuscation, returns instructions"""
        logger.info(f"Fetching {url}")
        
        # Run blocking request in thread pool
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: self.session.get(url, timeout=30))
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # TDS Project Specific: Decode Base64 in scripts
        scripts = soup.find_all('script')
        for script in scripts:
            if script.string and 'atob' in script.string:
                matches = re.findall(r'atob\([`"\']([A-Za-z0-9+/=\\s]+)[`"\']\)', script.string)
                for match in matches:
                    try:
                        clean_match = match.replace('\n', '').replace(' ', '')
                        decoded = base64.b64decode(clean_match).decode('utf-8')
                        # Inject decoded content back into soup
                        if 'result' in str(decoded):
                            soup.append(BeautifulSoup(decoded, 'html.parser'))
                    except:
                        pass

        # Fallback strategies to find the question text
        instructions = ""
        possible_ids = ['result', 'question', 'instruction', 'content']
        
        for pid in possible_ids:
            element = soup.find(id=pid)
            if element:
                instructions = element.get_text('\n').strip()
                break
        
        if not instructions:
            instructions = soup.get_text('\n').strip()
            
        return instructions

# --- API APPLICATION ---

app = FastAPI(title=settings.API_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

scraper = WebScraper()
llm = GeminiHelper()

@app.get("/health")
async def health():
    return {"status": "ok", "model": settings.GEMINI_MODEL}

@app.post("/solve-quiz", response_model=QuizResponse)
async def solve_quiz(request: QuizRequest):
    try:
        url_str = str(request.url)
        
        # 1. Get Instructions
        instructions = await scraper.fetch_and_parse(url_str)
        logger.info(f"Instructions length: {len(instructions)}")
        
        # 2. Solve
        answer = await llm.generate_answer(instructions)
        logger.info(f"Solved answer: {answer}")
        
        # 3. Submit
        parsed = urlparse(url_str)
        # Handle TDS submission URL convention
        submit_url = f"{parsed.scheme}://{parsed.netloc}/submit"
        
        payload = {
            "email": settings.EMAIL,
            "secret": settings.SECRET,
            "url": url_str,
            "answer": answer
        }
        
        logger.info(f"Submitting to {submit_url}")
        loop = asyncio.get_event_loop()
        submit_res = await loop.run_in_executor(
            None, 
            lambda: requests.post(submit_url, json=payload, timeout=30)
        )
        submit_res.raise_for_status()
        api_response = submit_res.json()
        
        return QuizResponse(
            success=True,
            answer=answer,
            api_response=api_response
        )

    except Exception as e:
        logger.error(f"Process failed: {e}")
        return QuizResponse(success=False, error=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
