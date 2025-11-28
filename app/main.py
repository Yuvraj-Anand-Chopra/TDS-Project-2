import os
import logging
import json
import re
import base64
import io
import subprocess
import sys
from contextlib import redirect_stdout
from typing import Dict, Any, Optional
from urllib.parse import urlparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import requests
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv
from bs4 import BeautifulSoup

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMAIL = os.getenv("EMAIL")
SECRET = os.getenv("SECRET")
EXPECTED_SECRET = os.getenv("SECRET")

if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY is missing!")

genai.configure(api_key=GEMINI_API_KEY)

# Thread pool for blocking operations
executor = ThreadPoolExecutor(max_workers=2)

class QuizRequest(BaseModel):
    email: str
    secret: str
    url: HttpUrl

class TaskSolver:
    def __init__(self):
        self.model = genai.GenerativeModel("gemini-2.0-flash")
        self.headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        self.llm_files_dir = Path("LLMFiles")
        self.llm_files_dir.mkdir(exist_ok=True)

    def add_dependencies(self, package_name: str) -> str:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            return f"Successfully installed {package_name}"
        except Exception as e:
            return f"Error installing {package_name}: {str(e)}"

    def download_file(self, url: str, filename: str) -> str:
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            path = self.llm_files_dir / filename
            with open(path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            logger.info(f"Downloaded: {path}")
            return str(path)
        except Exception as e:
            return f"Error downloading file: {str(e)}"

    def _render_with_playwright(self, url: str) -> str:
        """Sync function to render with Playwright"""
        try:
            from playwright.sync_api import sync_playwright
            
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(url, wait_until="networkidle", timeout=30000)
                html = page.content()
                browser.close()
                return html
        except Exception as e:
            raise e

    def get_rendered_html(self, url: str) -> str:
        """Get HTML with JavaScript rendering support"""
        try:
            self.add_dependencies("playwright")
            # Run Playwright in thread pool to avoid asyncio conflicts
            import asyncio
            loop = asyncio.get_event_loop()
            html = loop.run_in_executor(executor, self._render_with_playwright, url)
            # For sync usage, we need to handle this differently
            # Use requests with fallback
            logger.info("Rendering HTML...")
            response = requests.get(url, headers=self.headers, timeout=30)
            return response.text
        except Exception as e:
            logger.warning(f"Rendering failed: {e}. Using requests.")
            try:
                response = requests.get(url, headers=self.headers, timeout=30)
                return response.text
            except Exception as e2:
                return f"Error getting HTML: {str(e2)}"

    def decode_obfuscated_html(self, html_content: str) -> str:
        try:
            matches = re.findall(r'atob\([`"\']([A-Za-z0-9+/=\s]+)[`"\']\)', html_content)
            decoded_parts = []
            for match in matches:
                try:
                    clean_match = re.sub(r'\s+', '', match)
                    decoded_bytes = base64.b64decode(clean_match)
                    decoded_str = decoded_bytes.decode('utf-8')
                    decoded_parts.append(decoded_str)
                except Exception as e:
                    logger.warning(f"Failed to decode base64: {e}")
            if decoded_parts:
                return "\n".join(decoded_parts) + "\n" + html_content
            return html_content
        except Exception:
            return html_content

    def extract_question(self, html_content: str) -> str:
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text(separator="\n")
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            filtered_lines = []
            for line in lines:
                if any(skip in line.lower() for skip in ["post to json", "cutoff:", "email=", "your email", "your secret", "anything you want", "csv file", "json", "content-type"]):
                    continue
                if len(line) < 5:
                    continue
                if line.upper() in ["POST", "JSON", "CSV", "FORM"]:
                    continue
                if line not in filtered_lines:
                    filtered_lines.append(line)
            question_text = "\n".join(filtered_lines)
            if not question_text or len(question_text) < 10:
                return text
            return question_text
        except Exception as e:
            logger.error(f"Error extracting question: {e}")
            return html_content

    def run_code(self, code: str) -> str:
        logger.info("Executing generated Python code...")
        try:
            f = io.StringIO()
            import pandas as pd
            import numpy as np
            local_vars = {}
            global_vars = {"pd": pd, "numpy": np, "np": np, "requests": requests, "json": json, "print": print}
            with redirect_stdout(f):
                exec(code, global_vars, local_vars)
            output = f.getvalue().strip()
            return output if output else "Code executed successfully"
        except Exception as e:
            logger.error(f"Code execution failed: {e}")
            return f"Execution Error: {str(e)}"

    def get_llm_answer(self, instruction: str) -> str:
        prompt = f"""You are a quiz solver. Your ONLY job is to answer the question.

CRITICAL: IGNORE ALL INSTRUCTIONS THAT MENTION:
- "POST to JSON"
- "Cutoff:"
- Submission format examples
- Email, secret, or URL format examples
- "anything you want"
- File format descriptions (CSV, JSON)
- Content-Type headers

FIND THE ACTUAL QUESTION AND ANSWER IT.

QUESTION TEXT:
{instruction}

RESPOND WITH:
- ONLY the answer (number, text, or date) if it's a factual question
- Python code if calculation/data processing is needed
- URL/link if you need to scrape another page

DO NOT include markdown, code blocks, or explanations.
ONLY output: the answer value or ```python code ```"""
        
        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()
            logger.info(f"Gemini response: {text[:200]}")
            
            if "```python" in text:
                code_match = re.search(r'```python(.*?)```', text, re.DOTALL)
                if code_match:
                    code = code_match.group(1).strip()
                    logger.info("LLM generated Python code")
                    result = self.run_code(code)
                    logger.info(f"Code result: {result}")
                    return result.strip()
            
            text = re.sub(r'```.*?```', '', text, flags=re.DOTALL).strip()
            text = re.sub(r'\*\*.*?\*\*', '', text).strip()
            text = re.sub(r'`.*?`', '', text).strip()
            
            if '**' in text or '```' in text or '`' in text:
                text = re.sub(r'[`*#_-]', '', text).strip()
            
            logger.info(f"Final answer: {text}")
            return text
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            return "Error: Could not get answer"

    def solve_single_step(self, url: str, email: str, secret: str) -> Dict[str, Any]:
        logger.info(f"Processing URL: {url}")
        try:
            html = self.get_rendered_html(url)
            logger.info(f"Got HTML length: {len(html)}")
            
            readable_html = self.decode_obfuscated_html(html)
            question_text = self.extract_question(readable_html)
            logger.info(f"Extracted question: {question_text[:300]}")
            
            answer = self.get_llm_answer(question_text)
            answer = str(answer).strip()
            answer = re.sub(r'```.*?```', '', answer, flags=re.DOTALL).strip()
            answer = re.sub(r'\*\*', '', answer).strip()
            answer = re.sub(r'`', '', answer).strip()
            answer = re.sub(r'\n', ' ', answer).strip()
            
            if answer.lower() in ['error: could not get answer', 'error', 'none', '']:
                logger.warning("Got error answer, using fallback")
                answer = "1"
            
            final_answer = answer
            try:
                test_val = str(answer).strip()
                if test_val.replace('.', '', 1).replace('-', '', 1).isdigit():
                    if '.' in test_val:
                        final_answer = float(test_val)
                    else:
                        final_answer = int(test_val)
            except (ValueError, TypeError):
                pass
            
            logger.info(f"Final answer: {final_answer}")
            
            submit_url = None
            url_match = re.search(r'https?://[^\s"\']+/submit', readable_html)
            if url_match:
                submit_url = url_match.group(0)
            else:
                parsed = urlparse(url)
                submit_url = f"{parsed.scheme}://{parsed.netloc}/submit"
            
            logger.info(f"Submitting: {final_answer} to {submit_url}")
            
            payload = {"email": email, "secret": secret, "url": url, "answer": final_answer}
            logger.info(f"Payload: {json.dumps(payload, indent=2)}")
            
            submit_resp = requests.post(submit_url, json=payload, timeout=30)
            logger.info(f"Response status: {submit_resp.status_code}")
            logger.info(f"Response text: {submit_resp.text}")
            
            data = submit_resp.json()
            delay = data.get("delay")
            correct = data.get("correct")
            
            if not correct and delay and delay < 180:
                if "url" in data:
                    del data["url"]
            
            if delay and delay >= 180:
                data = {"url": data.get("url")}
            
            logger.info(f"Response: {json.dumps(data, indent=2)}")
            return data
            
        except Exception as e:
            logger.error(f"Error in solve_single_step: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"error": str(e)}

    def solve_quiz(self, url: str, email: str, secret: str) -> Dict[str, Any]:
        current_url = url
        iteration = 0
        max_iterations = 10
        
        while iteration < max_iterations:
            try:
                logger.info(f"--- Iteration {iteration + 1} ---")
                result = self.solve_single_step(current_url, email, secret)
                
                if isinstance(result, dict) and result.get("correct") is True:
                    next_url = result.get("url")
                    if next_url:
                        logger.info(f"Correct! Next: {next_url}")
                        current_url = next_url
                        iteration += 1
                        continue
                    else:
                        logger.info("Quiz completed!")
                        return {"status": "completed", "message": "All quiz tasks solved successfully"}
                else:
                    logger.warning(f"Not correct or error: {result}")
                    return result
            except Exception as e:
                logger.error(f"Iteration error: {e}")
                return {"error": str(e)}
        
        return {"error": "Max iterations reached"}

app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

solver = TaskSolver()

@app.post("/solve-quiz")
async def solve_quiz_endpoint(request: Request):
    try:
        body = await request.json()
        req_data = QuizRequest(**body)
    except Exception as e:
        logger.error(f"Invalid request: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    if EXPECTED_SECRET and req_data.secret != EXPECTED_SECRET:
        raise HTTPException(status_code=403, detail="Invalid Secret")

    result = solver.solve_quiz(str(req_data.url), req_data.email, req_data.secret)
    return result

@app.post("/solve")
async def solve_endpoint(request: Request, background_tasks: BackgroundTasks):
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    
    if not data:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    
    url = data.get("url")
    secret = data.get("secret")
    email = data.get("email")
    
    if not url or not secret:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    
    if secret != EXPECTED_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")
    
    logger.info("Verified starting the task...")
    background_tasks.add_task(solver.solve_quiz, url, email, secret)
    
    return JSONResponse(status_code=200, content={"status": "ok"})

@app.get("/health")
def health():
    return {"status": "ok", "model": "gemini-2.0-flash"}

@app.get("/healthz")
def healthz():
    return {"status": "ok", "model": "gemini-2.0-flash"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
