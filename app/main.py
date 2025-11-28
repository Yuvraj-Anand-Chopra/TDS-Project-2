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
from urllib.parse import urlparse, urljoin
from pathlib import Path

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

    def get_rendered_html(self, url: str) -> str:
        """Get HTML - render with Playwright if JS needed, fall back to requests"""
        try:
            # First try plain requests
            response = requests.get(url, headers=self.headers, timeout=30)
            html = response.text
            
            # Check if page has JavaScript that needs rendering
            if "atob(" in html or "document.querySelector" in html or "<script>" in html:
                logger.info("Detected JavaScript content, attempting Playwright render...")
                try:
                    self.add_dependencies("playwright")
                    from playwright.sync_api import sync_playwright
                    
                    with sync_playwright() as p:
                        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
                        page = browser.new_page()
                        page.goto(url, wait_until="networkidle", timeout=30000)
                        html = page.content()
                        browser.close()
                        logger.info("Playwright rendered HTML successfully")
                        return html
                except Exception as e:
                    logger.warning(f"Playwright failed ({e}), using plain HTML")
                    return html
            
            return html
        except Exception as e:
            logger.error(f"Error getting HTML: {e}")
            return ""

    def decode_obfuscated_html(self, html_content: str) -> str:
        """Decode base64-encoded content from atob() calls"""
        try:
            matches = re.findall(r'atob\([`"\']([A-Za-z0-9+/=\s]+)[`"\']\)', html_content)
            decoded_parts = []
            for match in matches:
                try:
                    clean_match = re.sub(r'\s+', '', match)
                    decoded_bytes = base64.b64decode(clean_match)
                    decoded_str = decoded_bytes.decode('utf-8')
                    decoded_parts.append(decoded_str)
                    logger.info(f"Decoded base64: {decoded_str[:100]}")
                except Exception as e:
                    logger.warning(f"Failed to decode base64: {e}")
            
            if decoded_parts:
                return "\n".join(decoded_parts) + "\n" + html_content
            return html_content
        except Exception:
            return html_content

    def extract_scraping_task(self, html_content: str, current_url: str) -> Optional[str]:
        """Extract URLs that need to be scraped (e.g., /demo-scrape-data)"""
        try:
            # Look for href attributes in links
            link_pattern = r'href=["\']([^"\']+)["\']'
            links = re.findall(link_pattern, html_content)
            
            for link in links:
                # Normalize relative URLs
                if link.startswith('/'):
                    parsed = urlparse(current_url)
                    full_url = f"{parsed.scheme}://{parsed.netloc}{link}"
                else:
                    full_url = urljoin(current_url, link)
                
                # Look for data-fetching URLs (scrape-data, demo-audio, etc)
                if any(x in full_url for x in ['scrape-data', 'audio', 'file', 'data']):
                    logger.info(f"Found scraping URL: {full_url}")
                    return full_url
            
            return None
        except Exception as e:
            logger.error(f"Error extracting scraping task: {e}")
            return None

    def extract_secret_from_content(self, content: str) -> Optional[str]:
        """Extract secret/answer from fetched content"""
        try:
            # Look for common secret/code patterns
            secret_patterns = [
                r'secret["\s:=]+([A-Za-z0-9_\-]+)',
                r'code["\s:=]+([A-Za-z0-9_\-]+)',
                r'answer["\s:=]+([A-Za-z0-9_\-]+)',
                r'<code>([A-Za-z0-9_\-]+)</code>',
                r'<pre>([A-Za-z0-9_\-]+)</pre>',
                r'SECRET["\s:=]+([A-Za-z0-9_\-]+)',
            ]
            
            for pattern in secret_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    secret = match.group(1).strip()
                    logger.info(f"Extracted secret: {secret}")
                    return secret
            
            # If no pattern matches, try to find any long alphanumeric string
            candidates = re.findall(r'[A-Za-z0-9_]{8,}', content)
            if candidates:
                logger.info(f"Found potential secrets: {candidates}")
                return candidates[0]
            
            return None
        except Exception as e:
            logger.error(f"Error extracting secret: {e}")
            return None

    def extract_question(self, html_content: str) -> str:
        """Extract question text from HTML"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style tags
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text(separator="\n")
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            
            filtered_lines = []
            for line in lines:
                # Don't filter out important instructions
                skip_patterns = [
                    "post to json",
                    "cutoff:",
                    "your secret",
                    "anything you want",
                    "csv file",
                    "content-type",
                    "application/json",
                ]
                
                if any(skip in line.lower() for skip in skip_patterns):
                    continue
                
                if len(line) < 3:
                    continue
                
                if line.upper() in ["POST", "JSON", "CSV", "FORM", "GET"]:
                    continue
                
                if line not in filtered_lines:
                    filtered_lines.append(line)
            
            question_text = "\n".join(filtered_lines)
            
            if not question_text or len(question_text.strip()) < 5:
                return text
            
            return question_text
        except Exception as e:
            logger.error(f"Error extracting question: {e}")
            return html_content

    def run_code(self, code: str) -> str:
        """Execute generated Python code"""
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

    def get_llm_answer(self, instruction: str, page_url: str = "") -> str:
        """Get answer from LLM based on instructions"""
        prompt = f"""You are a quiz solver. Your job is to answer the question based on instructions.

CRITICAL INSTRUCTIONS:
- If you see "Scrape" followed by a URL → that's a SCRAPING task
- If you see "download" → that's a FILE task
- If you see "calculate" → that's a CODE task
- Otherwise → answer the QUESTION directly

PAGE CONTEXT: {page_url}

QUESTION TEXT:
{instruction}

RESPOND WITH ONLY:
- The answer (number, text, or URL)
- Python code if calculation needed (wrapped in ```python ```)
- A URL if scraping/downloading is needed
- NO explanations or markdown

CRITICAL: Never make up answers. Only respond with what the page asks for."""
        
        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()
            logger.info(f"Gemini response: {text[:200]}")
            
            # Handle Python code
            if "```python" in text:
                code_match = re.search(r'```python(.*?)```', text, re.DOTALL)
                if code_match:
                    code = code_match.group(1).strip()
                    logger.info("LLM generated Python code")
                    result = self.run_code(code)
                    logger.info(f"Code result: {result}")
                    return result.strip()
            
            # Clean up response
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
        """Solve a single quiz step"""
        logger.info(f"Processing URL: {url}")
        try:
            # Get HTML (with JS rendering if needed)
            html = self.get_rendered_html(url)
            logger.info(f"Got HTML length: {len(html)}")
            
            # Decode obfuscated content
            readable_html = self.decode_obfuscated_html(html)
            
            # Check if this is a scraping task
            scrape_url = self.extract_scraping_task(readable_html, url)
            
            if scrape_url:
                logger.info(f"Scraping task detected: {scrape_url}")
                # Fetch the scraping URL
                try:
                    scrape_response = requests.get(scrape_url, headers=self.headers, timeout=30)
                    scrape_content = scrape_response.text
                    logger.info(f"Scrape content length: {len(scrape_content)}")
                    
                    # Try to extract secret from scraped content
                    extracted_secret = self.extract_secret_from_content(scrape_content)
                    if extracted_secret:
                        answer = extracted_secret
                        logger.info(f"Extracted secret from scrape: {answer}")
                    else:
                        # If no secret found, ask Gemini about the scraped content
                        answer = self.get_llm_answer(scrape_content[:1000], scrape_url)
                except Exception as e:
                    logger.error(f"Scraping failed: {e}")
                    answer = "1"
            else:
                # Regular question extraction
                question_text = self.extract_question(readable_html)
                logger.info(f"Extracted question: {question_text[:300]}")
                
                answer = self.get_llm_answer(question_text, url)
            
            # Clean answer
            answer = str(answer).strip()
            answer = re.sub(r'```.*?```', '', answer, flags=re.DOTALL).strip()
            answer = re.sub(r'\*\*', '', answer).strip()
            answer = re.sub(r'`', '', answer).strip()
            answer = re.sub(r'\n', ' ', answer).strip()
            
            if answer.lower() in ['error: could not get answer', 'error', 'none', '']:
                logger.warning("Got error answer, using fallback")
                answer = "1"
            
            # Try to convert to numeric if possible
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
            
            # Find submit URL
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
            
            # Handle time limit and retry logic
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
        """Solve entire quiz with retries"""
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
