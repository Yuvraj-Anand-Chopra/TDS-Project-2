import os
import logging
import json
import re
import base64
import io
import subprocess
import sys
from contextlib import redirect_stdout
from typing import Dict, Any, Optional, List
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
        self.resource_cache = {}  # Cache downloaded resources

    def add_dependencies(self, package_name: str) -> str:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            return f"Successfully installed {package_name}"
        except Exception as e:
            return f"Error installing {package_name}: {str(e)}"

    def download_file(self, url: str, filename: str = None) -> str:
        """Download file and return content"""
        try:
            if url in self.resource_cache:
                logger.info(f"Using cached resource: {url}")
                return self.resource_cache[url]
            
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            content = response.text
            
            # Cache it
            self.resource_cache[url] = content
            
            if filename:
                path = self.llm_files_dir / filename
                with open(path, "w") as f:
                    f.write(content)
                logger.info(f"Downloaded and saved: {path}")
            
            logger.info(f"Downloaded from {url}, length: {len(content)}")
            return content
        except Exception as e:
            logger.error(f"Error downloading file: {e}")
            return ""

    def get_rendered_html(self, url: str) -> str:
        """Get HTML using requests"""
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            html = response.text
            logger.info(f"Retrieved HTML from {url}, length: {len(html)}")
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
                result = "\n".join(decoded_parts) + "\n" + html_content
                logger.info(f"Total content after decode: {len(result)} chars")
                return result
            
            return html_content
        except Exception as e:
            logger.error(f"Error decoding: {e}")
            return html_content

    def extract_resources(self, html_content: str, current_url: str) -> List[str]:
        """Extract all resource URLs from HTML (href, src, links, etc)"""
        try:
            resources = []
            
            # href links
            hrefs = re.findall(r'href=["\']([^"\']+)["\']', html_content)
            # src attributes (scripts, images)
            srcs = re.findall(r'src=["\']([^"\']+)["\']', html_content)
            # fetch() calls
            fetches = re.findall(r'fetch\(["\']([^"\']+)["\']', html_content)
            # Plain URLs in text
            plain_urls = re.findall(r'(https?://[^\s"\'<>]+)', html_content)
            
            all_urls = hrefs + srcs + fetches + plain_urls
            
            for url in all_urls:
                # Normalize URL
                if url.startswith('/'):
                    parsed = urlparse(current_url)
                    full_url = f"{parsed.scheme}://{parsed.netloc}{url}"
                elif url.startswith('http'):
                    full_url = url
                else:
                    full_url = urljoin(current_url, url)
                
                # Filter for data/resource URLs
                if any(x in full_url.lower() for x in ['data', 'scrape', 'audio', 'file', '.js', '.json']):
                    if full_url not in resources:
                        resources.append(full_url)
                        logger.info(f"Found resource: {full_url}")
            
            logger.info(f"Total resources found: {len(resources)}")
            return resources
        except Exception as e:
            logger.error(f"Error extracting resources: {e}")
            return []

    def extract_question(self, html_content: str) -> str:
        """Extract question text from HTML"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text(separator="\n")
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            
            filtered_lines = []
            for line in lines:
                skip_patterns = ["post to json", "cutoff:", "your secret", "anything you want", "csv file", "content-type", "application/json"]
                
                if any(skip in line.lower() for skip in skip_patterns):
                    continue
                
                if len(line) < 2:
                    continue
                
                if line.upper() in ["POST", "JSON", "CSV", "FORM", "GET"]:
                    continue
                
                if line not in filtered_lines:
                    filtered_lines.append(line)
            
            question_text = "\n".join(filtered_lines)
            
            if not question_text or len(question_text.strip()) < 5:
                return text
            
            logger.info(f"Extracted question: {question_text[:200]}")
            return question_text
        except Exception as e:
            logger.error(f"Error extracting question: {e}")
            return html_content

    def run_code(self, code: str) -> str:
        """Execute generated Python code"""
        logger.info("Executing Python code...")
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

    def get_llm_answer(self, instruction: str, additional_context: str = "") -> str:
        """Get answer from LLM"""
        context_part = f"\n\nADDITIONAL CONTEXT:\n{additional_context}" if additional_context else ""
        
        prompt = f"""You are a quiz solver. Analyze the content and provide the answer.

RULES:
- Answer ONLY with the required value (number, text, URL, or code)
- Do NOT explain or provide reasoning
- Do NOT use markdown formatting
- If code is needed, wrap in ```python ```
- Calculate sums, values, or perform operations as requested

CONTENT:{context_part}

QUESTION:
{instruction}

ANSWER ONLY:"""
        
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
            return "Error"

    def solve_single_step(self, url: str, email: str, secret: str, retry_count: int = 0) -> Dict[str, Any]:
        """Solve a single quiz step with retry logic"""
        logger.info(f"Processing URL (attempt {retry_count + 1}): {url}")
        try:
            # Get HTML
            html = self.get_rendered_html(url)
            if not html:
                logger.error("Failed to get HTML")
                return {"error": "Failed to get HTML"}
            
            # Decode base64
            readable_html = self.decode_obfuscated_html(html)
            
            # Extract resources
            resources = self.extract_resources(readable_html, url)
            
            # Download and analyze resources
            resource_content = ""
            if resources:
                logger.info(f"Found {len(resources)} resources, downloading...")
                for res_url in resources[:3]:  # Limit to first 3 resources
                    try:
                        res_content = self.download_file(res_url)
                        if res_content:
                            resource_content += f"\n--- Content from {res_url} ---\n{res_content[:2000]}\n"
                    except Exception as e:
                        logger.warning(f"Failed to download {res_url}: {e}")
            
            # Extract question
            question_text = self.extract_question(readable_html)
            
            # Get answer
            answer = self.get_llm_answer(question_text, resource_content)
            
            # Clean answer
            answer = str(answer).strip()
            answer = re.sub(r'```.*?```', '', answer, flags=re.DOTALL).strip()
            answer = re.sub(r'\*\*', '', answer).strip()
            answer = re.sub(r'`', '', answer).strip()
            answer = re.sub(r'\n', ' ', answer).strip()
            
            if not answer or answer.lower() in ['error', 'none', '']:
                logger.warning("Empty answer")
                answer = "1"
            
            # Try to convert to numeric
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
            
            logger.info(f"Submitting to: {submit_url}")
            
            payload = {"email": email, "secret": secret, "url": url, "answer": final_answer}
            logger.info(f"Payload: {json.dumps(payload, indent=2)}")
            
            submit_resp = requests.post(submit_url, json=payload, timeout=30)
            logger.info(f"Response status: {submit_resp.status_code}")
            logger.info(f"Response text: {submit_resp.text}")
            
            data = submit_resp.json()
            delay = data.get("delay", 0)
            correct = data.get("correct", False)
            
            # Handle response
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
        """Solve entire quiz"""
        current_url = url
        iteration = 0
        max_iterations = 10
        
        while iteration < max_iterations:
            try:
                logger.info(f"=== Iteration {iteration + 1} ===")
                result = self.solve_single_step(current_url, email, secret, iteration)
                
                if isinstance(result, dict) and result.get("correct") is True:
                    next_url = result.get("url")
                    if next_url:
                        logger.info(f"✓ Correct! Moving to next task")
                        current_url = next_url
                        iteration += 1
                        continue
                    else:
                        logger.info("✓ Quiz completed!")
                        return {"status": "completed", "message": "All quiz tasks solved successfully"}
                else:
                    logger.warning(f"✗ Failed: {result}")
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
