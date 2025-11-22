import os
import logging
import json
import re
import base64
import sys
import traceback
import io
from contextlib import redirect_stdout
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse

import requests
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY is missing!")

genai.configure(api_key=GEMINI_API_KEY)

# --- DATA MODELS ---
class QuizRequest(BaseModel):
    email: str
    secret: str
    url: HttpUrl

# --- CORE LOGIC ---

class TaskSolver:
    def __init__(self):
        self.model = genai.GenerativeModel("gemini-2.0-flash")
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

    def decode_obfuscated_html(self, html_content: str) -> str:
        """
        Simulates 'document.body.innerHTML = atob(...)' by finding base64 strings
        in scripts and decoding them.
        """
        try:
            # Find all atob("...") patterns
            matches = re.findall(r'atob\([`"\']([A-Za-z0-9+/=\\s]+)[`"\']\)', html_content)
            
            decoded_parts = []
            for match in matches:
                try:
                    # Clean whitespace that might be in the JS string
                    clean_match = re.sub(r'\s+', '', match)
                    decoded_bytes = base64.b64decode(clean_match)
                    decoded_str = decoded_bytes.decode('utf-8')
                    decoded_parts.append(decoded_str)
                except Exception as e:
                    logger.warning(f"Failed to decode base64 segment: {e}")

            # If we found decoded parts, assume they replace or append to the body
            if decoded_parts:
                return "\n".join(decoded_parts) + "\n" + html_content
            
            return html_content
        except Exception:
            return html_content

    def execute_python_code(self, code: str) -> str:
        """
        Executes LLM-generated Python code safely to handle data tasks.
        """
        logger.info("Executing generated Python code...")
        try:
            # Create a string buffer to capture stdout
            f = io.StringIO()
            
            # Define safe globals - imports that the code might need
            # We import them here so the exec environment has access to them
            import pandas as pd
            import numpy as np
            import requests
            import json
            import pypdf
            
            local_vars = {}
            global_vars = {
                "pd": pd,
                "numpy": np,
                "np": np,
                "requests": requests,
                "json": json,
                "pypdf": pypdf,
                "print": print
            }

            with redirect_stdout(f):
                exec(code, global_vars, local_vars)
            
            output = f.getvalue().strip()
            return output if output else "Code executed but returned no output."

        except Exception:
            return f"Execution Error: {traceback.format_exc()}"

    def get_llm_answer(self, instruction: str) -> str:
        """
        Asks Gemini to solve the task. 
        If it's a data task, asks Gemini to write Python code, then executes it.
        """
        prompt = f"""
        You are an expert automated agent. 
        
        TASK:
        {instruction}
        
        INSTRUCTIONS:
        1. If the answer is directly available in the text, output ONLY the answer.
        2. If the task requires calculation, downloading files, parsing CSV/PDFs, or filtering data:
           Write a PYTHON SCRIPT to do it.
           - Use 'requests' to download.
           - Use 'pandas' for data.
           - PRINT the final answer at the end of the script.
           - Wrap the code in python markdown blocks (triple backticks).
        
        Output ONLY the Answer or the Python Code.
        """
        
        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()
            
            # Check for markdown code blocks safely
            marker = "```
            if marker in text:
                # Regex to find content between ```python and ```
                code_match = re.search(r'```python(.*?)```
                if code_match:
                    code = code_match.group(1)
                    logger.info("LLM generated Python code. Executing...")
                    result = self.execute_python_code(code)
                    logger.info(f"Code execution result: {result}")
                    return result.strip()
            
            return text
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            return "Error generating answer"

    def solve_single_step(self, url: str, email: str, secret: str) -> Dict:
        logger.info(f"Processing URL: {url}")
        
        # 1. Fetch Page
        resp = requests.get(url, headers=self.headers)
        resp.raise_for_status()
        
        # 2. Decode Content (Handle JS obfuscation)
        readable_html = self.decode_obfuscated_html(resp.text)
        
        # 3. Extract Instructions
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(readable_html, 'html.parser')
        visible_text = soup.get_text("\n").strip()
        
        # 4. Solve
        answer = self.get_llm_answer(visible_text)
        
        # Attempt to fix type (LLM returns string, but sometimes we need int/float)
        final_answer = answer
        # Heuristic: if it looks like a number, convert it
        if str(answer).replace('.', '', 1).isdigit():
            if '.' in str(answer):
                try:
                    final_answer = float(answer)
                except:
                    pass
            else:
                try:
                    final_answer = int(answer)
                except:
                    pass
        
        # 5. Find Submission URL
        submit_url = None
        # Try to find specific submit link in text
        url_match = re.search(r'https?://[^\s"\']+/submit', readable_html)
        if url_match:
            submit_url = url_match.group(0)
        else:
            # Fallback: assume /submit on same host
            parsed = urlparse(url)
            submit_url = f"{parsed.scheme}://{parsed.netloc}/submit"
            
        logger.info(f"Submitting '{final_answer}' to {submit_url}")
        
        # 6. Submit
        payload = {
            "email": email,
            "secret": secret,
            "url": url,
            "answer": final_answer
        }
        
        submit_resp = requests.post(submit_url, json=payload)
        try:
            return submit_resp.json()
        except:
            return {"error": submit_resp.text, "status": submit_resp.status_code}


# --- API SETUP ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

solver = TaskSolver()

@app.post("/solve-quiz")
async def solve_quiz_endpoint(request: Request):
    # 1. Validate JSON Body manually to handle 400 Bad Request explicitly
    try:
        body = await request.json()
        req_data = QuizRequest(**body)
    except Exception:
        # Requirement: Respond with HTTP 400 for invalid JSON
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    # 2. Security Check
    # Requirement: Respond with HTTP 403 for invalid secrets
    EXPECTED_SECRET = os.getenv("SECRET")
    if EXPECTED_SECRET and req_data.secret != EXPECTED_SECRET:
        raise HTTPException(status_code=403, detail="Invalid Secret")

    # 3. Recursive Solving Loop
    current_url = str(req_data.url)
    email = req_data.email
    secret = req_data.secret
    
    iteration = 0
    max_iterations = 10 # Safety break
    
    last_response = {}
    
    while iteration < max_iterations:
        try:
            logger.info(f"--- Iteration {iteration + 1} ---")
            result = solver.solve_single_step(current_url, email, secret)
            last_response = result
            
            # Check if we need to continue
            # The API returns {"correct": true, "url": "..."} if there is another step
            if isinstance(result, dict) and result.get("correct") is True:
                next_url = result.get("url")
                if next_url:
                    logger.info(f"Correct! Moving to next URL: {next_url}")
                    current_url = next_url
                    iteration += 1
                    continue
                else:
                    logger.info("Quiz Completed Successfully!")
                    break
            else:
                # Wrong answer or error, return what we got
                logger.warning(f"Submission failed or wrong answer: {result}")
                break
                
        except Exception as e:
            logger.error(f"Error in loop: {e}")
            return JSONResponse(
                status_code=500, 
                content={"error": str(e)}
            )

    # Return the final result to the evaluator
    return last_response

@app.get("/health")
def health():
    return {"status": "ok", "model": "gemini-2.0-flash"}
