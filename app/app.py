"""
CONSOLIDATED AI AGENT - Production Ready
Integrates: OCR, Audio, Web Scraping, Code Execution, and Smart Retry Logic.
"""

import os
import sys
import time
import json
import uuid
import base64
import logging
import math
import hashlib
import subprocess
import requests
import io
import re
from typing import TypedDict, Annotated, List, Any, Dict, Optional
from collections import defaultdict
from contextlib import redirect_stdout
from urllib.parse import urljoin

# Web Server
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import uvicorn

# AI & Graph
from langchain_core.tools import tool
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import trim_messages, HumanMessage
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_google_genai import HarmBlockThreshold, HarmCategory

# Multimedia Processing
try:
    import pytesseract
    from PIL import Image
    import speech_recognition as sr
    from pydub import AudioSegment
    from playwright.sync_api import sync_playwright
    from bs4 import BeautifulSoup
except ImportError as e:
    print(f"Warning: Optional dependency missing: {e}")

# ============================================================================
# CONFIGURATION
# ============================================================================
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Agent")

EMAIL = os.getenv("EMAIL")
SECRET = os.getenv("SECRET")
EXPECTED_SECRET = os.getenv("SECRET")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Shared State (In-Memory Database)
BASE64_STORE = {}  # Stores large image data
URL_TIME = {}      # Tracks when we last saw a URL
CACHE = defaultdict(int) # Tracks retries per URL

RETRY_LIMIT = 4

# ============================================================================
# TOOLS
# ============================================================================

@tool
def get_rendered_html(url: str) -> dict:
    """
    Fetch and return the fully rendered HTML of a webpage using Playwright.
    """
    print(f"\nFetching: {url}")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            try:
                page.goto(url, wait_until="networkidle", timeout=60000)
            except Exception:
                pass # Continue even if networkidle times out
            content = page.content()
            browser.close()

            # Truncate to prevent token overflow
            if len(content) > 200000:
                print("HTML too large, truncating...")
                content = content[:200000] + "... [TRUNCATED]"
            
            # Extract images for the LLM's awareness
            imgs = []
            if 'BeautifulSoup' in globals():
                soup = BeautifulSoup(content, "html.parser")
                imgs = [urljoin(url, img["src"]) for img in soup.find_all("img", src=True)]

            return {"html": content, "images": imgs, "url": url}
    except Exception as e:
        return {"error": f"Scraping failed: {str(e)}"}

@tool
def download_file(url: str, filename: str) -> str:
    """
    Download a file from a URL and save it to 'LLMFiles/'.
    """
    try:
        print(f"\nDownloading: {url}")
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()
        
        os.makedirs("LLMFiles", exist_ok=True)
        path = os.path.join("LLMFiles", filename)
        
        with open(path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        return f"Saved to {path}"
    except Exception as e:
        return f"Download failed: {str(e)}"

@tool
def run_code(code: str) -> dict:
    """
    Execute Python code.
    Pre-loads libraries: pandas, numpy, hashlib, base64, requests, bs4.
    """
    try:
        print(f"\nExecuting Code:\n{code[:150]}...")
        os.makedirs("LLMFiles", exist_ok=True)
        
        f_out = io.StringIO()
        f_err = io.StringIO()
        
        # Inject dependencies directly into the scope
        local_vars = {}
        global_vars = {
            "pd": __import__('pandas'),
            "np": __import__('numpy'),
            "requests": requests,
            "json": json,
            "print": print,
            "BeautifulSoup": BeautifulSoup,
            "os": os,
            "hashlib": hashlib,
            "math": math,
            "base64": base64,
            "re": re,
            "random": __import__('random')
        }
        
        with redirect_stdout(f_out):
            try:
                # Switch CWD so code can find files easily
                old_cwd = os.getcwd()
                os.chdir("LLMFiles")
                exec(code, global_vars, local_vars)
            except Exception as e:
                f_err.write(str(e))
            finally:
                os.chdir(old_cwd)
        
        stdout = f_out.getvalue().strip()
        stderr = f_err.getvalue().strip()
        
        return {
            "stdout": stdout[:5000] if len(stdout) > 5000 else stdout,
            "stderr": stderr,
            "return_code": 0 if not stderr else 1
        }
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "return_code": -1}

@tool
def transcribe_audio(file_path: str) -> str:
    """
    Transcribe .mp3 or .wav files to text.
    """
    try:
        if not file_path.startswith("LLMFiles") and not os.path.exists(file_path):
             file_path = os.path.join("LLMFiles", file_path)

        final_path = file_path
        
        # Convert MP3 -> WAV
        if file_path.lower().endswith(".mp3"):
            sound = AudioSegment.from_mp3(file_path)
            final_path = file_path.replace(".mp3", ".wav")
            sound.export(final_path, format="wav")

        recognizer = sr.Recognizer()
        with sr.AudioFile(final_path) as source:
            audio_data = recognizer.record(source)
            # Use Google Web Speech API (Free tier)
            text = recognizer.recognize_google(audio_data)

        if final_path != file_path and os.path.exists(final_path):
            os.remove(final_path)

        return f"Transcription: {text}"
    except Exception as e:
        return f"Audio Error: {e}"

@tool
def ocr_image_tool(image_path: str) -> str:
    """
    Extract text from an image using OCR (Tesseract).
    """
    try:
        if not image_path.startswith("LLMFiles") and not os.path.exists(image_path):
             image_path = os.path.join("LLMFiles", image_path)
             
        img = Image.open(image_path).convert("RGB")
        text = pytesseract.image_to_string(img)
        return f"OCR Result: {text.strip()}"
    except Exception as e:
        return f"OCR Error: {e}"

@tool
def encode_image_to_base64(image_path: str) -> str:
    """
    Convert image to Base64. Returns a KEY (BASE64_KEY:...) instead of the full string
    to prevent crashing the LLM context.
    """
    try:
        if not image_path.startswith("LLMFiles") and not os.path.exists(image_path):
             image_path = os.path.join("LLMFiles", image_path)

        with open(image_path, "rb") as f:
            raw = f.read()
    
        encoded = base64.b64encode(raw).decode("utf-8")
        key = str(uuid.uuid4())
        BASE64_STORE[key] = encoded # Save to global memory

        return f"BASE64_KEY:{key}"
    except Exception as e:
        return f"Encoding Error: {e}"

@tool
def post_request(url: str, payload: Dict[str, Any]) -> Any:
    """
    Submit answer. Handles Base64 injection and Retry Logic.
    """
    headers = {"Content-Type": "application/json"}
    
    # 1. Inject Base64 data if placeholder is found
    ans = payload.get("answer")
    if isinstance(ans, str) and ans.startswith("BASE64_KEY:"):
        key = ans.split(":", 1)[1]
        if key in BASE64_STORE:
            payload["answer"] = BASE64_STORE[key]
            print(" injected Base64 data from store.")
    
    try:
        cur_url = os.getenv("url", "unknown")
        CACHE[cur_url] += 1
        
        # Log preview
        log_pl = payload.copy()
        if isinstance(log_pl.get("answer"), str) and len(log_pl["answer"]) > 50:
             log_pl["answer"] = log_pl["answer"][:50] + "..."
        print(f"\nPOST {url}\nPayload: {json.dumps(log_pl)}")

        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        # 2. Friend's Retry Logic
        next_url = data.get("url")
        if not next_url:
            return "Tasks completed"
            
        if next_url not in URL_TIME:
            URL_TIME[next_url] = time.time()

        if not data.get("correct"):
            cur_time = time.time()
            prev_time = URL_TIME.get(cur_url, time.time())
            delay = cur_time - prev_time
            
            # If we've retried too many times or it's been too long, force move on
            if CACHE[cur_url] >= RETRY_LIMIT or delay >= 180:
                 print(f"Give up on {cur_url} (Attempts: {CACHE[cur_url]}, Delay: {delay:.1f}s)")
            else:
                 print("Answer incorrect. Retrying...")
                 data["message"] = "Incorrect answer. Analyze the data again and retry."

        # Update environment for next step
        os.environ["url"] = str(data.get("url", ""))
        return data

    except Exception as e:
        return f"HTTP Error: {str(e)}"

# ============================================================================
# AGENT SETUP
# ============================================================================

ALL_TOOLS = [
    get_rendered_html, 
    download_file, 
    run_code, 
    post_request, 
    transcribe_audio, 
    ocr_image_tool, 
    encode_image_to_base64
]

# Safety to prevent blocking
safety_settings = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

# Rate limit to avoid 429 errors (5 req/min)
rate_limiter = InMemoryRateLimiter(
    requests_per_second=5/60,
    check_every_n_seconds=1,
    max_bucket_size=5
)

# Using gemini-1.5-flash as default, essentially "Standard"
llm = init_chat_model(
    model_provider="google_genai",
    model="gemini-1.5-flash", 
    rate_limiter=rate_limiter,
    safety_settings=safety_settings
).bind_tools(ALL_TOOLS)

SYSTEM_PROMPT = f"""You are an autonomous agent solving web challenges.

GOAL:
1. Access the URL.
2. Analyze content (Text, Images, Audio).
3. Solve the puzzle.
4. Submit via `post_request`.

RULES:
- Files are in 'LLMFiles/'.
- Use `transcribe_audio` for .mp3/.wav.
- Use `ocr_image_tool` for images containing text.
- Use `encode_image_to_base64` if the answer requires an image file submission.
- Use `run_code` for math/data processing. `pandas`, `hashlib` are pre-installed.
- Identity: Email: {EMAIL}, Secret: {SECRET}

Start by fetching the page."""

class AgentState(TypedDict):
    messages: Annotated[List, add_messages]

def agent_node(state: AgentState):
    # Context Trimming
    trimmed = trim_messages(
        messages=state["messages"],
        max_tokens=50000,
        strategy="last",
        include_system=True,
        start_on="human",
        token_counter=llm
    )
    # Ensure system prompt is present
    if not any(isinstance(m, HumanMessage) for m in trimmed):
         trimmed = [HumanMessage(content=f"Resume solving: {os.getenv('url')}")] + trimmed
         
    result = llm.invoke(trimmed)
    return {"messages": [result]}

def route(state):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    if "Tasks completed" in str(last.content):
        return END
    return "agent"

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(ALL_TOOLS))
graph.add_edge(START, "agent")
graph.add_edge("tools", "agent")
graph.add_conditional_edges("agent", route, {"tools": "tools", "agent": "agent", END: END})
agent_runner = graph.compile()

# ============================================================================
# SERVER
# ============================================================================

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

START_TIME = time.time()

async def background_task(url: str):
    print(f"Starting task: {url}")
    # Reset State
    BASE64_STORE.clear()
    URL_TIME.clear()
    CACHE.clear()
    os.environ["url"] = url
    URL_TIME[url] = time.time()
    
    agent_runner.invoke(
        {"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Start here: {url}"}
        ]},
        config={"recursion_limit": 500}
    )
    print("Task finished.")

@app.post("/solve")
async def solve(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    url = data.get("url")
    secret = data.get("secret")
    
    if not url or secret != EXPECTED_SECRET:
        raise HTTPException(400, "Invalid inputs")
        
    background_tasks.add_task(background_task, url)
    return {"status": "ok", "message": "Agent started"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 1010))
    uvicorn.run(app, host="0.0.0.0", port=port)
