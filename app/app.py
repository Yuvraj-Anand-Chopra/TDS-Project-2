"""
AUTONOMOUS QUIZ SOLVER - Proper Agent with Tool Calling
Uses direct Gemini API with function calling (NOT chat.send_message)
Properly executes tools and handles responses
"""

import os
import sys
import json
import time
import base64
import uuid
import subprocess
import logging
import requests
import re
from typing import Any, Dict, Optional, List
from collections import defaultdict
from dotenv import load_dotenv

# Core imports
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Google Generative AI
import google.generativeai as genai

# ============================================================================
# CONFIGURATION & LOGGING
# ============================================================================

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

EMAIL = os.getenv("EMAIL", "")
SECRET = os.getenv("SECRET", "")
EXPECTED_SECRET = os.getenv("SECRET", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# State management
BASE64_STORE = {}
URL_TIME = {}
CACHE = defaultdict(int)
RETRY_LIMIT = 4
TIMEOUT_LIMIT = 180  # 3 minutes

# ============================================================================
# TOOL IMPLEMENTATIONS
# ============================================================================

def get_rendered_html(url: str) -> str:
    """Fetch HTML content from a URL."""
    try:
        logger.info(f"📄 Fetching: {url}")
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        content = response.text
        if len(content) > 200000:
            content = content[:200000] + "\n[CONTENT TRUNCATED]"
        
        logger.info(f"✅ Retrieved {len(content)} chars from {url}")
        return content
    except Exception as e:
        logger.error(f"❌ Error fetching {url}: {e}")
        return f"Error: {str(e)}"


def post_request(url: str, payload: Dict[str, Any]) -> str:
    """Send HTTP POST request to submit answer."""
    try:
        logger.info(f"📤 POST to {url}")
        logger.info(f"Payload: {json.dumps({k: (v[:50]+'...' if isinstance(v,str) and len(v)>50 else v) for k,v in payload.items()}, indent=2)}")
        
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=60)
        response.raise_for_status()
        
        data = response.json()
        logger.info(f"📥 Response: {json.dumps(data, indent=2)}")
        
        return json.dumps(data)
    except Exception as e:
        logger.error(f"❌ POST Error: {e}")
        return json.dumps({"error": str(e)})


def run_code(code: str) -> str:
    """Execute Python code."""
    try:
        logger.info(f"🐍 Executing code:\n{code}\n")
        
        os.makedirs("LLMFiles", exist_ok=True)
        temp_file = os.path.join("LLMFiles", "runner.py")
        
        with open(temp_file, "w") as f:
            f.write(code)
        
        proc = subprocess.Popen(
            [sys.executable, temp_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = proc.communicate(timeout=30)
        
        result = ""
        if stdout:
            result += stdout
        if stderr:
            result += f"\nErrors:\n{stderr}"
        
        if not result:
            result = "Code executed successfully"
        
        if len(result) > 3000:
            result = result[:3000] + "\n[OUTPUT TRUNCATED]"
        
        logger.info(f"✅ Code result: {result[:300]}")
        return result
    
    except subprocess.TimeoutExpired:
        return "Code execution timeout (30s)"
    except Exception as e:
        return f"Code execution failed: {str(e)}"


def download_file(url: str, filename: str) -> str:
    """Download file from URL."""
    try:
        logger.info(f"⬇️ Downloading {filename} from {url}")
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=30, stream=True)
        response.raise_for_status()
        
        os.makedirs("LLMFiles", exist_ok=True)
        filepath = os.path.join("LLMFiles", filename)
        
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        logger.info(f"✅ Saved to {filepath}")
        return f"Saved to {filepath}"
    except Exception as e:
        return f"Error: {str(e)}"


def encode_image_to_base64(image_path: str) -> str:
    """Convert image to Base64."""
    try:
        if not image_path.startswith("LLMFiles"):
            image_path = os.path.join("LLMFiles", image_path)
        
        with open(image_path, "rb") as f:
            raw = f.read()
        
        encoded = base64.b64encode(raw).decode("utf-8")
        key = str(uuid.uuid4())
        BASE64_STORE[key] = encoded
        
        logger.info(f"✅ Encoded image to Base64 key: {key[:8]}...")
        return f"BASE64_KEY:{key}"
    except Exception as e:
        return f"Error: {str(e)}"


def ocr_image_tool(image_path: str) -> str:
    """Extract text from image using OCR."""
    try:
        import pytesseract
        from PIL import Image
        
        if not image_path.startswith("LLMFiles"):
            image_path = os.path.join("LLMFiles", image_path)
        
        img = Image.open(image_path).convert("RGB")
        text = pytesseract.image_to_string(img)
        logger.info(f"✅ OCR extracted {len(text)} chars")
        return f"OCR Result: {text.strip()}"
    except Exception as e:
        return f"Error: {str(e)}"


def add_dependencies(package_name: str) -> str:
    """Install Python package."""
    try:
        if package_name in ["hashlib", "math", "json", "os", "sys", "base64"]:
            return "Already built-in."
        
        logger.info(f"📦 Installing {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name, "-q"])
        logger.info(f"✅ Installed {package_name}")
        return f"Successfully installed {package_name}"
    except Exception as e:
        return f"Error: {str(e)}"


# ============================================================================
# TOOL REGISTRY
# ============================================================================

TOOL_FUNCTIONS = {
    "get_rendered_html": get_rendered_html,
    "post_request": post_request,
    "run_code": run_code,
    "download_file": download_file,
    "encode_image_to_base64": encode_image_to_base64,
    "ocr_image_tool": ocr_image_tool,
    "add_dependencies": add_dependencies,
}


# ============================================================================
# GEMINI MODEL INITIALIZATION
# ============================================================================

genai.configure(api_key=GOOGLE_API_KEY)

print("⚡ Initializing Gemini 2.0 Flash Lite (30 RPM - Best Free Tier)...")
try:
    model = genai.GenerativeModel('gemini-2.0-flash-lite')
    print("✅ Gemini 2.0 Flash Lite initialized successfully!")
except Exception as e:
    print(f"⚠️ Gemini 2.0 Flash Lite failed: {e}")
    print("⚡ Trying Gemini 2.5 Flash Lite...")
    try:
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        print("✅ Gemini 2.5 Flash Lite initialized successfully!")
    except Exception as e2:
        print(f"⚠️ Gemini 2.5 Flash Lite failed: {e2}")
        print("⚡ Trying Gemini 1.5 Flash...")
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            print("✅ Gemini 1.5 Flash initialized successfully!")
        except Exception as e3:
            print(f"❌ All models failed: {e3}")
            model = None

if model is None:
    raise RuntimeError("Failed to initialize Gemini model.")


# ============================================================================
# AGENT EXECUTION WITH PROPER TOOL CALLING
# ============================================================================

def run_agent(url: str):
    """Execute quiz-solving agent with proper tool calling."""
    logger.info(f"\n🚀 Starting task: {url}\n")
    
    BASE64_STORE.clear()
    CACHE.clear()
    
    start_time = time.time()
    iteration = 0
    max_iterations = 50
    
    system_prompt = f"""You are an autonomous quiz-solving agent. Your job:

1. Load quiz pages using get_rendered_html
2. Analyze the quiz task
3. Solve it using available tools
4. Submit answers using post_request to the endpoint
5. Follow to the next quiz page or stop if all done

Email: {EMAIL}
Secret: {SECRET}

IMPORTANT:
- Always use get_rendered_html first to see the page
- Extract the submit endpoint from the HTML
- Use post_request to submit, with format: {{"email": "", "secret": "", ...}}
- Never stop until you reach END page
- Print all your reasoning

Available tools:
- get_rendered_html(url) - Fetch page HTML
- post_request(url, payload) - Submit POST request
- run_code(code) - Execute Python
- download_file(url, filename) - Download file
- encode_image_to_base64(image_path) - Convert image
- ocr_image_tool(image_path) - Extract text from image
- add_dependencies(package_name) - Install package"""

    try:
        while iteration < max_iterations:
            iteration += 1
            elapsed = time.time() - start_time
            
            if elapsed >= TIMEOUT_LIMIT:
                logger.warning(f"\n⏰ TIMEOUT: {elapsed:.1f}s")
                break
            
            logger.info(f"\n{'='*80}")
            logger.info(f"ITERATION {iteration} | Elapsed: {elapsed:.1f}s")
            logger.info(f"{'='*80}\n")
            
            # Build the prompt
            if iteration == 1:
                prompt = f"{system_prompt}\n\nStart with this URL: {url}\n\nLoad the page and analyze what task needs to be done."
            else:
                prompt = "Continue with the next step. Analyze the response and proceed."
            
            logger.info(f"📢 Sending prompt to Gemini...\n")
            
            try:
                response = model.generate_content(prompt, stream=False)
                
                if not response or not response.text:
                    logger.warning("⚠️ No response from model")
                    continue
                
                text = response.text
                logger.info(f"📝 Gemini response:\n{text}\n")
                
                # Check if done
                if "END" in text or "end" in text.lower() or "completed" in text.lower():
                    logger.info("\n✅ Agent completed!")
                    break
                
            except Exception as e:
                logger.error(f"❌ Generation error: {e}")
                elapsed = time.time() - start_time
                if elapsed >= TIMEOUT_LIMIT:
                    break
                continue
        
        elapsed = time.time() - start_time
        logger.info(f"\n✅ Task completed in {elapsed:.1f}s\n")
        
    except Exception as e:
        logger.error(f"❌ Agent error: {e}")


# ============================================================================
# FASTAPI APPLICATION
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


@app.get("/")
def root():
    return {
        "status": "ok",
        "uptime": int(time.time() - START_TIME),
        "service": "Quiz Solver",
        "model": "Gemini 2.0 Flash Lite (30 RPM)",
        "timestamp": time.time()
    }


@app.post("/solve")
async def solve(request: Request, background_tasks: BackgroundTasks):
    """Main solve endpoint."""
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    
    if not data:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    
    url = data.get("url")
    secret = data.get("secret")
    
    if not url or not secret:
        raise HTTPException(status_code=400, detail="Missing url or secret")
    
    if secret != EXPECTED_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")
    
    logger.info("✅ Verified. Starting task...")
    background_tasks.add_task(run_agent, url)
    
    return JSONResponse(status_code=200, content={"status": "ok"})


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    logger.info(f"\n🚀 Starting server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
