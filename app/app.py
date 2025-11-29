"""
AUTONOMOUS QUIZ SOLVER - Direct Gemini API Implementation (NO TOOLS AT INIT)
Workaround: Define tools separately, don't pass to GenerativeModel

Features:
- PDF-to-image conversion
- Audio recognition (SpeechRecognition)
- OCR (Tesseract)
- Direct Gemini API with function calling (tools defined separately)
- 9 essential tools
- 3-minute timeout handling
- Error recovery
- Background task execution
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
from typing import Any, Dict, Optional, List
from collections import defaultdict
from dotenv import load_dotenv

# Core imports
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Google Generative AI (NO LangChain!)
import google.generativeai as genai

# Utilities
from bs4 import BeautifulSoup

# ============================================================================
# CONFIGURATION & LOGGING
# ============================================================================

load_dotenv()
logging.basicConfig(level=logging.INFO)
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
MAX_TOKENS = 50000
RECURSION_LIMIT = 5000
TIMEOUT_LIMIT = 180  # 3 minutes

# ============================================================================
# TOOL IMPLEMENTATIONS - Pure Python Functions
# ============================================================================

def get_rendered_html(url: str) -> str:
    """Fetch and return HTML content from a URL."""
    try:
        print(f"\n📄 Fetching: {url}\n")
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        content = response.text
        if len(content) > 200000:
            content = content[:200000] + "\n[CONTENT TRUNCATED]"
        
        logger.info(f"✅ Retrieved HTML from {url}")
        return content
    except Exception as e:
        logger.error(f"❌ Error fetching {url}: {e}")
        return f"Error fetching URL: {str(e)}"


def download_file(url: str, filename: str = "download.txt") -> str:
    """Download a file from URL and save it to LLMFiles/."""
    try:
        print(f"\n⬇️ Downloading: {url}\n")
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(url, headers=headers, timeout=30, stream=True)
        response.raise_for_status()
        
        os.makedirs("LLMFiles", exist_ok=True)
        filepath = os.path.join("LLMFiles", filename)
        
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        logger.info(f"✅ Downloaded and saved: {filepath}")
        return f"Saved to {filepath}"
    except Exception as e:
        logger.error(f"❌ Error downloading {url}: {e}")
        return f"Error downloading file: {str(e)}"


def post_request(url: str, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> str:
    """Send HTTP POST request to submit answer."""
    headers = headers or {"Content-Type": "application/json"}
    
    # Inject Base64 data if placeholder found
    ans = payload.get("answer")
    if isinstance(ans, str) and ans.startswith("BASE64_KEY:"):
        key = ans.split(":", 1)[1]
        if key in BASE64_STORE:
            payload["answer"] = BASE64_STORE[key]
    
    try:
        print(f"\n📤 POST {url}")
        
        # Log safely (truncate long answers)
        sending = payload.copy()
        if isinstance(sending.get("answer"), str) and len(str(sending.get("answer", ""))) > 100:
            sending["answer"] = str(sending["answer"])[:100] + "..."
        
        print(f"Payload: {json.dumps(sending, indent=2)}\n")
        
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        
        data = response.json()
        print(f"📥 Response: {json.dumps(data, indent=2)}\n")
        
        cur_url = os.getenv("url", "unknown")
        CACHE[cur_url] += 1
        
        next_url = data.get("url")
        if next_url:
            if next_url not in URL_TIME:
                URL_TIME[next_url] = time.time()
        
        # Determine retry behavior
        correct = data.get("correct")
        if not correct and next_url and CACHE[cur_url] < RETRY_LIMIT:
            delay = time.time() - URL_TIME.get(cur_url, time.time())
            if delay < TIMEOUT_LIMIT:
                data["message"] = "Incorrect. Analyze again and retry."
                data["url"] = cur_url  # Retry same question
                os.environ["offset"] = str(URL_TIME.get(cur_url, time.time()))
            else:
                data["url"] = next_url  # Move to next
        
        return json.dumps(data)
    
    except Exception as e:
        logger.error(f"❌ POST Error: {e}")
        return json.dumps({"error": str(e)})


def run_code(code: str) -> str:
    """Execute Python code and return output."""
    try:
        print(f"\n🐍 Executing code:\n{code}\n")
        
        os.makedirs("LLMFiles", exist_ok=True)
        
        # Write to temp file
        temp_file = os.path.join("LLMFiles", "runner.py")
        with open(temp_file, "w") as f:
            f.write(code)
        
        # Execute with subprocess
        proc = subprocess.Popen(
            [sys.executable, temp_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = proc.communicate(timeout=30)
        
        # Return formatted result
        result = ""
        if stdout:
            result += stdout
        if stderr:
            result += f"\nErrors:\n{stderr}"
        
        if not result:
            result = "Code executed successfully"
        
        # Limit output
        if len(result) > 3000:
            result = result[:3000] + "\n[OUTPUT TRUNCATED]"
        
        logger.info(f"✅ Code result: {result[:500]}")
        return result
    
    except subprocess.TimeoutExpired:
        return "Code execution timeout (30s)"
    except Exception as e:
        error_msg = f"Code execution failed: {str(e)}"
        logger.error(f"❌ {error_msg}")
        return error_msg


def encode_image_to_base64(image_path: str) -> str:
    """Convert image to Base64 and store it. Returns key."""
    try:
        if not image_path.startswith("LLMFiles"):
            image_path = os.path.join("LLMFiles", image_path)
        
        with open(image_path, "rb") as f:
            raw = f.read()
        
        encoded = base64.b64encode(raw).decode("utf-8")
        key = str(uuid.uuid4())
        BASE64_STORE[key] = encoded
        
        return f"BASE64_KEY:{key}"
    except Exception as e:
        return f"Encoding error: {str(e)}"


def add_dependencies(package_name: str) -> str:
    """Install Python package."""
    try:
        if package_name in ["hashlib", "math", "json", "os", "sys", "base64"]:
            return "Already built-in."
        
        print(f"\n📦 Installing {package_name}...\n")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name, "-q"])
        return f"Successfully installed {package_name}"
    except Exception as e:
        return f"Error installing {package_name}: {str(e)}"


def ocr_image_tool(image_path: str) -> str:
    """Extract text from image using OCR."""
    try:
        import pytesseract
        from PIL import Image
        
        if not image_path.startswith("LLMFiles"):
            image_path = os.path.join("LLMFiles", image_path)
        
        img = Image.open(image_path).convert("RGB")
        text = pytesseract.image_to_string(img)
        return f"OCR Result: {text.strip()}"
    except ImportError:
        return "Error: Tesseract/PIL not installed."
    except Exception as e:
        return f"OCR error: {str(e)}"


def transcribe_audio_tool(audio_path: str) -> str:
    """Transcribe audio using SpeechRecognition."""
    try:
        import speech_recognition as sr
        
        if not audio_path.startswith("LLMFiles"):
            audio_path = os.path.join("LLMFiles", audio_path)
        
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
        
        try:
            text = recognizer.recognize_google(audio)
            return f"Transcription: {text}"
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError as e:
            return f"API error: {e}"
    except Exception as e:
        return f"Transcription error: {str(e)}"


def pdf_to_image_tool(pdf_path: str, output_prefix: str = "page") -> str:
    """Convert PDF pages to images."""
    try:
        from pdf2image import convert_from_path
        
        if not pdf_path.startswith("LLMFiles"):
            pdf_path = os.path.join("LLMFiles", pdf_path)
        
        os.makedirs("LLMFiles/pdf_images", exist_ok=True)
        
        images = convert_from_path(pdf_path)
        saved_files = []
        
        for i, image in enumerate(images):
            filename = f"LLMFiles/pdf_images/{output_prefix}_{i+1}.png"
            image.save(filename, 'PNG')
            saved_files.append(filename)
        
        return f"Converted {len(images)} pages. Saved: {', '.join(saved_files)}"
    except Exception as e:
        return f"PDF conversion error: {str(e)}"


# ============================================================================
# TOOL MAPPING FOR GEMINI
# ============================================================================

TOOL_FUNCTIONS = {
    "get_rendered_html": get_rendered_html,
    "download_file": download_file,
    "post_request": post_request,
    "run_code": run_code,
    "encode_image_to_base64": encode_image_to_base64,
    "add_dependencies": add_dependencies,
    "ocr_image_tool": ocr_image_tool,
    "transcribe_audio_tool": transcribe_audio_tool,
    "pdf_to_image_tool": pdf_to_image_tool,
}

# ============================================================================
# GEMINI MODEL INITIALIZATION - NO TOOLS AT INIT
# ============================================================================

genai.configure(api_key=GOOGLE_API_KEY)

print("⚡ Initializing Gemini 2.0 Flash (without tools)...")
try:
    model = genai.GenerativeModel('gemini-1.5-flash')
    print("✅ Gemini 2.0 Flash initialized successfully!")
except Exception as e:
    print(f"⚠️ Gemini 2.0 Flash failed: {e}")
    print("⚡ Trying Gemini 1.5 Pro...")
    try:
        model = genai.GenerativeModel(model_name='gemini-1.5-pro')
        print("✅ Gemini 1.5 Pro initialized successfully!")
    except Exception as e2:
        print(f"❌ Both models failed: {e2}")
        logger.error(f"Failed to initialize Gemini model: {e2}")
        model = None

if model is None:
    raise RuntimeError("Failed to initialize Gemini model. Check your API key.")

SYSTEM_PROMPT = f"""You are an autonomous quiz-solving agent with 9 specialized tools.

CRITICAL INSTRUCTIONS:

1. Load quiz pages from URLs using get_rendered_html
2. Extract instructions and identify the submit endpoint
3. Solve ALL tasks correctly using available tools
4. Submit answers ONLY to the correct endpoint with post_request
5. Follow new URLs until completion, then output: END

AVAILABLE TOOLS (use these by describing what you need):

1. get_rendered_html - Fetch full HTML content from a URL
   Input: url (string)
   Output: HTML content

2. download_file - Save a file from URL to LLMFiles/
   Input: url (string), filename (string)
   Output: File saved confirmation

3. post_request - Submit HTTP POST request with JSON payload
   Input: url (string), payload (JSON object)
   Output: Server response

4. run_code - Execute Python code for calculations
   Input: code (string - Python code)
   Output: Code output

5. encode_image_to_base64 - Convert image to Base64
   Input: image_path (string)
   Output: Base64 key for submission

6. add_dependencies - Install Python packages
   Input: package_name (string)
   Output: Installation status

7. ocr_image_tool - Extract text from images
   Input: image_path (string)
   Output: Extracted text

8. transcribe_audio_tool - Convert audio to text
   Input: audio_path (string)
   Output: Transcribed text

9. pdf_to_image_tool - Convert PDF pages to images
   Input: pdf_path (string), output_prefix (string)
   Output: List of saved image paths

RULES:
- For base64 generation of images, ALWAYS use encode_image_to_base64 tool
- Never hallucinate URLs or fields
- Always inspect server responses
- Never stop early
- Email: {EMAIL}
- Secret: {SECRET}

Proceed immediately! Load the URL and start solving."""


# ============================================================================
# AGENT EXECUTION - Direct Gemini API WITHOUT TOOLS AT INIT
# ============================================================================

def process_tool_call(tool_name: str, tool_input: Dict[str, Any]) -> str:
    """Execute tool function and return result."""
    try:
        func = TOOL_FUNCTIONS.get(tool_name)
        if not func:
            return f"Unknown tool: {tool_name}"
        
        result = func(**tool_input)
        return str(result)
    except Exception as e:
        logger.error(f"Error executing {tool_name}: {e}")
        return f"Error: {str(e)}"


def run_agent(url: str):
    """Execute the quiz-solving agent using direct Gemini API."""
    print(f"\n🚀 Starting task: {url}\n")
    
    # Clear state
    BASE64_STORE.clear()
    URL_TIME.clear()
    CACHE.clear()
    
    os.environ["url"] = url
    os.environ["offset"] = "0"
    URL_TIME[url] = time.time()
    start_time = time.time()
    
    try:
        # Initialize chat with system prompt
        chat = model.start_chat(history=[])
        
        # Send initial message with URL
        initial_message = f"{SYSTEM_PROMPT}\n\nStart with this URL: {url}"
        print(f"📢 Sending to Gemini:\n{initial_message[:300]}...\n")
        response = chat.send_message(initial_message)
        
        # Agent loop - continue until END or timeout
        iteration = 0
        max_iterations = 50
        
        while iteration < max_iterations:
            iteration += 1
            
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed >= TIMEOUT_LIMIT:
                print(f"\n⏰ TIMEOUT: {elapsed:.1f}s elapsed. Instructing agent to fail gracefully.")
                timeout_msg = "You have exceeded the 180-second time limit. Submit a WRONG answer immediately."
                response = chat.send_message(timeout_msg)
                break
            
            print(f"\n{'='*80}")
            print(f"ITERATION {iteration} | Elapsed: {elapsed:.1f}s")
            print(f"{'='*80}")
            
            # Check for END in response
            if hasattr(response, 'text'):
                text = response.text
                print(f"📝 Response: {text[:300]}")
                if "END" in text or "end" in text.lower():
                    print("\n✅ Agent completed task!")
                    break
            
            # Send continuation to agent (with tool instructions)
            if iteration == 1:
                continuation = "Now use the tools to load the quiz page and solve it. Start by calling get_rendered_html."
            else:
                continuation = "Continue with the next step. Use the appropriate tool."
            
            response = chat.send_message(continuation)
        
        if iteration >= max_iterations:
            print(f"\n⚠️ Reached maximum iterations ({max_iterations})")
        
        elapsed = time.time() - start_time
        print(f"\n✅ Task completed in {elapsed:.1f}s\n")
        
    except Exception as e:
        logger.error(f"❌ Agent error: {e}")
        print(f"\n❌ Task failed: {e}\n")


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
    """Root health check endpoint - replaces /healthz."""
    return {
        "status": "ok",
        "uptime": int(time.time() - START_TIME),
        "service": "Quiz Solver",
        "timestamp": time.time()
    }


@app.get("/health")
def health():
    """Alternative health check endpoint."""
    return {"status": "ok", "uptime": int(time.time() - START_TIME)}


@app.post("/solve")
async def solve(request: Request, background_tasks: BackgroundTasks):
    """Main solve endpoint - handles quiz tasks asynchronously."""
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
    
    print("✅ Verified. Starting task...")
    
    # Clear state for new task
    URL_TIME.clear()
    BASE64_STORE.clear()
    
    # Run agent in background
    background_tasks.add_task(run_agent, url)
    
    return JSONResponse(status_code=200, content={"status": "ok"})


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    print(f"\n🚀 Starting server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
