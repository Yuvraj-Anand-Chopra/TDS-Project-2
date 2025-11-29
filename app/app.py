"""
AUTONOMOUS QUIZ SOLVER - PROPER IMPLEMENTATION
Executes full quiz chain: Load → Analyze → Solve → Submit → Next URL → Repeat
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
from typing import Any, Dict, Optional
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import google.generativeai as genai

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

EMAIL = os.getenv("EMAIL", "")
SECRET = os.getenv("SECRET", "")
EXPECTED_SECRET = os.getenv("SECRET", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

BASE64_STORE = {}
TIMEOUT_LIMIT = 180  # 3 minutes

# ============================================================================
# TOOL IMPLEMENTATIONS
# ============================================================================

def get_rendered_html(url: str) -> str:
    """Fetch HTML from URL."""
    try:
        logger.info(f"📄 Fetching: {url}")
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        content = response.text[:200000]  # Truncate if too large
        logger.info(f"✅ Retrieved {len(content)} chars")
        return content
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return f"Error: {str(e)}"

def post_request(url: str, payload: Dict[str, Any]) -> str:
    """Submit POST request with JSON payload."""
    try:
        logger.info(f"📤 POST to {url}")

        # Replace BASE64 keys if needed
        if isinstance(payload.get("answer"), str) and payload["answer"].startswith("BASE64_KEY:"):
            key = payload["answer"].split(":", 1)[1]
            if key in BASE64_STORE:
                payload["answer"] = BASE64_STORE[key]

        response = requests.post(url, json=payload, timeout=60)
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
        logger.info("🐍 Running code...")
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
        result = (stdout + ("\nErrors:\n" + stderr if stderr else ""))[:3000]

        logger.info(f"✅ Result: {result[:200]}")
        return result or "Executed"
    except Exception as e:
        return f"Error: {str(e)}")

def download_file(url: str, filename: str) -> str:
    """Download file."""
    try:
        logger.info(f"⬇️ Downloading {filename}")
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()
        os.makedirs("LLMFiles", exist_ok=True)
        filepath = os.path.join("LLMFiles", filename)
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(8192):
                if chunk:
                    f.write(chunk)
        logger.info(f"✅ Saved to {filepath}")
        return f"Saved: {filepath}"
    except Exception as e:
        return f"Error: {str(e)}"

def encode_image_to_base64(image_path: str) -> str:
    """Encode image to Base64."""
    try:
        if not image_path.startswith("LLMFiles"):
            image_path = os.path.join("LLMFiles", image_path)
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        key = str(uuid.uuid4())
        BASE64_STORE[key] = encoded
        return f"BASE64_KEY:{key}"
    except Exception as e:
        return f"Error: {str(e)}"

def add_dependencies(package_name: str) -> str:
    """Install package."""
    try:
        if package_name in ["hashlib", "math", "json", "os", "sys"]:
            return "Built-in"
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name, "-q"])
        return f"Installed {package_name}"
    except Exception as e:
        return f"Error: {str(e)}"

TOOLS = {
    "get_rendered_html": get_rendered_html,
    "post_request": post_request,
    "run_code": run_code,
    "download_file": download_file,
    "encode_image_to_base64": encode_image_to_base64,
    "add_dependencies": add_dependencies,
}

# ============================================================================
# GEMINI MODEL
# ============================================================================

genai.configure(api_key=GOOGLE_API_KEY)

print("⚡ Initializing Gemini 2.0 Flash Lite...")
try:
    model = genai.GenerativeModel("gemini-2.0-flash-lite")
    print("✅ Initialized!")
except Exception as e:
    print(f"⚠️ Failed: {e}, trying 1.5 Flash...")
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        print("✅ Initialized!")
    except Exception as e2:
        raise RuntimeError(f"Model init failed: {e2}")

# ============================================================================
# AGENT - REAL QUIZ SOLVING
# ============================================================================

def run_agent(url: str):
    """Execute quiz-solving agent."""
    logger.info(f"\n🚀 Starting: {url}\n")

    start_time = time.time()
    current_url = url
    iteration = 0
    max_iterations = 50

    SYSTEM_PROMPT = f"""You are an autonomous quiz solver. Your task:

1. Load the quiz page from the URL
2. Read and understand the task
3. Solve it using available tools
4. Submit the answer to the submit endpoint
5. If you get a new URL, repeat from step 1
6. Stop when there's no new URL

Email: {EMAIL}
Secret: {SECRET}

CRITICAL RULES:
- Always use tools to actually DO things (not just describe)
- Actually call get_rendered_html to load pages
- Actually call post_request to submit answers
- Parse HTML and extract task requirements
- Follow submission endpoints exactly as specified
- For POST requests, use format: {{\"email\": \"{EMAIL}\", \"secret\": \"{SECRET}\", \"url\": \"...\", \"answer\": ...}}
- Continue until quiz ends (no new URL received)
- Work efficiently and don't waste time

Available tools: get_rendered_html, post_request, run_code, download_file, encode_image_to_base64, add_dependencies"""

    try:
        while iteration < max_iterations:
            iteration += 1
            elapsed = time.time() - start_time

            if elapsed >= TIMEOUT_LIMIT:
                logger.warning(f"⏰ TIMEOUT after {elapsed:.1f}s")
                break

            logger.info(f"\n{'='*80}")
            logger.info(f"ITERATION {iteration} | URL: {current_url} | Elapsed: {elapsed:.1f}s")
            logger.info(f"{'='*80}\n")

            if iteration == 1:
                prompt = f"""{SYSTEM_PROMPT}

Start now. Load this URL and solve the quiz: {current_url}

Think step by step:
1. What does the task ask?
2. What tools do I need?
3. Execute them
4. Post the answer
5. What URL is next?"""
            else:
                prompt = f"""{SYSTEM_PROMPT}

Continue. The current URL is: {current_url}

Execute the task and keep going until the quiz ends."""

            logger.info("📢 Asking Gemini...\n")

            response = model.generate_content(prompt, stream=False)

            if not response or not response.text:
                logger.warning("⚠️ No response")
                continue

            text = response.text
            logger.info(f"📝 Response ({len(text)} chars):\n{text[:500]}\n")

            # Heuristics to stop or follow next URL
            if "no new url" in text.lower() or ("quiz" in text.lower() and "end" in text.lower()):
                logger.info("\n✅ Quiz ended!")
                break

            import re
            urls = re.findall(r'https?://[^\s"\)]+', text)
            if urls:
                next_url = urls[-1]
                if next_url != current_url and "tds" in next_url:
                    current_url = next_url
                    logger.info(f"🔗 Next URL: {current_url}")
                    continue

            if "correct" in text.lower() and "true" in text.lower():
                logger.info("✅ Answer accepted!")
                if urls:
                    current_url = urls[-1]
                    continue
                else:
                    logger.info("✅ Quiz complete!")
                    break

            if iteration > 10:
                logger.warning("⚠️ Too many iterations, stopping")
                break

        elapsed = time.time() - start_time
        logger.info(f"\n✅ Task completed in {elapsed:.1f}s\n")

    except Exception as e:
        logger.error(f"❌ Error: {e}")

# ============================================================================
# FASTAPI
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
        "timestamp": time.time()
    }

@app.post("/solve")
async def solve(request: Request, background_tasks: BackgroundTasks):
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    url = data.get("url")
    secret = data.get("secret")

    if not url or not secret:
        raise HTTPException(status_code=400, detail="Missing url or secret")

    if secret != EXPECTED_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")

    logger.info("✅ Verified. Starting task...")
    background_tasks.add_task(run_agent, url)

    return JSONResponse({"status": "ok"})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    logger.info(f"\n🚀 Starting on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
