"""
AI-POWERED QUIZ SOLVER - PRODUCTION READY
- Receives quiz URLs via POST /solve-quiz endpoint
- Uses Gemini 2.0 Flash to intelligently solve any question type
- Automatically submits answers back to server
- Handles chains of questions seamlessly
- Can perform: file downloads, image parsing, calculations, text analysis, etc.
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
from typing import Any, Dict
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
url_time = {}
retry_cache = {}
RETRY_LIMIT = 4
TIMEOUT_LIMIT = 180

# ============================================================================
# TOOL IMPLEMENTATIONS - THESE ACTUALLY EXECUTE
# ============================================================================

def get_rendered_html(url: str) -> str:
    """Fetch HTML from URL - handles quiz page loading."""
    try:
        logger.info(f"📄 Fetching: {url}")
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        content = response.text[:200000]
        logger.info(f"✅ Retrieved {len(content)} chars")
        return content
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return f"Error: {str(e)}"

def post_request(url: str, payload: Dict[str, Any]) -> str:
    """Submit answer to quiz server - THIS IS THE KEY FUNCTION."""
    try:
        logger.info(f"📤 POSTING ANSWER to {url}")

        if isinstance(payload.get("answer"), str) and payload["answer"].startswith("BASE64_KEY:"):
            key = payload["answer"].split(":", 1)[1]
            if key in BASE64_STORE:
                payload["answer"] = BASE64_STORE[key]

        sending = {
            "email": payload.get("email", ""),
            "url": payload.get("url", ""),
            "answer": str(payload.get("answer", ""))[:100] if isinstance(payload.get("answer"), str) else payload.get("answer")
        }
        logger.info(f"📦 Payload: {json.dumps(sending, indent=2)}")

        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()

        logger.info(f"📥 Server Response: {json.dumps(data, indent=2)}")

        cur_url = os.getenv("url", "")
        if cur_url not in retry_cache:
            retry_cache[cur_url] = 0
        retry_cache[cur_url] += 1

        next_url = data.get("url")
        correct = data.get("correct", False)

        if correct:
            logger.info("✅ ANSWER CORRECT!")
        else:
            logger.info("❌ Answer incorrect, will retry or move to next")

        if not correct and next_url:
            if next_url not in url_time:
                url_time[next_url] = time.time()

            cur_time = time.time()
            prev = url_time.get(cur_url, cur_time)
            delay = cur_time - prev

            if retry_cache[cur_url] >= RETRY_LIMIT or delay >= TIMEOUT_LIMIT:
                logger.info("⏭️ Moving to next question")
                return json.dumps({"url": next_url, "message": "Move to next"})
            else:
                logger.info("🔄 Retrying this question")
                return json.dumps({"url": cur_url, "message": "Retry"})

        if not next_url:
            logger.info("✅ Quiz chain complete!")
            return json.dumps({"message": "Tasks completed"})

        return json.dumps(data)

    except Exception as e:
        logger.error(f"❌ POST Error: {e}")
        return json.dumps({"error": str(e)})

def run_code(code: str) -> str:
    """Execute Python code for calculations/data processing."""
    try:
        logger.info("🐍 Executing Python code...")
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

        logger.info(f"✅ Code executed: {result[:200]}")
        return result or "Executed"
    except Exception as e:
        return f"Error: {str(e)}"

def download_file(url: str, filename: str) -> str:
    """Download files needed for solving."""
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
    """Encode images for submission."""
    try:
        if not image_path.startswith("LLMFiles"):
            image_path = os.path.join("LLMFiles", image_path)
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        key = str(uuid.uuid4())
        BASE64_STORE[key] = encoded
        logger.info(f"✅ Image encoded, key: {key[:8]}...")
        return f"BASE64_KEY:{key}"
    except Exception as e:
        return f"Error: {str(e)}"

def add_dependencies(package_name: str) -> str:
    """Install Python packages on demand."""
    try:
        if package_name in ["hashlib", "math", "json", "os", "sys"]:
            return "Built-in"
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name, "-q"])
        logger.info(f"✅ Installed {package_name}")
        return f"Installed {package_name}"
    except Exception as e:
        return f"Error: {str(e)}"

TOOLS_MAP = {
    "get_rendered_html": get_rendered_html,
    "post_request": post_request,
    "run_code": run_code,
    "download_file": download_file,
    "encode_image_to_base64": encode_image_to_base64,
    "add_dependencies": add_dependencies,
}

# ============================================================================
# TOOL PARSING - EXTRACT VALUES FROM GEMINI RESPONSES
# ============================================================================

def parse_tool_call(response_text: str) -> tuple:
    """Parse Gemini's tool calls and extract actual parameter values."""
    
    if "```" not in response_text:
        return None, None
    
    code_match = re.search(r'```(?:python|tool_code)?\s*(.*?)```', response_text, re.DOTALL)
    if not code_match:
        return None, None
    
    code = code_match.group(1).strip()
    
    func_pattern = r'(\w+)\s*\((.*?)\)'
    matches = re.findall(func_pattern, code, re.DOTALL)
    
    if not matches:
        return None, None
    
    func_name, params_str = matches[0]
    
    try:
        params = {}
        
        # Priority 1: Extract quoted strings (ACTUAL VALUES)
        quoted_params = re.findall(r'(\w+)\s*=\s*"([^"]*)"', params_str)
        for key, val in quoted_params:
            params[key] = val
        
        # Priority 2: Extract unquoted values
        unquoted_params = re.findall(r'(\w+)\s*=\s*([^\s,\)]+)', params_str)
        for key, val in unquoted_params:
            if key not in params and not val.startswith('"'):
                params[key] = val.strip(',)')
        
        # Single positional argument
        if not params:
            single_quote = re.search(r'"([^"]*)"', params_str)
            if single_quote:
                params['arg'] = single_quote.group(1)
        
        if params:
            logger.info(f"✅ Parsed: {func_name}({params})")
            return func_name, params
    
    except Exception as e:
        logger.error(f"❌ Parse error: {e}")
    
    return None, None

def execute_tool(tool_name: str, params: dict) -> str:
    """Execute the parsed tool with actual values."""
    if tool_name not in TOOLS_MAP:
        return f"Error: Unknown tool {tool_name}"
    
    try:
        logger.info(f"🔧 Executing: {tool_name}({params})")
        tool_func = TOOLS_MAP[tool_name]
        
        if 'arg' in params:
            result = tool_func(params['arg'])
        elif 'url' in params and 'payload' in params:
            import json as json_module
            payload = json_module.loads(params['payload']) if isinstance(params['payload'], str) else params['payload']
            result = tool_func(params['url'], payload)
        elif 'url' in params and 'filename' in params:
            result = tool_func(params['url'], params['filename'])
        elif 'image_path' in params:
            result = tool_func(params['image_path'])
        elif 'package_name' in params:
            result = tool_func(params['package_name'])
        elif 'code' in params:
            result = tool_func(params['code'])
        else:
            result = tool_func(**params)
        
        logger.info(f"✅ Result: {str(result)[:300]}")
        return str(result)
    except Exception as e:
        logger.error(f"❌ Execution error: {e}")
        return f"Error: {str(e)}"

# ============================================================================
# GEMINI AI MODEL SETUP
# ============================================================================

genai.configure(api_key=GOOGLE_API_KEY)

print("⚡ Initializing Gemini...")
try:
    model = genai.GenerativeModel("gemini-2.0-flash-lite")
    print("✅ Gemini 2.0 Flash Lite initialized!")
except Exception as e:
    print(f"⚠️ Trying Gemini 1.5 Flash...")
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        print("✅ Gemini 1.5 Flash initialized!")
    except Exception as e2:
        raise RuntimeError(f"Model init failed: {e2}")

# ============================================================================
# AUTONOMOUS QUIZ SOLVER AGENT
# ============================================================================

def run_agent(url: str):
    """Main agent loop - visits URL, reads question, solves it, submits answer."""
    logger.info(f"\n🚀 Starting quiz solver at: {url}\n")

    current_url = url
    iteration = 0
    max_iterations = 50
    message_history = []

    SYSTEM_PROMPT = (
        "You are an expert autonomous quiz solver. Your ONLY job is to CALL TOOLS WITH ACTUAL VALUES.\n\n"
        "CRITICAL RULES:\n"
        "1. ALWAYS call tools directly with REAL values - NEVER use Python variables\n"
        "2. EVERY iteration, you MUST make a tool call - don't explain, just call\n"
        "3. After fetching page: analyze it and IMMEDIATELY post the answer\n"
        "4. Always format tool calls like this:\n"
        "   ```tool_code\n"
        "   get_rendered_html(url=\"https://example.com/quiz\")\n"
        "   ```\n"
        "   OR:\n"
        "   ```tool_code\n"
        "   post_request(url=\"https://example.com/submit\", payload={\"email\": \"user@example.com\", \"secret\": \"pass\", \"url\": \"https://...\", \"answer\": \"42\"})\n"
        "   ```\n\n"
        f"CREDENTIALS (use exactly as shown):\n"
        f"  Email: {EMAIL}\n"
        f"  Secret: {SECRET}\n"
        f"  Submit to: https://tds-llm-analysis.s-anand.net/submit\n\n"
        "WORKFLOW:\n"
        "1. Call get_rendered_html to fetch the question page\n"
        "2. Analyze the HTML to understand what's being asked\n"
        "3. Solve the problem (math, parsing, download, etc)\n"
        "4. Call post_request with your answer - INCLUDE ALL 4 FIELDS: email, secret, url, answer\n"
        "5. If server returns new URL, continue with that URL\n"
        "6. If server says 'correct', move to next URL\n"
        "7. Repeat until no new URLs\n\n"
        "Available tools:\n"
        "- get_rendered_html(url=\"...\") - fetch webpage\n"
        "- post_request(url=\"...\", payload={...}) - submit answer\n"
        "- run_code(code=\"...\") - execute Python for calculations\n"
        "- download_file(url=\"...\", filename=\"...\") - download files\n"
        "- encode_image_to_base64(image_path=\"...\") - convert image to base64\n"
        "- add_dependencies(package_name=\"...\") - install Python packages"
    )

    try:
        while iteration < max_iterations:
            iteration += 1
            elapsed = time.time() - (url_time.get(current_url, time.time()))

            if elapsed >= TIMEOUT_LIMIT:
                logger.warning(f"⏰ TIMEOUT after {elapsed:.1f}s")
                break

            logger.info(f"\n{'='*80}")
            logger.info(f"ITERATION {iteration} | URL: {current_url} | Elapsed: {elapsed:.1f}s")
            logger.info(f"{'='*80}\n")

            if iteration == 1:
                prompt = f"{SYSTEM_PROMPT}\n\n🎯 START: Fetch and solve the quiz at {current_url}"
            else:
                prompt = f"Continue. Next question URL: {current_url}\n\nFetch the page, analyze, solve, and POST your answer with ALL fields (email, secret, url, answer)."

            message_history.append({"role": "user", "content": prompt})

            logger.info("📢 Asking Gemini...\n")

            response = model.generate_content(
                [{"role": msg["role"], "parts": [msg["content"]]} for msg in message_history],
                stream=False
            )

            if not response or not response.text:
                logger.warning("⚠️ No response from model")
                continue

            text = response.text
            logger.info(f"📝 Gemini response ({len(text)} chars):\n{text[:800]}\n")

            # PARSE AND EXECUTE TOOL
            tool_name, params = parse_tool_call(text)
            
            if tool_name:
                logger.info(f"🔧 Tool call detected: {tool_name}")
                tool_result = execute_tool(tool_name, params)
                
                # Add to history for Gemini to see result
                message_history.append({"role": "assistant", "content": text})
                message_history.append({
                    "role": "user",
                    "content": f"Tool {tool_name} returned:\n{tool_result}\n\nAnalyze this and continue solving. If correct, look for next URL. If not correct, try another answer."
                })
                
                # Keep conversation history manageable
                if len(message_history) > 20:
                    logger.info("📊 Trimming history...")
                    message_history = message_history[:2] + message_history[-15:]
                
                continue
            
            message_history.append({"role": "assistant", "content": text})

            # Check for quiz completion
            if "END" in text or "complete" in text.lower():
                logger.info("✅ Quiz marked complete by Gemini")
                break

            # Extract new URLs from response
            urls = re.findall(r'https?://[^\s"\)\]]+', text)
            if urls:
                next_url = urls[-1]
                if next_url != current_url:
                    current_url = next_url
                    if current_url not in url_time:
                        url_time[current_url] = time.time()
                    logger.info(f"🔗 Next URL found: {current_url}")
                    continue

            if iteration > 15:
                logger.warning("⚠️ Too many iterations without tool call")
                break

        elapsed = time.time() - url_time.get(current_url, time.time())
        logger.info(f"\n✅ Quiz solving session completed in {elapsed:.1f}s\n")

    except Exception as e:
        logger.error(f"❌ Agent error: {e}")

# ============================================================================
# FASTAPI SERVER - RECEIVES REQUESTS AND STARTS SOLVER
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

@app.get("/health")
def health():
    """Health check endpoint - returns model status."""
    return {
        "status": "ok",
        "model": "gemini-2.0-flash"
    }

@app.get("/")
def root():
    """Root endpoint - same as /health."""
    return health()

@app.post("/solve-quiz")
async def solve_quiz(request: Request, background_tasks: BackgroundTasks):
    """Receive quiz URL and start solving."""
    try:
        data = await request.json()
    except:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    email = data.get("email")
    secret = data.get("secret")
    url = data.get("url")

    if not url or not secret or not email:
        raise HTTPException(status_code=400, detail="Missing email, url, or secret")

    if secret != EXPECTED_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")

    # Clear state
    retry_cache.clear()
    url_time.clear()
    BASE64_STORE.clear()

    logger.info("✅ Authenticated. Starting quiz solver in background...")
    os.environ["url"] = url
    os.environ["email"] = email
    url_time[url] = time.time()

    # Run agent in background
    background_tasks.add_task(run_agent, url)

    return JSONResponse(
        status_code=200, 
        content={
            "correct": True,
            "url": url,
            "reason": None
        }
    )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    logger.info(f"\n🚀 Starting AI Quiz Solver on port {port}...\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
