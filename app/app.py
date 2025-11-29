"""
LIGHTWEIGHT QUIZ SOLVER - WITH ACTUAL TOOL EXECUTION (FIXED PARSER)
Key fix: Properly extract quoted strings from tool parameters
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
# ACTUAL TOOL IMPLEMENTATIONS
# ============================================================================

def get_rendered_html(url: str) -> str:
    """Fetch HTML from URL."""
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
    """Submit POST request with retry logic."""
    try:
        logger.info(f"📤 POST to {url}")

        if isinstance(payload.get("answer"), str) and payload["answer"].startswith("BASE64_KEY:"):
            key = payload["answer"].split(":", 1)[1]
            if key in BASE64_STORE:
                payload["answer"] = BASE64_STORE[key]

        sending = {
            "email": payload.get("email", ""),
            "url": payload.get("url", ""),
            "answer": str(payload.get("answer", ""))[:100] if isinstance(payload.get("answer"), str) else payload.get("answer")
        }
        logger.info(f"Payload: {json.dumps(sending, indent=2)}")

        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()

        logger.info(f"📥 Response: {json.dumps(data, indent=2)}")

        cur_url = os.getenv("url", "")
        if cur_url not in retry_cache:
            retry_cache[cur_url] = 0
        retry_cache[cur_url] += 1

        next_url = data.get("url")
        correct = data.get("correct", False)

        if not correct and next_url:
            if next_url not in url_time:
                url_time[next_url] = time.time()

            cur_time = time.time()
            prev = url_time.get(cur_url, cur_time)
            delay = cur_time - prev

            if retry_cache[cur_url] >= RETRY_LIMIT or delay >= TIMEOUT_LIMIT or (url_time.get(next_url) and (cur_time - url_time[next_url]) > 90):
                logger.info("⏭️ Moving to next question")
                return json.dumps({"url": next_url, "message": "Move to next"})
            else:
                logger.info("🔄 Retrying")
                return json.dumps({"url": cur_url, "message": "Retry"})

        if not next_url:
            logger.info("✅ Quiz complete!")
            return json.dumps({"message": "Tasks completed"})

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
        return f"Error: {str(e)}"

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
        logger.info(f"✅ Encoded image, key: {key[:8]}...")
        return f"BASE64_KEY:{key}"
    except Exception as e:
        return f"Error: {str(e)}"

def add_dependencies(package_name: str) -> str:
    """Install package."""
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
# TOOL CALLING AND EXECUTION ENGINE - FIXED PARSER
# ============================================================================

def parse_tool_call(response_text: str) -> tuple:
    """Extract tool name and parameters from LLM response - FIXED VERSION."""
    
    if "```" not in response_text:
        return None, None
    
    code_match = re.search(r'```(?:python)?\s*(.*?)```', response_text, re.DOTALL)
    if not code_match:
        return None, None
    
    code = code_match.group(1).strip()
    
    # Extract function calls: tool_name(...)
    func_pattern = r'(\w+)\s*\((.*?)\)'
    matches = re.findall(func_pattern, code, re.DOTALL)
    
    if not matches:
        return None, None
    
    func_name, params_str = matches[0]
    
    try:
        # Parse parameters: extract quoted strings AND other values
        params = {}
        
        # Pattern 1: key="value" (quoted strings - PRIORITY)
        quoted_params = re.findall(r'(\w+)\s*=\s*"([^"]*)"', params_str)
        for key, val in quoted_params:
            params[key] = val
        
        # Pattern 2: key=value (unquoted - for numbers, booleans)
        unquoted_params = re.findall(r'(\w+)\s*=\s*([^\s,\)]+)', params_str)
        for key, val in unquoted_params:
            if key not in params and not val.startswith('"'):
                params[key] = val.strip(',)')
        
        # If no named params, try single positional argument
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
    """Execute a tool and return result."""
    if tool_name not in TOOLS_MAP:
        return f"Error: Unknown tool {tool_name}"
    
    try:
        logger.info(f"🔧 Executing tool: {tool_name} with {params}")
        tool_func = TOOLS_MAP[tool_name]
        
        # Handle different parameter patterns
        if 'arg' in params:
            result = tool_func(params['arg'])
        elif 'url' in params and 'payload' in params:
            result = tool_func(params['url'], json.loads(params['payload']))
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
        
        logger.info(f"✅ Tool result: {str(result)[:200]}")
        return str(result)
    except Exception as e:
        logger.error(f"❌ Tool execution error: {e}")
        return f"Error executing {tool_name}: {str(e)}"

# ============================================================================
# GEMINI MODEL
# ============================================================================

genai.configure(api_key=GOOGLE_API_KEY)

print("⚡ Initializing Gemini...")
try:
    model = genai.GenerativeModel("gemini-2.0-flash-lite")
    print("✅ Gemini 2.0 Flash Lite initialized!")
except Exception as e:
    print(f"⚠️ Trying Gemini 1.5 Flash... ({e})")
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        print("✅ Gemini 1.5 Flash initialized!")
    except Exception as e2:
        raise RuntimeError(f"Model init failed: {e2}")

# ============================================================================
# AGENT - QUIZ SOLVER WITH TOOL EXECUTION
# ============================================================================

def run_agent(url: str):
    """Execute quiz-solving agent with ACTUAL tool execution."""
    logger.info(f"\n🚀 Starting: {url}\n")

    current_url = url
    iteration = 0
    max_iterations = 50
    message_history = []

    SYSTEM_PROMPT = (
        "You are an autonomous quiz solver.\n\n"
        "Your job:\n"
        "1. Load quiz page from URL using get_rendered_html\n"
        "2. Read instructions and extract task\n"
        "3. Solve the task using tools\n"
        "4. Submit answer using post_request\n"
        "5. Follow new URLs until none remain\n"
        "6. Output END when complete\n\n"
        "CRITICAL:\n"
        "- ALWAYS call tools with proper syntax: tool_name(param=\"value\")\n"
        "- For images: use encode_image_to_base64(image_path=\"...\")\n"
        "- For POST: use post_request(url=\"...\", payload={...})\n"
        "- Include email, secret, url, answer in payload\n"
        "- Never hallucinate URLs\n"
        "- Continue until no new URL\n\n"
        f"Tools: get_rendered_html, post_request, run_code, download_file, encode_image_to_base64, add_dependencies\n"
        f"Email: {EMAIL}\nSecret: {SECRET}"
    )

    try:
        while iteration < max_iterations:
            iteration += 1
            elapsed = time.time() - (url_time.get(current_url, time.time()))

            offset = os.getenv("offset", "0")
            if elapsed >= TIMEOUT_LIMIT or (offset != "0" and elapsed > 90):
                logger.warning(f"⏰ TIMEOUT: {elapsed:.1f}s")
                break

            logger.info(f"\n{'='*80}")
            logger.info(f"ITERATION {iteration} | URL: {current_url} | Elapsed: {elapsed:.1f}s")
            logger.info(f"{'='*80}\n")

            if iteration == 1:
                prompt = f"{SYSTEM_PROMPT}\n\nSTART: Load and solve {current_url}"
            else:
                prompt = f"Continue with URL: {current_url}\n\nSolve and submit, then follow new URL if provided. If no new URL, output END."

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
            logger.info(f"📝 Response ({len(text)} chars):\n{text[:500]}\n")

            # ==== CRITICAL FIX: Actually execute tools ====
            tool_name, params = parse_tool_call(text)
            
            if tool_name:
                logger.info(f"🔧 Tool detected: {tool_name}")
                tool_result = execute_tool(tool_name, params)
                
                # Add tool result back to history
                message_history.append({"role": "assistant", "content": text})
                message_history.append({
                    "role": "user",
                    "content": f"Tool {tool_name} returned:\n{tool_result}\n\nContinue solving."
                })
                
                # Trim history if too long
                if len(message_history) > 20:
                    logger.info("📊 Trimming message history...")
                    message_history = message_history[:2] + message_history[-15:]
                
                continue
            
            # ==========================================

            message_history.append({"role": "assistant", "content": text})

            # Check for completion
            if "END" in text or "complete" in text.lower():
                logger.info("✅ Quiz complete!")
                break

            # Extract next URL
            urls = re.findall(r'https?://[^\s"\)\]]+', text)
            if urls:
                next_url = urls[-1]
                if next_url != current_url and "tds" in next_url:
                    current_url = next_url
                    if current_url not in url_time:
                        url_time[current_url] = time.time()
                    logger.info(f"🔗 Next URL: {current_url}")
                    continue

            if "correct" in text.lower() and ("true" in text.lower() or "accepted" in text.lower()):
                logger.info("✅ Answer accepted!")
                if urls:
                    current_url = urls[-1]
                    if current_url not in url_time:
                        url_time[current_url] = time.time()
                    continue

            if iteration > 20:
                logger.warning("⚠️ Too many iterations, stopping")
                break

        elapsed = time.time() - url_time.get(current_url, time.time())
        logger.info(f"\n✅ Task completed in {elapsed:.1f}s\n")

    except Exception as e:
        logger.error(f"❌ Agent error: {e}")

# ============================================================================
# FASTAPI APP
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

    if not data:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    url = data.get("url")
    secret = data.get("secret")

    if not url or not secret:
        raise HTTPException(status_code=400, detail="Missing url or secret")

    if secret != EXPECTED_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")

    retry_cache.clear()
    url_time.clear()
    BASE64_STORE.clear()

    logger.info("✅ Verified. Starting task...")
    os.environ["url"] = url
    os.environ["offset"] = "0"
    url_time[url] = time.time()

    background_tasks.add_task(run_agent, url)

    return JSONResponse(status_code=200, content={"status": "ok"})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    logger.info(f"\n🚀 Starting on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
