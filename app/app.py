"""
AI Quiz Solver - Project 2 Edition (FIXED)

Enhanced FastAPI application for Project 2 with:
- Proper quota detection and handling
- Exponential backoff with API retry_delay extraction
- Better next URL parsing
- Graceful failure on quota exceeded
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
from typing import Any, Dict, Optional, Tuple
from functools import wraps
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import google.generativeai as genai
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Configuration
EMAIL = os.getenv("EMAIL", "")
SECRET = os.getenv("SECRET", "")
EXPECTED_SECRET = os.getenv("SECRET", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Project 2 specific configuration
PROJECT_TYPE = "project2"
INITIAL_URL = "https://tds-llm-analysis.s-anand.net/project2"
SUBMIT_URL = "https://tds-llm-analysis.s-anand.net/submit"

# Difficulty levels mapping
DIFFICULTY_LEVELS = {
    1: {'name': 'Easy', 'next_url_always_shown': True, 'max_retries': 5},
    2: {'name': 'Medium', 'next_url_always_shown': True, 'max_retries': 5},
    3: {'name': 'Hard', 'next_url_always_shown': False, 'max_retries': 3},
    4: {'name': 'Expert', 'next_url_always_shown': False, 'max_retries': 3},
    5: {'name': 'Master', 'next_url_always_shown': False, 'max_retries': 2},
}

# Global stores
BASE64_STORE = {}
url_time = {}
retry_cache = {}
QUOTA_EXHAUSTED = False  # CRITICAL: Track if quota exceeded

RETRY_LIMIT = 4
TIMEOUT_LIMIT = 300

# ============================================================================
# ERROR HANDLING & CLASSIFICATION
# ============================================================================

class ErrorHandler:
    """
    Unified error classification and recovery strategy.
    Distinguishes between rate_limit (temporary) and quota_exceeded (permanent).
    """
    
    ERROR_TYPES = {
        '429': 'rate_limit',
        '403': 'auth',
        '404': 'not_found',
        'timeout': 'timeout',
        'quota exceeded': 'quota_exceeded',
        'Resource exhausted': 'quota_exceeded',
        'limit: 0': 'quota_exceeded',
        'Connection': 'connection_error',
        'wrong': 'answer_incorrect',
    }
    
    @staticmethod
    def classify_error(error: Exception) -> str:
        """Classify the type of error that occurred."""
        error_str = str(error).lower()
        
        # Check for quota exhaustion FIRST (most critical)
        if 'limit: 0' in error_str or ('quota exceeded' in error_str and 'limit: 0' in error_str):
            return 'quota_exceeded'
        if 'quota exceeded' in error_str and 'daily' in error_str:
            return 'quota_exceeded'
            
        for pattern, error_type in ErrorHandler.ERROR_TYPES.items():
            if pattern.lower() in error_str:
                logger.info(f"Error classified as: {error_type}")
                return error_type
        
        return 'unknown'
    
    @staticmethod
    def get_retry_delay(error: Exception) -> Optional[int]:
        """Extract retry_delay from API error response."""
        error_str = str(error)
        
        # Look for "retry_delay { seconds: XX }"
        match = re.search(r'seconds:\s*(\d+)', error_str)
        if match:
            delay = int(match.group(1))
            logger.info(f"API suggests retry after {delay} seconds")
            return delay
        
        return None
    
    @staticmethod
    def get_recovery_strategy(error_type: str) -> dict:
        """Get the recovery strategy for a specific error type."""
        strategies = {
            'quota_exceeded': {
                'retry': False,
                'action': 'fail_fast',
                'log': 'critical',
                'reason': 'Daily quota exhausted - cannot proceed'
            },
            'rate_limit': {
                'retry': True,
                'wait': 2,
                'backoff': 'exponential',
                'max_retries': 3,
                'action': 'exponential_backoff'
            },
            'parser_error': {
                'retry': True,
                'wait': 1,
                'action': 'restructure_input',
                'max_retries': 2,
                'backoff': 'linear'
            },
            'timeout': {
                'retry': True,
                'wait': 2,
                'backoff': 'exponential',
                'max_retries': 2,
                'action': 'retry_with_delay'
            },
            'answer_incorrect': {
                'retry': True,
                'wait': 1,
                'action': 'retry_different_approach',
                'max_retries': 2,
                'backoff': 'none'
            },
            'auth': {
                'retry': False,
                'action': 'fail_fast',
                'log': 'critical'
            },
        }
        return strategies.get(error_type, {'retry': False})


# ============================================================================
# IMPROVED RATE LIMITING DECORATOR WITH QUOTA DETECTION
# ============================================================================

def rate_limit_aware(max_retries: int = 3):
    """
    Decorator that handles rate limiting with exponential backoff.
    CRITICAL: Detects quota_exceeded and fails fast.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            global QUOTA_EXHAUSTED
            
            wait_time = 1
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                
                except Exception as e:
                    error_type = ErrorHandler.classify_error(e)
                    
                    # CRITICAL: Quota exhausted = permanent failure
                    if error_type == 'quota_exceeded':
                        QUOTA_EXHAUSTED = True
                        logger.error("⚠️  QUOTA EXHAUSTED - Cannot proceed")
                        logger.error(f"   Reason: {str(e)[:200]}")
                        raise
                    
                    # Rate limit = temporary, apply backoff
                    if error_type == 'rate_limit':
                        # Check if API suggests retry delay
                        api_delay = ErrorHandler.get_retry_delay(e)
                        if api_delay:
                            wait_time = api_delay
                        
                        logger.warning(f"Rate limited. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                        time.sleep(wait_time)
                        wait_time = min(wait_time * 2, 32)  # Exponential backoff up to 32s
                        
                        if attempt < max_retries - 1:
                            continue
                    
                    raise
        
        return wrapper
    return decorator


# ============================================================================
# ENHANCED CONTENT ANALYZER
# ============================================================================

class ContentAnalyzer:
    """
    Analyzes HTML content to understand task requirements.
    Enhanced for Project 2 to detect difficulty, format, and personalization.
    """
    
    @staticmethod
    def extract_difficulty_level(html: str) -> int:
        """Extract difficulty level from HTML (1-5)."""
        for level in range(1, 6):
            patterns = [
                f"Difficulty {level}",
                f"difficulty {level}",
                f"DIFFICULTY {level}",
                f"Level {level}",
            ]
            for pattern in patterns:
                if pattern in html:
                    logger.info(f"Difficulty level detected: {level}")
                    return level
        return 1  # Default to Easy
    
    @staticmethod
    def extract_format_requirement(html: str) -> str:
        """Extract required answer format from task page."""
        patterns = [
            r'answer\s+as\s+([^\.\:\n]+)',
            r'required\s+format[:\s]+([^\.\:\n]+)',
            r'format:\s*([^\.\:\n]+)',
            r'submit\s+as\s+([^\.\:\n]+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, html, re.IGNORECASE)
            if match:
                format_str = match.group(1).strip()
                logger.info(f"Answer format detected: {format_str}")
                return format_str
        return "unknown"
    
    @staticmethod
    def analyze_task(html: str) -> dict:
        """Comprehensive task analysis for Project 2."""
        difficulty = ContentAnalyzer.extract_difficulty_level(html)
        format_req = ContentAnalyzer.extract_format_requirement(html)
        
        return {
            'difficulty': difficulty,
            'difficulty_name': DIFFICULTY_LEVELS.get(difficulty, {}).get('name', 'Unknown'),
            'format': format_req,
        }


# ============================================================================
# IMPROVED URL PARSER WITH BETTER NEXT_URL EXTRACTION
# ============================================================================

def parse_tool_call(response_text: str) -> Tuple[Optional[str], Optional[dict]]:
    """
    Parse Gemini's tool calls from response text.
    IMPROVED: Better boundary detection and parameter extraction.
    """
    if "```" not in response_text:
        return None, None
    
    code_match = re.search(r'```(?:python|tool_code)?\s*(.*?)```', response_text, re.DOTALL)
    if not code_match:
        return None, None
    
    code = code_match.group(1).strip()
    func_pattern = r'^(\w+)\s*\((.*)?\)$'
    
    matches = re.findall(func_pattern, code, re.MULTILINE | re.DOTALL)
    if not matches:
        return None, None
    
    func_name, params_str = matches[0]
    if params_str is None:
        params_str = ""
    
    params = {}
    
    try:
        # Extract quoted strings
        quoted_pattern = r'(\w+)\s*=\s*"([^"]*)"'
        for key, val in re.findall(quoted_pattern, params_str):
            params[key] = val
        
        # Extract dictionary payloads
        dict_pattern = r'(\w+)\s*=\s*(\{(?:[^{}]|(?:\{[^{}]*\}))*\})'
        for key, val in re.findall(dict_pattern, params_str):
            try:
                params[key] = json.loads(val)
            except:
                params[key] = val
        
        # Extract unquoted values
        unquoted_pattern = r'(\w+)\s*=\s*([^\s,\)=]+)(?![\w&])'
        for key, val in re.findall(unquoted_pattern, params_str):
            if key not in params and not val.startswith('"') and not val.startswith('{'):
                params[key] = val.strip(',)')
        
        if params:
            logger.info(f"Parsed: {func_name}({list(params.keys())})")
        
        return func_name, params
    
    except Exception as e:
        logger.error(f"Parse error: {e}")
        return None, None


def extract_next_url(response_text: str, current_url: str) -> Optional[str]:
    """
    IMPROVED: Extract next URL from Gemini response with better pattern matching.
    Tries multiple extraction strategies.
    """
    
    # Strategy 1: Look for direct "next_url" or "nextUrl"
    patterns = [
        r'next_url["\']?\s*[:=]\s*["\']?(https?://[^\s"\']+)',
        r'nextUrl["\']?\s*[:=]\s*["\']?(https?://[^\s"\']+)',
        r'next URL is[:\s]+["\']?(https?://[^\s"\']+)',
        r'continuing to[:\s]+["\']?(https?://[^\s"\']+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            url = match.group(1).strip('"\'')
            if url != current_url and 'http' in url:
                logger.info(f"Next URL extracted: {url}")
                return url
    
    # Strategy 2: Find ALL URLs and return the last one that's different
    all_urls = re.findall(r'https?://[^\s"\')\]]+', response_text)
    
    if all_urls:
        # Filter for valid next URLs
        valid_urls = [u for u in all_urls if u != current_url and 'tds-llm' in u]
        if valid_urls:
            next_url = valid_urls[-1]
            logger.info(f"Next URL extracted (from URL list): {next_url}")
            return next_url
    
    logger.warning("Could not extract next URL from response")
    return None


# ============================================================================
# TOOL IMPLEMENTATIONS
# ============================================================================

def get_rendered_html(url: str) -> str:
    """Fetch rendered HTML content from a given URL."""
    try:
        logger.info(f"Fetching URL: {url}")
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        content = response.text[:200000]
        logger.info(f"Retrieved {len(content)} characters")
        return content
    except Exception as e:
        logger.error(f"Error fetching URL: {str(e)}")
        return f"Error: {str(e)}"


def post_request(url: str, payload: Dict[str, Any]) -> str:
    """Submit an answer to the quiz server via POST request."""
    try:
        logger.info(f"Posting answer to: {url}")
        
        if isinstance(payload.get("answer"), str) and payload["answer"].startswith("BASE64_KEY:"):
            key = payload["answer"].split(":", 1)[1]
            if key in BASE64_STORE:
                payload["answer"] = BASE64_STORE[key]
        
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        is_correct = data.get("correct", False)
        if is_correct:
            logger.info("✅ Answer is correct")
        else:
            logger.warning(f"❌ Answer is incorrect: {data.get('reason', 'Unknown')}")
        
        return json.dumps(data)
    
    except Exception as e:
        logger.error(f"POST Error: {str(e)}")
        return json.dumps({"error": str(e)})


def run_code(code: str) -> str:
    """Execute Python code in a subprocess."""
    try:
        logger.info("Executing Python code")
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
        logger.info(f"Code executed: {result[:200]}")
        
        return result or "Executed"
    
    except Exception as e:
        logger.error(f"Code execution error: {str(e)}")
        return f"Error: {str(e)}"


def download_file(url: str, filename: str) -> str:
    """Download a file from a URL and save it locally."""
    try:
        logger.info(f"Downloading {filename} from {url}")
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()
        
        os.makedirs("LLMFiles", exist_ok=True)
        filepath = os.path.join("LLMFiles", filename)
        
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(8192):
                if chunk:
                    f.write(chunk)
        
        logger.info(f"File saved to {filepath}")
        return f"Saved: {filepath}"
    
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        return f"Error: {str(e)}"


def encode_image_to_base64(image_path: str) -> str:
    """Convert an image file to base64 encoding."""
    try:
        if not image_path.startswith("LLMFiles"):
            image_path = os.path.join("LLMFiles", image_path)
        
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        
        key = str(uuid.uuid4())
        BASE64_STORE[key] = encoded
        logger.info(f"Image encoded, key: {key[:8]}")
        
        return f"BASE64_KEY:{key}"
    
    except Exception as e:
        logger.error(f"Image encoding error: {str(e)}")
        return f"Error: {str(e)}"


def add_dependencies(package_name: str) -> str:
    """Install a Python package using pip."""
    try:
        if package_name in ["hashlib", "math", "json", "os", "sys", "re"]:
            return "Built-in"
        
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name, "-q"])
        logger.info(f"Installed {package_name}")
        
        return f"Installed {package_name}"
    
    except Exception as e:
        logger.error(f"Package install error: {str(e)}")
        return f"Error: {str(e)}"


TOOLS_MAP = {
    "get_rendered_html": get_rendered_html,
    "post_request": post_request,
    "run_code": run_code,
    "download_file": download_file,
    "encode_image_to_base64": encode_image_to_base64,
    "add_dependencies": add_dependencies,
}


def execute_tool(tool_name: str, params: dict) -> str:
    """Execute a tool with the parsed parameters."""
    if tool_name not in TOOLS_MAP:
        return f"Error: Unknown tool {tool_name}"
    
    try:
        logger.info(f"Executing tool: {tool_name}")
        tool_func = TOOLS_MAP[tool_name]
        
        if 'arg' in params:
            result = tool_func(params['arg'])
        elif 'url' in params and 'payload' in params:
            payload = params['payload'] if isinstance(params['payload'], dict) else json.loads(params['payload'])
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
        
        logger.info(f"Tool result: {str(result)[:300]}")
        return str(result)
    
    except Exception as e:
        error_type = ErrorHandler.classify_error(e)
        logger.error(f"Execution error ({error_type}): {e}")
        return f"Error: {str(e)}"


# ============================================================================
# GEMINI MODEL INITIALIZATION
# ============================================================================

logger.info("Initializing Gemini API...")
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash-lite")
    logger.info("✅ Gemini 2.0 Flash Lite initialized")
except Exception as e:
    logger.warning(f"Gemini 2.0 Flash failed: {e}")
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        logger.info("✅ Gemini 1.5 Flash initialized (fallback)")
    except Exception as e2:
        logger.error(f"❌ Model initialization failed: {e2}")
        model = None


# ============================================================================
# AUTONOMOUS QUIZ SOLVER AGENT
# ============================================================================

@rate_limit_aware(max_retries=3)
def ask_gemini_with_rate_limit(messages: list) -> str:
    """Ask Gemini a question with built-in rate limit handling."""
    global QUOTA_EXHAUSTED
    
    if QUOTA_EXHAUSTED:
        raise RuntimeError("Quota exhausted - cannot call Gemini API")
    
    if not model:
        raise RuntimeError("Gemini model not initialized")
    
    response = model.generate_content(messages, stream=False)
    return response.text if response else ""


def run_agent(url: str):
    """
    Main agent loop for solving Project 2 quiz questions.
    IMPROVED: Detects quota exhaustion and fails fast.
    """
    global QUOTA_EXHAUSTED
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Starting Project 2 quiz solver for: {url}")
    logger.info(f"{'='*80}\n")
    
    current_url = url
    iteration = 0
    max_iterations = 50
    message_history = []
    task_attempts = {}
    
    SYSTEM_PROMPT = (
        "You are an expert PROJECT 2 SOLVER.\n\n"
        "CRITICAL RULES:\n"
        "1. Difficulty 1-2: Next URL always shown (even if wrong)\n"
        "2. Difficulty 3-5: Next URL ONLY if answer is CORRECT\n"
        "3. Read answer format requirement CAREFULLY\n"
        "4. Check if task is personalized or not\n"
        "5. Solve correctly on difficulties 3-5 or you get stuck\n\n"
        "WORKFLOW:\n"
        "1. Call get_rendered_html to fetch the task page\n"
        "2. READ the task carefully\n"
        "3. Identify: difficulty, format requirement, personalization\n"
        "4. Solve the problem CORRECTLY\n"
        "5. Format answer exactly as required\n"
        "6. POST to https://tds-llm-analysis.s-anand.net/submit\n"
        "7. Continue until all tasks completed\n\n"
        f"YOUR CREDENTIALS:\n"
        f"Email: {EMAIL}\n"
        f"Secret: {SECRET}\n"
        f"Submit to: {SUBMIT_URL}\n\n"
        "Available tools: get_rendered_html, post_request, run_code"
    )
    
    try:
        while iteration < max_iterations:
            iteration += 1
            
            if QUOTA_EXHAUSTED:
                logger.error("⚠️  QUOTA EXHAUSTED - Stopping agent")
                break
            
            logger.info(f"\n{'='*80}")
            logger.info(f"ITERATION {iteration} | URL: {current_url}")
            logger.info(f"{'='*80}\n")
            
            attempts = task_attempts.get(current_url, 0)
            if attempts > 0:
                logger.info(f"Attempt {attempts + 1} for this task\n")
            
            # Build prompt
            if iteration == 1:
                prompt = (
                    f"{SYSTEM_PROMPT}\n\n"
                    f"START: You are beginning Project 2.\n"
                    f"First URL: {current_url}\n\n"
                    f"Fetch this URL, analyze the task, solve it, "
                    f"and POST your answer with ALL fields: email, secret, url, answer."
                )
            else:
                prompt = (
                    f"Continue to next task.\n"
                    f"URL: {current_url}\n\n"
                    f"Fetch the page, analyze, solve it, "
                    f"and POST with email, secret, url, answer.\n"
                    f"Format your answer EXACTLY as specified."
                )
            
            message_history.append({"role": "user", "content": prompt})
            
            logger.info("Calling Gemini...\n")
            
            try:
                text = ask_gemini_with_rate_limit(
                    [{"role": msg["role"], "parts": [msg["content"]]} for msg in message_history]
                )
            
            except Exception as e:
                error_type = ErrorHandler.classify_error(e)
                
                if error_type == 'quota_exceeded':
                    QUOTA_EXHAUSTED = True
                    logger.error(f"⚠️  QUOTA EXHAUSTED: {str(e)[:200]}")
                    break
                
                logger.error(f"Gemini error ({error_type}): {e}")
                message_history.append({"role": "assistant", "content": f"Error: {str(e)}"})
                continue
            
            if not text:
                logger.warning("No response from model")
                continue
            
            logger.info(f"Gemini response ({len(text)} chars):\n{text[:600]}\n")
            
            # Parse tool call
            tool_name, params = parse_tool_call(text)
            
            if tool_name:
                logger.info(f"Tool call detected: {tool_name}\n")
                tool_result = execute_tool(tool_name, params)
                
                # Analyze if HTML
                if tool_name == 'get_rendered_html' and "Error" not in tool_result:
                    analysis = ContentAnalyzer.analyze_task(tool_result)
                    logger.info(f"Task Analysis: Difficulty={analysis['difficulty']}, Format={analysis['format']}\n")
                
                message_history.append({"role": "assistant", "content": text})
                message_history.append({
                    "role": "user",
                    "content": f"Tool {tool_name} result:\n{tool_result}\n\nBased on this, solve and POST your answer."
                })
                
                # Trim history
                if len(message_history) > 20:
                    message_history = message_history[:2] + message_history[-15:]
                
                continue
            
            message_history.append({"role": "assistant", "content": text})
            
            # Check for completion
            if "END" in text or "complete" in text.lower() or "all tasks" in text.lower():
                logger.info("✅ Quiz marked complete by Gemini")
                break
            
            # Track attempts
            if current_url not in task_attempts:
                task_attempts[current_url] = 0
            task_attempts[current_url] += 1
            
            # Extract next URL
            next_url = extract_next_url(text, current_url)
            
            if next_url and next_url != current_url:
                current_url = next_url
                task_attempts[current_url] = 0
                logger.info(f"Moving to next task: {current_url}\n")
                continue
            
            if iteration > 20:
                logger.warning("Too many iterations without tool call")
                break
        
        logger.info(f"\n{'='*80}")
        logger.info("✅ Quiz solving session completed")
        logger.info(f"{'='*80}\n")
    
    except Exception as e:
        logger.error(f"Agent error: {e}")


# ============================================================================
# FASTAPI SERVER
# ============================================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model": "gemini-2.0-flash",
        "project": "project2",
        "quota_exhausted": QUOTA_EXHAUSTED
    }


@app.get("/")
def root():
    """Root endpoint."""
    return health()


@app.post("/solve-quiz")
async def solve_quiz(request: Request, background_tasks: BackgroundTasks):
    """
    Receive quiz URL and start solving.
    Validates credentials and starts quiz solver in background.
    """
    global QUOTA_EXHAUSTED
    
    try:
        data = await request.json()
    except:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    
    email = data.get("email")
    secret = data.get("secret")
    quiz_url = data.get("url")
    
    if not quiz_url or not secret or not email:
        raise HTTPException(status_code=400, detail="Missing email, url, or secret")
    
    if secret != EXPECTED_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")
    
    if QUOTA_EXHAUSTED:
        raise HTTPException(status_code=503, detail="API quota exhausted - upgrade to paid tier")
    
    # Reset state
    retry_cache.clear()
    url_time.clear()
    BASE64_STORE.clear()
    
    logger.info(f"✅ Authentication successful. Starting quiz solver for {quiz_url}...")
    
    os.environ["url"] = quiz_url
    os.environ["email"] = email
    url_time[quiz_url] = time.time()
    
    # Run agent in background
    background_tasks.add_task(run_agent, quiz_url)
    
    return JSONResponse(
        status_code=200,
        content={
            "correct": True,
            "url": quiz_url,
            "reason": None
        }
    )


@app.post("/quiz")
async def quiz_endpoint(request: Request, background_tasks: BackgroundTasks):
    """Alternative endpoint for /quiz (same as /solve-quiz)."""
    return await solve_quiz(request, background_tasks)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    logger.info(f"\n{'='*80}")
    logger.info(f"Starting TDS Project 2 Quiz Solver on port {port}")
    logger.info(f"{'='*80}\n")
    
    uvicorn.run(app, host="0.0.0.0", port=port)
