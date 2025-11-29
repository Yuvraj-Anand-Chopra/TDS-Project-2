"""
AI Quiz Solver - Production Version
Main FastAPI application with all improvements integrated
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
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Configuration
EMAIL = os.getenv("EMAIL", "")
SECRET = os.getenv("SECRET", "")
EXPECTED_SECRET = os.getenv("SECRET", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Global stores
BASE64_STORE = {}
url_time = {}
retry_cache = {}
RETRY_LIMIT = 4
TIMEOUT_LIMIT = 180

# ============================================================================
# ERROR HANDLING & CLASSIFICATION (NEW - IMPROVEMENT #4)
# ============================================================================

class ErrorHandler:
    """
    Unified error classification and recovery strategy.
    
    Classifies errors into types and provides appropriate recovery strategies
    for each error category.
    """
    
    ERROR_TYPES = {
        '429': 'rate_limit',
        '403': 'auth',
        '404': 'not_found',
        'timeout': 'timeout',
        'unexpected keyword': 'parser_error',
        'Resource exhausted': 'quota_exceeded',
        'Connection': 'connection_error',
    }
    
    @staticmethod
    def classify_error(error: Exception) -> str:
        """
        Classify the type of error that occurred.
        
        Args:
            error (Exception): The exception to classify
            
        Returns:
            str: Error type category
        """
        error_str = str(error).lower()
        for pattern, error_type in ErrorHandler.ERROR_TYPES.items():
            if pattern.lower() in error_str:
                logger.info(f"Error classified as: {error_type}")
                return error_type
        return 'unknown'
    
    @staticmethod
    def get_recovery_strategy(error_type: str) -> dict:
        """
        Get the recovery strategy for a specific error type.
        
        Args:
            error_type (str): Type of error
            
        Returns:
            dict: Recovery strategy with retry parameters
        """
        strategies = {
            'rate_limit': {
                'retry': True,
                'wait': 2,
                'backoff': 'exponential',
                'max_retries': 5,
                'action': 'exponential_backoff'
            },
            'parser_error': {
                'retry': True,
                'wait': 1,
                'action': 'restructure_input',
                'max_retries': 3,
                'backoff': 'linear'
            },
            'timeout': {
                'retry': True,
                'wait': 2,
                'backoff': 'exponential',
                'max_retries': 3,
                'action': 'retry_with_delay'
            },
            'connection_error': {
                'retry': True,
                'wait': 3,
                'backoff': 'exponential',
                'max_retries': 3,
                'action': 'retry_connection'
            },
            'auth': {
                'retry': False,
                'action': 'fail_fast',
                'log': 'critical'
            },
            'unknown': {
                'retry': False,
                'action': 'log_and_fail',
                'log': 'error'
            }
        }
        return strategies.get(error_type, strategies['unknown'])


# ============================================================================
# RATE LIMITING DECORATOR (NEW - IMPROVEMENT #2)
# ============================================================================

def rate_limit_aware(max_retries: int = 5):
    """
    Decorator that handles rate limiting with exponential backoff.
    
    Catches 429 errors and implements exponential backoff strategy.
    
    Args:
        max_retries (int): Maximum number of retries
        
    Returns:
        function: Decorated function with rate limit handling
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            wait_time = 1
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_str = str(e).lower()
                    if '429' in str(e) or 'resource exhausted' in error_str:
                        logger.warning(f"Rate limited on attempt {attempt + 1}. Waiting {wait_time}s...")
                        time.sleep(wait_time)
                        wait_time = min(wait_time * 2, 32)  # Cap at 32s
                        if attempt < max_retries - 1:
                            continue
                    raise
            return None
        return wrapper
    return decorator


# ============================================================================
# CONTENT ANALYZER (NEW - IMPROVEMENT #3)
# ============================================================================

class ContentAnalyzer:
    """
    Analyzes HTML content to understand task requirements and extract data.
    
    Can identify task types (sum, count, scrape) and extract relevant data
    from HTML responses.
    """
    
    @staticmethod
    def extract_numbers(html: str) -> list:
        """
        Extract all numbers from HTML content.
        
        Args:
            html (str): HTML content to parse
            
        Returns:
            list: List of integers found in content
        """
        try:
            text = BeautifulSoup(html, 'html.parser').get_text()
            numbers = re.findall(r'\d+', text)
            return [int(n) for n in numbers]
        except Exception as e:
            logger.error(f"Error extracting numbers: {e}")
            return []
    
    @staticmethod
    def extract_text_content(html: str) -> str:
        """
        Extract plain text from HTML.
        
        Args:
            html (str): HTML content
            
        Returns:
            str: Plain text content
        """
        try:
            return BeautifulSoup(html, 'html.parser').get_text(strip=True)
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            return ""
    
    @staticmethod
    def analyze_task(html: str) -> dict:
        """
        Determine what task the HTML is asking for.
        
        Analyzes content and identifies task type (sum, count, scrape, etc.)
        along with relevant data and computed answer.
        
        Args:
            html (str): HTML content
            
        Returns:
            dict: Task analysis with type, data, and answer
        """
        text_lower = html.lower()
        
        # Check for sum task
        if 'sum' in text_lower:
            numbers = ContentAnalyzer.extract_numbers(html)
            if numbers:
                answer = sum(numbers)
                logger.info(f"Task detected: SUM. Numbers: {numbers}, Answer: {answer}")
                return {
                    'type': 'sum',
                    'data': numbers,
                    'answer': str(answer),
                    'confidence': 'high'
                }
        
        # Check for count task
        if 'count' in text_lower:
            numbers = ContentAnalyzer.extract_numbers(html)
            if numbers:
                logger.info(f"Task detected: COUNT. Count: {len(numbers)}")
                return {
                    'type': 'count',
                    'data': numbers,
                    'answer': str(len(numbers)),
                    'confidence': 'high'
                }
        
        # Check for scrape task
        if 'scrape' in text_lower or 'extract' in text_lower:
            text_content = ContentAnalyzer.extract_text_content(html)
            logger.info(f"Task detected: SCRAPE")
            return {
                'type': 'scrape',
                'data': text_content[:100],
                'confidence': 'medium'
            }
        
        # Unknown task
        return {
            'type': 'unknown',
            'data': None,
            'confidence': 'low'
        }


# ============================================================================
# IMPROVED PARSER (IMPROVEMENT #1 - Fixed)
# ============================================================================

def parse_tool_call(response_text: str) -> Tuple[Optional[str], Optional[dict]]:
    """
    Parse Gemini's tool calls from response text.
    
    FIXED: Now uses strict boundary detection to avoid extracting URL
    query parameters as function arguments.
    
    Args:
        response_text (str): Gemini's response text
        
    Returns:
        tuple: (function_name, parameters_dict) or (None, None)
    """
    
    if "```" not in response_text:
        return None, None
    
    # Extract code block
    code_match = re.search(r'```(?:python|tool_code)?\s*(.*?)```', response_text, re.DOTALL)
    if not code_match:
        return None, None
    
    code = code_match.group(1).strip()
    
    # FIXED: Use strict boundary detection (^ and $ anchors)
    # This prevents matching URL parameters as function arguments
    func_pattern = r'^(\w+)\s*\((.*)\)$'
    matches = re.findall(func_pattern, code, re.MULTILINE | re.DOTALL)
    
    if not matches:
        return None, None
    
    func_name, params_str = matches[0]
    params = {}
    
    try:
        # Extract quoted strings first (highest priority)
        quoted_pattern = r'(\w+)\s*=\s*"([^"]*)"'
        for key, val in re.findall(quoted_pattern, params_str):
            params[key] = val
        
        # Extract dictionary payloads (complete JSON with balanced braces)
        dict_pattern = r'(\w+)\s*=\s*(\{(?:[^{}]|(?:\{[^{}]*\}))*\})'
        for key, val in re.findall(dict_pattern, params_str):
            try:
                params[key] = json.loads(val)
                logger.info(f"Parsed dict: {key}=<dict with {len(json.loads(val))} fields>")
            except Exception as e:
                logger.warning(f"Could not parse dict {key}: {e}")
                params[key] = val
        
        # Extract unquoted values (but not if they look like URL parameters)
        # Only match if not preceded by & and not containing special URL chars
        unquoted_pattern = r'(\w+)\s*=\s*([^\s,\)=]+)(?![\w&])'
        for key, val in re.findall(unquoted_pattern, params_str):
            if key not in params and not val.startswith('"') and not val.startswith('{'):
                # Don't include if it looks like a URL parameter
                if not (key in ['id', 'email', 'token'] and '=' in params_str[params_str.find(key):]):
                    params[key] = val.strip(',)')
        
        if params:
            logger.info(f"Parsed: {func_name}({list(params.keys())})")
            return func_name, params
    
    except Exception as e:
        logger.error(f"Parse error: {e}")
    
    return None, None


# ============================================================================
# TOOL IMPLEMENTATIONS
# ============================================================================

def get_rendered_html(url: str) -> str:
    """
    Fetch rendered HTML content from a given URL.
    
    Makes an HTTP GET request to the provided URL and retrieves
    the HTML content with proper headers and timeout handling.
    
    Args:
        url (str): The URL to fetch
        
    Returns:
        str: The HTML content of the page (limited to 200KB)
    """
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
    """
    Submit an answer to the quiz server via POST request.
    
    Sends a JSON payload to the quiz server endpoint, handles the response,
    logs server responses, and manages retry logic based on correctness
    and timeout.
    
    Args:
        url (str): The endpoint URL to POST to
        payload (dict): JSON payload (email, secret, url, answer)
        
    Returns:
        str: JSON string containing server response
    """
    try:
        logger.info(f"Posting answer to: {url}")
        
        # Handle base64 encoded answers
        if isinstance(payload.get("answer"), str) and payload["answer"].startswith("BASE64_KEY:"):
            key = payload["answer"].split(":", 1)[1]
            if key in BASE64_STORE:
                payload["answer"] = BASE64_STORE[key]
        
        # Log payload being sent (truncated for security)
        sending = {
            "email": payload.get("email", "")[:20] + "...",
            "url": payload.get("url", "")[:50] + "...",
            "answer": str(payload.get("answer", ""))[:100]
        }
        logger.info(f"Payload being sent: {json.dumps(sending, indent=2)}")
        
        # Make request
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        # Log full server response with formatting
        logger.info(f"Server Response: {json.dumps(data, indent=2)}")
        
        # Track retries
        cur_url = os.getenv("url", "")
        if cur_url not in retry_cache:
            retry_cache[cur_url] = 0
        retry_cache[cur_url] += 1
        
        # Check if answer is correct
        is_correct = data.get("correct", False)
        if is_correct:
            logger.info("Answer is correct")
        else:
            logger.warning(f"Answer is incorrect: {data.get('reason', 'Unknown')}")
        
        return json.dumps(data)
    
    except Exception as e:
        logger.error(f"POST Error: {str(e)}")
        return json.dumps({"error": str(e)})


def run_code(code: str) -> str:
    """
    Execute Python code in a subprocess.
    
    Creates a temporary Python file with the provided code and executes it
    in a subprocess. Output and errors are captured and returned.
    
    Args:
        code (str): Python code to execute
        
    Returns:
        str: Execution output (stdout + stderr, max 3000 chars)
    """
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
    """
    Download a file from a URL and save it locally.
    
    Downloads a file from the provided URL and saves it to the LLMFiles
    directory. Handles streaming for large files.
    
    Args:
        url (str): URL of the file to download
        filename (str): Name to save the file as
        
    Returns:
        str: Path to the saved file or error message
    """
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
    """
    Convert an image file to base64 encoding.
    
    Reads an image file and encodes it as base64. The encoded value is
    stored in BASE64_STORE and a key is returned for later reference.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Key reference in format BASE64_KEY:uuid
    """
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
    """
    Install a Python package using pip.
    
    Installs a Python package on demand using subprocess. Built-in
    modules are skipped.
    
    Args:
        package_name (str): Name of the package to install
        
    Returns:
        str: Success or error message
    """
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
    """
    Execute a tool with the parsed parameters.
    
    Calls the appropriate tool function based on tool_name and passes
    the parsed parameters to it. Includes error handling.
    
    Args:
        tool_name (str): Name of the tool to execute
        params (dict): Parsed parameters for the tool
        
    Returns:
        str: Tool execution result
    """
    if tool_name not in TOOLS_MAP:
        return f"Error: Unknown tool {tool_name}"
    
    try:
        logger.info(f"Executing tool: {tool_name}")
        tool_func = TOOLS_MAP[tool_name]
        
        # Route to correct tool based on parameters
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
# GEMINI AI MODEL SETUP
# ============================================================================

print("Initializing Gemini...")
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash-lite")
    print("Gemini 2.0 Flash Lite initialized")
except Exception as e:
    print(f"Trying Gemini 1.5 Flash...")
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        print("Gemini 1.5 Flash initialized")
    except Exception as e2:
        raise RuntimeError(f"Model initialization failed: {e2}")


# ============================================================================
# AUTONOMOUS QUIZ SOLVER AGENT (IMPROVED)
# ============================================================================

@rate_limit_aware(max_retries=5)
def ask_gemini_with_rate_limit(messages: list) -> str:
    """
    Ask Gemini a question with built-in rate limit handling.
    
    Wrapped with rate_limit_aware decorator for automatic retry
    with exponential backoff on 429 errors.
    
    Args:
        messages (list): List of message objects for Gemini
        
    Returns:
        str: Gemini's response text
    """
    response = model.generate_content(messages, stream=False)
    return response.text if response else ""


def run_agent(url: str):
    """
    Main agent loop for solving quiz questions.
    
    Iterative loop that:
    1. Fetches the quiz page using Gemini
    2. Analyzes the content (NEW - IMPROVEMENT #3)
    3. Solves the problem
    4. Submits the answer
    5. Handles new URLs and retries with improved error handling (NEW - IMPROVEMENT #4)
    
    Args:
        url (str): Starting URL for the quiz
    """
    logger.info(f"\nStarting quiz solver at: {url}\n")
    
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
                logger.warning(f"Timeout after {elapsed:.1f}s")
                break
            
            logger.info(f"\n{'='*80}")
            logger.info(f"ITERATION {iteration} | URL: {current_url} | Elapsed: {elapsed:.1f}s")
            logger.info(f"{'='*80}\n")
            
            if iteration == 1:
                prompt = f"{SYSTEM_PROMPT}\n\nSTART: Fetch and solve the quiz at {current_url}"
            else:
                prompt = f"Continue. Next question URL: {current_url}\n\nFetch the page, analyze, solve, and POST your answer with ALL fields (email, secret, url, answer)."
            
            message_history.append({"role": "user", "content": prompt})
            
            logger.info("Asking Gemini...\n")
            
            try:
                text = ask_gemini_with_rate_limit(
                    [{"role": msg["role"], "parts": [msg["content"]]} for msg in message_history]
                )
            except Exception as e:
                error_type = ErrorHandler.classify_error(e)
                strategy = ErrorHandler.get_recovery_strategy(error_type)
                logger.error(f"Gemini error ({error_type}): {e}")
                
                if strategy['retry']:
                    message_history.append({"role": "assistant", "content": str(e)})
                    continue
                else:
                    break
            
            if not text:
                logger.warning("No response from model")
                continue
            
            logger.info(f"Gemini response ({len(text)} chars):\n{text[:800]}\n")
            
            # Parse tool call
            tool_name, params = parse_tool_call(text)
            
            if tool_name:
                logger.info(f"Tool call detected: {tool_name}")
                tool_result = execute_tool(tool_name, params)
                
                # Analyze content if HTML retrieved (NEW - IMPROVEMENT #3)
                if tool_name == 'get_rendered_html':
                    analysis = ContentAnalyzer.analyze_task(tool_result)
                    logger.info(f"Content analysis: {analysis}")
                    if analysis['type'] != 'unknown':
                        tool_result += f"\n\n[ANALYSIS: Task type={analysis['type']}, Answer suggestion={analysis.get('answer', 'N/A')}]"
                
                message_history.append({"role": "assistant", "content": text})
                message_history.append({
                    "role": "user",
                    "content": f"Tool {tool_name} returned:\n{tool_result}\n\nAnalyze this and continue solving. If correct, look for next URL. If not correct, try another answer."
                })
                
                # Trim history to prevent memory bloat
                if len(message_history) > 20:
                    logger.info("Trimming conversation history...")
                    message_history = message_history[:2] + message_history[-15:]
                
                continue
            
            message_history.append({"role": "assistant", "content": text})
            
            if "END" in text or "complete" in text.lower():
                logger.info("Quiz marked complete by Gemini")
                break
            
            # Extract URLs to continue chain
            urls = re.findall(r'https?://[^\s"\)\]]+', text)
            if urls:
                next_url = urls[-1]
                if next_url != current_url:
                    current_url = next_url
                    if current_url not in url_time:
                        url_time[current_url] = time.time()
                    logger.info(f"Next URL found: {current_url}")
                    continue
            
            if iteration > 15:
                logger.warning("Too many iterations without tool call")
                break
        
        elapsed = time.time() - url_time.get(current_url, time.time())
        logger.info(f"\nQuiz solving session completed in {elapsed:.1f}s\n")
    
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
    """
    Health check endpoint.
    
    Returns the current status of the server and the Gemini model being used.
    """
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
    """
    Receive quiz URL and start solving.
    
    This endpoint receives a POST request with email, secret, and quiz URL.
    It validates the secret and starts the quiz solver in the background.
    The response is sent immediately while the solver runs in the background.
    
    Args:
        request (Request): FastAPI request object
        background_tasks (BackgroundTasks): Background task scheduler
        
    Returns:
        JSONResponse: Acknowledgment response with status code 200
        
    Raises:
        HTTPException: If JSON is invalid, missing fields, or secret is wrong
    """
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
    
    # Reset state
    retry_cache.clear()
    url_time.clear()
    BASE64_STORE.clear()
    
    logger.info("Authentication successful. Starting quiz solver in background...")
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


if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    logger.info(f"\nStarting AI Quiz Solver on port {port}...\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
