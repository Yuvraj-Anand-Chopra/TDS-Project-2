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
    """
    Fetch rendered HTML content from a given URL.
    
    This function makes an HTTP GET request to the provided URL and retrieves
    the HTML content. It includes proper headers and timeout handling.
    
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
    
    This function sends a JSON payload to the quiz server endpoint and handles
    the response. It logs the server response for debugging and manages retry
    logic based on correctness and timeout.
    
    Args:
        url (str): The endpoint URL to POST to
        payload (dict): The JSON payload containing email, secret, url, and answer
        
    Returns:
        str: JSON string containing server response
    """
    try:
        logger.info(f"Posting answer to: {url}")

        if isinstance(payload.get("answer"), str) and payload["answer"].startswith("BASE64_KEY:"):
            key = payload["answer"].split(":", 1)[1]
            if key in BASE64_STORE:
                payload["answer"] = BASE64_STORE[key]

        sending = {
            "email": payload.get("email", ""),
            "url": payload.get("url", ""),
            "answer": str(payload.get("answer", ""))[:100] if isinstance(payload.get("answer"), str) else payload.get("answer")
        }
        logger.info(f"Payload being sent: {json.dumps(sending, indent=2)}")

        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()

        logger.info(f"Server Response: {json.dumps(data, indent=2)}")

        cur_url = os.getenv("url", "")
        if cur_url not in retry_cache:
            retry_cache[cur_url] = 0
        retry_cache[cur_url] += 1

        next_url = data.get("url")
        correct = data.get("correct", False)

        if correct:
            logger.info("Answer is correct")
        else:
            logger.info("Answer is incorrect, will retry or move to next")

        if not correct and next_url:
            if next_url not in url_time:
                url_time[next_url] = time.time()

            cur_time = time.time()
            prev = url_time.get(cur_url, cur_time)
            delay = cur_time - prev

            if retry_cache[cur_url] >= RETRY_LIMIT or delay >= TIMEOUT_LIMIT:
                logger.info("Moving to next question")
                return json.dumps({"url": next_url, "message": "Move to next"})
            else:
                logger.info("Retrying this question")
                return json.dumps({"url": cur_url, "message": "Retry"})

        if not next_url:
            logger.info("Quiz chain complete")
            return json.dumps({"message": "Tasks completed"})

        return json.dumps(data)

    except Exception as e:
        logger.error(f"POST Error: {str(e)}")
        return json.dumps({"error": str(e)})


def run_code(code: str) -> str:
    """
    Execute Python code in a subprocess.
    
    This function creates a temporary Python file with the provided code and
    executes it in a subprocess. Output and errors are captured and returned.
    
    Args:
        code (str): Python code to execute
        
    Returns:
        str: Execution output (stdout + stderr)
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

        logger.info(f"Code executed successfully: {result[:200]}")
        return result or "Executed"
    except Exception as e:
        return f"Error: {str(e)}"


def download_file(url: str, filename: str) -> str:
    """
    Download a file from a URL and save it locally.
    
    This function downloads a file from the provided URL and saves it to the
    LLMFiles directory. It handles streaming for large files.
    
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
        return f"Error: {str(e)}"


def encode_image_to_base64(image_path: str) -> str:
    """
    Convert an image file to base64 encoding.
    
    This function reads an image file and encodes it as base64. The encoded
    value is stored in BASE64_STORE and a key is returned for later reference.
    
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
        logger.info(f"Image encoded successfully, key: {key[:8]}")
        return f"BASE64_KEY:{key}"
    except Exception as e:
        return f"Error: {str(e)}"


def add_dependencies(package_name: str) -> str:
    """
    Install a Python package using pip.
    
    This function installs a Python package on demand using subprocess.
    Built-in modules are skipped.
    
    Args:
        package_name (str): Name of the package to install
        
    Returns:
        str: Success or error message
    """
    try:
        if package_name in ["hashlib", "math", "json", "os", "sys"]:
            return "Built-in"
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name, "-q"])
        logger.info(f"Installed {package_name}")
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
# PARSER - HANDLES DICTIONARY PAYLOADS CORRECTLY
# ============================================================================

def parse_tool_call(response_text: str) -> tuple:
    """
    Parse Gemini's tool calls from response text.
    
    This function extracts tool calls formatted in code blocks and parses
    function names and parameters. It handles dictionary payloads with
    balanced braces, quoted strings, and unquoted values.
    
    Args:
        response_text (str): Gemini's response text
        
    Returns:
        tuple: (function_name, parameters_dict) or (None, None) if not found
    """
    
    if "```" not in response_text:
        return None, None
    
    code_match = re.search(r'```(?:python|tool_code)?\s*(.*?)```', response_text, re.DOTALL)
    if not code_match:
        return None, None
    
    code = code_match.group(1).strip()
    
    func_pattern = r'(\w+)\s*\((.*)\)'
    matches = re.findall(func_pattern, code, re.DOTALL)
    
    if not matches:
        return None, None
    
    func_name, params_str = matches[0]
    
    try:
        params = {}
        
        # Extract dictionary payloads by matching balanced braces
        dict_pattern = r'(\w+)\s*=\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})'
        for key, val in re.findall(dict_pattern, params_str):
            try:
                params[key] = json.loads(val)
                logger.info(f"Parsed dictionary: {key}=<dict with {len(json.loads(val))} fields>")
            except Exception as e:
                logger.warning(f"Could not parse dict {key}: {e}")
                params[key] = val
        
        # Extract quoted strings (don't override dict values)
        quoted_pattern = r'(\w+)\s*=\s*"([^"]*)"'
        for key, val in re.findall(quoted_pattern, params_str):
            if key not in params:
                params[key] = val
        
        # Extract unquoted values
        unquoted_pattern = r'(\w+)\s*=\s*([^\s,\)=]+)'
        for key, val in re.findall(unquoted_pattern, params_str):
            if key not in params and not val.startswith('"') and not val.startswith('{'):
                params[key] = val.strip(',)')
        
        # Single positional argument
        if not params:
            single_quote = re.search(r'"([^"]*)"', params_str)
            if single_quote:
                params['arg'] = single_quote.group(1)
        
        if params:
            logger.info(f"Parsed: {func_name}({list(params.keys())})")
            return func_name, params
    
    except Exception as e:
        logger.error(f"Parse error: {e}")
    
    return None, None


def execute_tool(tool_name: str, params: dict) -> str:
    """
    Execute a tool with the parsed parameters.
    
    This function calls the appropriate tool function based on tool_name and
    passes the parsed parameters to it.
    
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
        logger.error(f"Execution error: {e}")
        return f"Error: {str(e)}"

# ============================================================================
# GEMINI AI MODEL SETUP
# ============================================================================

genai.configure(api_key=GOOGLE_API_KEY)

print("Initializing Gemini...")
try:
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
# AUTONOMOUS QUIZ SOLVER AGENT
# ============================================================================

def run_agent(url: str):
    """
    Main agent loop for solving quiz questions.
    
    This function runs an iterative loop that:
    1. Fetches the quiz page using Gemini
    2. Analyzes the content
    3. Solves the problem
    4. Submits the answer
    5. Handles new URLs and retries
    
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

            response = model.generate_content(
                [{"role": msg["role"], "parts": [msg["content"]]} for msg in message_history],
                stream=False
            )

            if not response or not response.text:
                logger.warning("No response from model")
                continue

            text = response.text
            logger.info(f"Gemini response ({len(text)} chars):\n{text[:800]}\n")

            tool_name, params = parse_tool_call(text)
            
            if tool_name:
                logger.info(f"Tool call detected: {tool_name}")
                tool_result = execute_tool(tool_name, params)
                
                message_history.append({"role": "assistant", "content": text})
                message_history.append({
                    "role": "user",
                    "content": f"Tool {tool_name} returned:\n{tool_result}\n\nAnalyze this and continue solving. If correct, look for next URL. If not correct, try another answer."
                })
                
                if len(message_history) > 20:
                    logger.info("Trimming conversation history...")
                    message_history = message_history[:2] + message_history[-15:]
                
                continue
            
            message_history.append({"role": "assistant", "content": text})

            if "END" in text or "complete" in text.lower():
                logger.info("Quiz marked complete by Gemini")
                break

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
    url = data.get("url")

    if not url or not secret or not email:
        raise HTTPException(status_code=400, detail="Missing email, url, or secret")

    if secret != EXPECTED_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")

    retry_cache.clear()
    url_time.clear()
    BASE64_STORE.clear()

    logger.info("Authentication successful. Starting quiz solver in background...")
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
    logger.info(f"\nStarting AI Quiz Solver on port {port}...\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
