"""
AI Quiz Solver - Project 2 Edition
Enhanced FastAPI application for Project 2 with difficulty-aware solving
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
RETRY_LIMIT = 4
TIMEOUT_LIMIT = 300  # Increased for longer Project 2 chains


# ============================================================================
# ERROR HANDLING & CLASSIFICATION
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
        'wrong': 'answer_incorrect',
    }
    
    @staticmethod
    def classify_error(error: Exception) -> str:
        """Classify the type of error that occurred."""
        error_str = str(error).lower()
        for pattern, error_type in ErrorHandler.ERROR_TYPES.items():
            if pattern.lower() in error_str:
                logger.info(f"Error classified as: {error_type}")
                return error_type
        return 'unknown'
    
    @staticmethod
    def get_recovery_strategy(error_type: str) -> dict:
        """Get the recovery strategy for a specific error type."""
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
            'answer_incorrect': {
                'retry': True,
                'wait': 1,
                'action': 'retry_different_approach',
                'max_retries': 3,
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
# RATE LIMITING DECORATOR
# ============================================================================

def rate_limit_aware(max_retries: int = 5):
    """Decorator that handles rate limiting with exponential backoff."""
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
                        wait_time = min(wait_time * 2, 32)
                        if attempt < max_retries - 1:
                            continue
                    raise
            return None
        return wrapper
    return decorator


# ============================================================================
# ENHANCED CONTENT ANALYZER (Project 2 Edition)
# ============================================================================

class ContentAnalyzer:
    """
    Analyzes HTML content to understand task requirements and extract data.
    
    Enhanced for Project 2 to detect:
    - Difficulty levels (1-5)
    - Personalization status
    - Required answer format
    - Task instructions and hints
    """
    
    @staticmethod
    def extract_numbers(html: str) -> list:
        """Extract all numbers from HTML content."""
        try:
            text = BeautifulSoup(html, 'html.parser').get_text()
            numbers = re.findall(r'\d+', text)
            return [int(n) for n in numbers]
        except Exception as e:
            logger.error(f"Error extracting numbers: {e}")
            return []
    
    @staticmethod
    def extract_text_content(html: str) -> str:
        """Extract plain text from HTML."""
        try:
            return BeautifulSoup(html, 'html.parser').get_text(strip=True)
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            return ""
    
    @staticmethod
    def extract_difficulty_level(html: str) -> int:
        """
        Extract difficulty level from HTML (1-5).
        
        Args:
            html (str): HTML content
            
        Returns:
            int: Difficulty level 1-5, default 1 if not found
        """
        for level in range(1, 6):
            patterns = [
                f"Difficulty {level}",
                f"difficulty {level}",
                f"DIFFICULTY {level}",
                f"Difficulty: {level}",
                f"Level {level}",
            ]
            for pattern in patterns:
                if pattern in html:
                    logger.info(f"Difficulty level detected: {level}")
                    return level
        return 1  # Default to Easy
    
    @staticmethod
    def extract_format_requirement(html: str) -> str:
        """
        Extract required answer format from task page.
        
        Looks for patterns like "Answer as [FORMAT]" or "Required format: [FORMAT]"
        
        Args:
            html (str): HTML content
            
        Returns:
            str: Required format or "unknown"
        """
        patterns = [
            r'answer\s+as\s+([^\.:\n]+)',
            r'required\s+format[:\s]+([^\.:\n]+)',
            r'format:\s*([^\.:\n]+)',
            r'submit\s+as\s+([^\.:\n]+)',
            r'answer\s*:\s*([^\.:\n]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, html, re.IGNORECASE)
            if match:
                format_str = match.group(1).strip()
                logger.info(f"Answer format detected: {format_str}")
                return format_str
        
        return "unknown"
    
    @staticmethod
    def check_personalization(html: str) -> dict:
        """
        Check if task is personalized to email.
        
        Returns dict with personalization status and email if mentioned.
        
        Args:
            html (str): HTML content
            
        Returns:
            dict: {'personalized': bool, 'email': str or None, 'note': str}
        """
        # Check for explicit "Not personalized" marker
        if "Not personalized" in html:
            return {
                'personalized': False,
                'email': None,
                'note': 'This task accepts same answer for everyone'
            }
        
        # Check if email is mentioned
        if EMAIL and EMAIL in html:
            return {
                'personalized': True,
                'email': EMAIL,
                'note': f'This task is personalized to {EMAIL}'
            }
        
        # Check for generic email pattern mention
        email_match = re.search(r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', html)
        if email_match:
            return {
                'personalized': True,
                'email': email_match.group(1),
                'note': 'This task mentions an email address'
            }
        
        # Default: assume personalized if email could be relevant
        return {
            'personalized': False,
            'email': None,
            'note': 'No personalization markers found'
        }
    
    @staticmethod
    def extract_hints(html: str) -> list:
        """
        Extract hints or examples from the task page.
        
        Looks for common hint patterns like "Example:", "Hint:", "Note:", etc.
        
        Args:
            html (str): HTML content
            
        Returns:
            list: List of hints/notes found
        """
        hints = []
        patterns = [
            r'(?:Example|Hint|Note|Tip|Important)[:\s]+([^\n]+)',
            r'(?:e\.g\.|For example)[:\s]+([^\n]+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, html, re.IGNORECASE)
            hints.extend(matches)
        
        return list(set(hints))[:5]  # Return unique hints, max 5
    
    @staticmethod
    def analyze_task(html: str) -> dict:
        """
        Comprehensive task analysis for Project 2.
        
        Analyzes HTML and returns all relevant task metadata.
        
        Args:
            html (str): HTML content
            
        Returns:
            dict: Complete task analysis
        """
        difficulty = ContentAnalyzer.extract_difficulty_level(html)
        format_req = ContentAnalyzer.extract_format_requirement(html)
        personalization = ContentAnalyzer.check_personalization(html)
        hints = ContentAnalyzer.extract_hints(html)
        numbers = ContentAnalyzer.extract_numbers(html)
        
        # Determine task type based on content
        task_type = 'unknown'
        answer_hint = None
        
        if 'sum' in html.lower():
            task_type = 'sum'
            if numbers:
                answer_hint = str(sum(numbers))
        elif 'count' in html.lower():
            task_type = 'count'
            if numbers:
                answer_hint = str(len(numbers))
        elif 'scrape' in html.lower() or 'extract' in html.lower():
            task_type = 'scrape'
        elif 'calculate' in html.lower():
            task_type = 'calculation'
        
        analysis = {
            'difficulty': difficulty,
            'difficulty_name': DIFFICULTY_LEVELS.get(difficulty, {}).get('name', 'Unknown'),
            'next_url_logic': 'always' if DIFFICULTY_LEVELS[difficulty]['next_url_always_shown'] else 'only_if_correct',
            'format': format_req,
            'personalized': personalization['personalized'],
            'personalization_note': personalization['note'],
            'task_type': task_type,
            'hints': hints,
            'numbers': numbers,
            'answer_suggestion': answer_hint,
        }
        
        logger.info(f"Task Analysis: {json.dumps({k: v for k, v in analysis.items() if k not in ['hints', 'numbers']}, indent=2)}")
        
        return analysis


# ============================================================================
# IMPROVED PARSER
# ============================================================================

def parse_tool_call(response_text: str) -> Tuple[Optional[str], Optional[dict]]:
    """
    Parse Gemini's tool calls from response text with strict boundary detection.
    
    Args:
        response_text (str): Gemini's response text
        
    Returns:
        tuple: (function_name, parameters_dict) or (None, None)
    """
    
    if "```" not in response_text:
        return None, None
    
    code_match = re.search(r'```(?:python|tool_code)?\s*(.*?)```', response_text, re.DOTALL)
    if not code_match:
        return None, None
    
    code = code_match.group(1).strip()
    func_pattern = r'^(\w+)\s*\((.*)\)$'
    matches = re.findall(func_pattern, code, re.MULTILINE | re.DOTALL)
    
    if not matches:
        return None, None
    
    func_name, params_str = matches[0]
    params = {}
    
    try:
        # Extract quoted strings first
        quoted_pattern = r'(\w+)\s*=\s*"([^"]*)"'
        for key, val in re.findall(quoted_pattern, params_str):
            params[key] = val
        
        # Extract dictionary payloads
        dict_pattern = r'(\w+)\s*=\s*(\{(?:[^{}]|(?:\{[^{}]*\}))*\})'
        for key, val in re.findall(dict_pattern, params_str):
            try:
                params[key] = json.loads(val)
            except Exception as e:
                logger.warning(f"Could not parse dict {key}: {e}")
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
        
        sending = {
            "email": payload.get("email", "")[:20] + "...",
            "url": payload.get("url", "")[:50] + "...",
            "answer": str(payload.get("answer", ""))[:100]
        }
        logger.info(f"Payload being sent: {json.dumps(sending, indent=2)}")
        
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        logger.info(f"Server Response: {json.dumps(data, indent=2)}")
        
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
# AUTONOMOUS QUIZ SOLVER AGENT (Project 2 Edition)
# ============================================================================

@rate_limit_aware(max_retries=5)
def ask_gemini_with_rate_limit(messages: list) -> str:
    """Ask Gemini a question with built-in rate limit handling."""
    response = model.generate_content(messages, stream=False)
    return response.text if response else ""


def run_agent(url: str):
    """
    Main agent loop for solving Project 2 quiz questions.
    
    Enhanced for Project 2 with:
    - Difficulty level awareness
    - Format compliance checking
    - Personalization handling
    - Retry logic based on difficulty
    """
    logger.info(f"\nStarting Project 2 quiz solver at: {url}\n")
    
    current_url = url
    iteration = 0
    max_iterations = 100  # Increased for longer Project 2 chains
    message_history = []
    task_attempts = {}  # Track attempts per task URL
    
    SYSTEM_PROMPT = (
        "You are an expert PROJECT 2 SOLVER. CRITICAL DIFFERENCES FROM DEMO:\n\n"
        
        "PROJECT 2 RULES:\n"
        "1. Difficulty 1-2: Next URL always shown (even if wrong answer)\n"
        "2. Difficulty 3-5: Next URL ONLY if answer is CORRECT\n"
        "3. Each task page explicitly states required answer format\n"
        "4. Some tasks are personalized to email, some are universal\n"
        "5. Must solve correctly on difficulties 3-5 or get stuck\n\n"
        
        "CRITICAL SUCCESS FACTORS:\n"
        "- Read answer format requirement CAREFULLY\n"
        "- Check if task is personalized or not\n"
        "- Understand the actual task (sum, extract, calculate, etc)\n"
        "- Format your answer EXACTLY as required\n"
        "- On difficulty 3-5, analyze failures and retry with different approaches\n\n"
        
        "WORKFLOW:\n"
        "1. Call get_rendered_html to fetch the task page\n"
        "2. READ the task carefully\n"
        "3. Identify: difficulty, format requirement, personalization status\n"
        "4. Analyze what's being asked\n"
        "5. Solve the problem CORRECTLY\n"
        "6. Format answer exactly as required\n"
        "7. POST to https://tds-llm-analysis.s-anand.net/submit\n"
        "8. If WRONG on difficulty 3-5: analyze and retry (different approach)\n"
        "9. When CORRECT: continue to next URL\n"
        "10. Repeat until all tasks completed\n\n"
        
        "CRITICAL WARNINGS:\n"
        "- Difficulty 3-5 requires CORRECT answers - wrong = STUCK\n"
        "- Answer format is NOT optional - must be exact\n"
        "- Some tasks personalized - read carefully\n"
        "- If wrong: analyze the error and try completely different approach\n\n"
        
        f"YOUR CREDENTIALS:\n"
        f"  Email: {EMAIL}\n"
        f"  Secret: {SECRET}\n"
        f"  Submit to: {SUBMIT_URL}\n\n"
        
        "Available tools: get_rendered_html, post_request, run_code, download_file"
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
            
            # Track attempts for this URL
            attempts = task_attempts.get(current_url, 0)
            if attempts > 0:
                logger.info(f"Attempt {attempts + 1} for this task")
            
            if iteration == 1:
                prompt = (
                    f"{SYSTEM_PROMPT}\n\n"
                    f"START: You are beginning Project 2.\n"
                    f"First URL: {current_url}\n\n"
                    f"Fetch this URL and analyze the task carefully.\n"
                    f"Read ALL instructions before answering."
                )
            else:
                prompt = (
                    f"Continue to next task.\n"
                    f"URL: {current_url}\n\n"
                    f"Fetch the page, analyze the task, solve it, "
                    f"and POST your answer with ALL fields: "
                    f"email, secret, url, answer.\n"
                    f"Format your answer EXACTLY as specified."
                )
            
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
                
                # Analyze content if HTML retrieved
                if tool_name == 'get_rendered_html':
                    analysis = ContentAnalyzer.analyze_task(tool_result)
                    logger.info(f"Task Analysis: Difficulty={analysis['difficulty']}, "
                               f"Format={analysis['format']}, "
                               f"Personalized={analysis['personalized']}, "
                               f"Type={analysis['task_type']}")
                    
                    # Add analysis to tool result for Gemini
                    tool_result += (
                        f"\n\n[TASK METADATA]\n"
                        f"Difficulty: {analysis['difficulty']} ({analysis['difficulty_name']})\n"
                        f"Next URL will be {'always' if analysis['next_url_logic'] == 'always' else 'ONLY if CORRECT'} revealed\n"
                        f"Required Format: {analysis['format']}\n"
                        f"Personalized: {analysis['personalized']}\n"
                        f"Task Type: {analysis['task_type']}\n"
                        f"Personalization Note: {analysis['personalization_note']}\n"
                    )
                    if analysis['hints']:
                        tool_result += f"Hints: {', '.join(analysis['hints'][:3])}\n"
                    if analysis['answer_suggestion']:
                        tool_result += f"Answer Suggestion: {analysis['answer_suggestion']}\n"
                    tool_result += "[/TASK METADATA]"
                
                message_history.append({"role": "assistant", "content": text})
                message_history.append({
                    "role": "user",
                    "content": f"Tool {tool_name} result:\n{tool_result}\n\n"
                              f"Based on this, what should be your next action? "
                              f"Solve the task and POST your answer if ready."
                })
                
                # Trim history to prevent memory bloat
                if len(message_history) > 20:
                    logger.info("Trimming conversation history...")
                    message_history = message_history[:2] + message_history[-15:]
                
                continue
            
            message_history.append({"role": "assistant", "content": text})
            
            # Check if task marked complete
            if "END" in text or "complete" in text.lower() or "all tasks" in text.lower():
                logger.info("Quiz marked complete by Gemini")
                break
            
            # Track attempts
            if current_url not in task_attempts:
                task_attempts[current_url] = 0
            task_attempts[current_url] += 1
            
            # Extract next URLs from response
            urls = re.findall(r'https?://[^\s"\)\]]+', text)
            if urls:
                next_url = urls[-1]
                if next_url != current_url:
                    current_url = next_url
                    task_attempts[current_url] = 0  # Reset attempts for new task
                    if current_url not in url_time:
                        url_time[current_url] = time.time()
                    logger.info(f"Next URL extracted: {current_url}")
                    continue
            
            if iteration > 20:
                logger.warning("Too many iterations without tool call")
                break
        
        elapsed = time.time() - url_time.get(current_url, time.time())
        logger.info(f"\nProject 2 quiz solving session completed in {elapsed:.1f}s\n")
    
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
        "project": "project2"
    }


@app.get("/")
def root():
    """Root endpoint."""
    return health()


@app.post("/solve-quiz")
async def solve_quiz(request: Request, background_tasks: BackgroundTasks):
    """
    Receive quiz URL and start solving.
    
    This endpoint receives a POST request with email, secret, and quiz URL.
    It validates the secret and starts the quiz solver in the background.
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
    
    logger.info(f"Project 2 authentication successful. Starting quiz solver for {quiz_url}...")
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
    logger.info(f"\nStarting Project 2 Quiz Solver on port {port}...\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
