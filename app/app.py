"""
AUTONOMOUS QUIZ SOLVER - Production Ready
Self-contained app with all tools and advanced logic.
Fixed dependency conflict: langchain-core==0.2.0 (compatible with langgraph 0.0.69)
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
from langchain_core.messages import trim_messages, HumanMessage
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_google_genai import HarmBlockThreshold, HarmCategory

# Multimedia Processing (Graceful Fallbacks)
try:
    import pytesseract
    from PIL import Image
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False
    print("Warning: pytesseract/PIL not available. OCR will fail gracefully.")

try:
    import speech_recognition as sr
    from pydub import AudioSegment
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Warning: speech_recognition/pydub not available. Audio will fail gracefully.")

try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("Warning: playwright not available. Will use requests fallback.")

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    print("Warning: BeautifulSoup not available.")

# ============================================================================
# CONFIGURATION
# ============================================================================

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("QuizAgent")

EMAIL = os.getenv("EMAIL", "")
SECRET = os.getenv("SECRET", "")
EXPECTED_SECRET = os.getenv("SECRET", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

RECURSION_LIMIT = 5000
MAX_TOKENS = 60000
RETRY_LIMIT = 4

# ============================================================================
# SHARED STATE (In-Memory Database)
# ============================================================================

BASE64_STORE = {}    # Stores large base64 image data
URL_TIME = {}        # Tracks when we saw each URL
CACHE = defaultdict(int)  # Tracks retry attempts per URL

# ============================================================================
# TOOLS
# ============================================================================


@tool
def get_rendered_html(url: str) -> str:
    """
    Fetch and return the fully rendered HTML of a webpage.
    Uses Playwright if available, falls back to requests.
    """
    print(f"\nFetching: {url}")
    try:
        # Try Playwright first
        if PLAYWRIGHT_AVAILABLE:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                try:
                    page.goto(url, wait_until="networkidle", timeout=60000)
                except Exception:
                    pass
                content = page.content()
                browser.close()
        else:
            # Fallback to requests
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            content = response.text

        if len(content) > 200000:
            print("HTML too large, truncating...")
            content = content[:200000] + "... [TRUNCATED]"

        return content
    except Exception as e:
        return f"Error fetching page: {str(e)}"


@tool
def download_file(url: str, filename: str) -> str:
    """Download a file from a URL and save it to 'LLMFiles/' directory."""
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
def run_code(code: str) -> str:
    """
    Execute Python code safely.
    Pre-loads: pandas, numpy, hashlib, base64, requests, bs4, math, re, random.
    """
    try:
        print(f"\nExecuting code...")
        os.makedirs("LLMFiles", exist_ok=True)

        f_out = io.StringIO()
        f_err = io.StringIO()

        local_vars = {}
        global_vars = {
            "pd": __import__("pandas"),
            "np": __import__("numpy"),
            "requests": requests,
            "json": json,
            "print": print,
            "os": os,
            "hashlib": hashlib,
            "math": math,
            "base64": base64,
            "re": re,
            "random": __import__("random"),
        }

        if BS4_AVAILABLE:
            global_vars["BeautifulSoup"] = BeautifulSoup

        with redirect_stdout(f_out):
            try:
                old_cwd = os.getcwd()
                os.chdir("LLMFiles")
                exec(code, global_vars, local_vars)
            except Exception as e:
                f_err.write(str(e))
            finally:
                os.chdir(old_cwd)

        stdout = f_out.getvalue().strip()
        stderr = f_err.getvalue().strip()

        output = stdout if not stderr else f"ERROR: {stderr}"
        return output[:5000] if len(output) > 5000 else output

    except Exception as e:
        return f"Execution failed: {str(e)}"


@tool
def transcribe_audio(file_path: str) -> str:
    """Transcribe .mp3 or .wav files to text."""
    if not AUDIO_AVAILABLE:
        return "Error: Audio libraries not installed."

    try:
        if not file_path.startswith("LLMFiles") and not os.path.exists(file_path):
            file_path = os.path.join("LLMFiles", file_path)

        final_path = file_path

        if file_path.lower().endswith(".mp3"):
            sound = AudioSegment.from_mp3(file_path)
            final_path = file_path.replace(".mp3", ".wav")
            sound.export(final_path, format="wav")

        recognizer = sr.Recognizer()
        with sr.AudioFile(final_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)

        if final_path != file_path and os.path.exists(final_path):
            os.remove(final_path)

        return f"Transcription: {text}"
    except Exception as e:
        return f"Audio error: {str(e)}"


@tool
def ocr_image_tool(image_path: str) -> str:
    """Extract text from an image using OCR (Tesseract)."""
    if not PYTESSERACT_AVAILABLE:
        return "Error: Tesseract not installed."

    try:
        if not image_path.startswith("LLMFiles") and not os.path.exists(image_path):
            image_path = os.path.join("LLMFiles", image_path)

        img = Image.open(image_path).convert("RGB")
        text = pytesseract.image_to_string(img)
        return f"OCR Result: {text.strip()}"
    except Exception as e:
        return f"OCR error: {str(e)}"


@tool
def encode_image_to_base64(image_path: str) -> str:
    """Convert image to Base64 and store it. Returns a key instead of full string."""
    try:
        if not image_path.startswith("LLMFiles") and not os.path.exists(image_path):
            image_path = os.path.join("LLMFiles", image_path)

        with open(image_path, "rb") as f:
            raw = f.read()

        encoded = base64.b64encode(raw).decode("utf-8")
        key = str(uuid.uuid4())
        BASE64_STORE[key] = encoded

        return f"BASE64_KEY:{key}"
    except Exception as e:
        return f"Encoding error: {str(e)}"


@tool
def post_request(url: str, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> str:
    """Send HTTP POST request with retry logic and Base64 injection."""
    headers = headers or {"Content-Type": "application/json"}

    # Inject Base64 data if placeholder found
    ans = payload.get("answer")
    if isinstance(ans, str) and ans.startswith("BASE64_KEY:"):
        key = ans.split(":", 1)[1]
        if key in BASE64_STORE:
            payload["answer"] = BASE64_STORE[key]
            print("Injected Base64 data from store.")

    try:
        cur_url = os.getenv("url", "unknown")
        CACHE[cur_url] += 1

        # Log preview
        log_pl = payload.copy()
        if isinstance(log_pl.get("answer"), str) and len(log_pl["answer"]) > 50:
            log_pl["answer"] = log_pl["answer"][:50] + "..."
        print(f"\nPOST {url}\nPayload: {json.dumps(log_pl, indent=2)}")

        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        data = response.json()

        print(f"Response: {json.dumps(data, indent=2)}")

        next_url = data.get("url")
        if not next_url:
            return "Tasks completed"

        if next_url not in URL_TIME:
            URL_TIME[next_url] = time.time()

        # Retry Logic
        if not data.get("correct"):
            cur_time = time.time()
            prev_time = URL_TIME.get(cur_url, time.time())
            delay = cur_time - prev_time

            if CACHE[cur_url] >= RETRY_LIMIT or delay >= 180:
                print(f"Give up (Attempts: {CACHE[cur_url]}, Delay: {delay:.1f}s)")
            else:
                print("Answer incorrect. Retrying...")
                data["message"] = "Incorrect. Analyze again and retry."

        os.environ["url"] = str(data.get("url", ""))
        return json.dumps(data)

    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def add_dependencies(package_name: str) -> str:
    """Install Python packages dynamically."""
    try:
        if package_name in ["hashlib", "math", "json", "os", "sys"]:
            return "Already built-in."
        print(f"\nInstalling {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name, "-q"])
        return f"Installed {package_name}"
    except Exception as e:
        return f"Installation failed: {str(e)}"


# ============================================================================
# AGENT SETUP
# ============================================================================

ALL_TOOLS = [
    get_rendered_html,
    download_file,
    run_code,
    post_request,
    add_dependencies,
    ocr_image_tool,
    transcribe_audio,
    encode_image_to_base64,
]

# Safety settings
safety_settings = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

# Rate limiter (5 req/min)
rate_limiter = InMemoryRateLimiter(
    requests_per_second=5 / 60,
    check_every_n_seconds=1,
    max_bucket_size=5,
)

# Initialize LLM with gemini-1.5-flash
llm = init_chat_model(
    model_provider="google_genai",
    model="gemini-1.5-flash",
    rate_limiter=rate_limiter,
    safety_settings=safety_settings,
).bind_tools(ALL_TOOLS)

SYSTEM_PROMPT = f"""You are an autonomous quiz-solving agent.

YOUR GOAL:
1. Load each quiz page from the URL
2. Extract instructions and submit endpoint
3. Solve tasks exactly
4. Submit answers ONLY to the correct endpoint
5. Follow URLs until none remain, then output END

CRITICAL RULES:
- For base64 image encoding, ALWAYS use the "encode_image_to_base64" tool
- Never hallucinate URLs or fields
- Never shorten endpoints
- Always inspect server responses
- Use tools for HTML, downloading, rendering, OCR, or code execution
- Email: {EMAIL}
- Secret: {SECRET}

Start by fetching the page."""


class AgentState(TypedDict):
    messages: Annotated[List, add_messages]


def handle_malformed_node(state: AgentState):
    """Handles malformed JSON from LLM."""
    print("--- DETECTED MALFORMED JSON. REQUESTING RETRY ---")
    return {
        "messages": [
            HumanMessage(
                content="SYSTEM ERROR: Your last tool call was malformed (invalid JSON). "
                "Please rewrite the code and try again. Ensure you escape newlines and quotes correctly."
            )
        ]
    }


def agent_node(state: AgentState):
    """Main agent node with time handling and context trimming."""
    cur_time = time.time()
    cur_url = os.getenv("url")
    prev_time = URL_TIME.get(cur_url)
    offset = os.getenv("offset", "0")

    # Time handling
    if prev_time is not None:
        prev_time = float(prev_time)
        diff = cur_time - prev_time
        if diff >= 180 or (offset != "0" and (cur_time - float(offset)) > 90):
            print(f"Timeout exceeded ({diff}s) — submitting wrong answer")
            fail_instruction = (
                "You have exceeded the time limit. "
                "Immediately call post_request and submit a WRONG answer."
            )
            fail_msg = HumanMessage(content=fail_instruction)
            result = llm.invoke(state["messages"] + [fail_msg])
            return {"messages": [result]}

    # Context trimming
    trimmed_messages = trim_messages(
        messages=state["messages"],
        max_tokens=MAX_TOKENS,
        strategy="last",
        include_system=True,
        start_on="human",
        token_counter=llm,
    )

    # Ensure we have at least one human message
    has_human = any(msg.type == "human" for msg in trimmed_messages)
    if not has_human:
        print("WARNING: Context trimmed too far. Injecting reminder.")
        current_url = os.getenv("url", "Unknown")
        reminder = HumanMessage(content=f"Continue processing URL: {current_url}")
        trimmed_messages.append(reminder)

    print(f"--- INVOKING AGENT (Context: {len(trimmed_messages)} items) ---")
    result = llm.invoke(trimmed_messages)
    return {"messages": [result]}


def route(state):
    """Routing logic with malformed handling."""
    last = state["messages"][-1]

    # Check for malformed function calls
    if hasattr(last, "response_metadata"):
        if last.response_metadata.get("finish_reason") == "MALFORMED_FUNCTION_CALL":
            return "handle_malformed"

    # Check for valid tools
    tool_calls = getattr(last, "tool_calls", None)
    if tool_calls:
        print("Route → tools")
        return "tools"

    # Check for END
    content = getattr(last, "content", None)
    if isinstance(content, str) and content.strip() == "END":
        return END

    print("Route → agent")
    return "agent"


# Build graph
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(ALL_TOOLS))
graph.add_node("handle_malformed", handle_malformed_node)

graph.add_edge(START, "agent")
graph.add_edge("tools", "agent")
graph.add_edge("handle_malformed", "agent")

graph.add_conditional_edges(
    "agent",
    route,
    {
        "tools": "tools",
        "agent": "agent",
        "handle_malformed": "handle_malformed",
        END: END,
    },
)

agent_runner = graph.compile()


def run_agent(url: str):
    """Execute agent for given URL."""
    print(f"Starting task: {url}")
    BASE64_STORE.clear()
    URL_TIME.clear()
    CACHE.clear()
    os.environ["url"] = url
    URL_TIME[url] = time.time()

    initial_msg = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Start: {url}"},
    ]

    try:
        agent_runner.invoke(
            {"messages": initial_msg},
            config={"recursion_limit": RECURSION_LIMIT},
        )
        print("Task completed successfully!")
    except Exception as e:
        print(f"Task failed: {e}")


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

START_TIME = time.time()


@app.get("/healthz")
def healthz():
    return {"status": "ok", "uptime": int(time.time() - START_TIME)}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/solve-quiz")
async def solve_quiz(request: Request):
    """Synchronous endpoint."""
    try:
        data = await request.json()
        url = data.get("url")
        secret = data.get("secret")

        if not url or secret != EXPECTED_SECRET:
            raise HTTPException(status_code=400, detail="Invalid request")

        try:
            run_agent(url)
            return {"status": "completed"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/solve")
async def solve_async(request: Request, background_tasks: BackgroundTasks):
    """Asynchronous endpoint."""
    try:
        data = await request.json()
        url = data.get("url")
        secret = data.get("secret")

        if not url or secret != EXPECTED_SECRET:
            raise HTTPException(status_code=400, detail="Invalid request")

        background_tasks.add_task(run_agent, url)
        return {"status": "ok"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.getenv("PORT", 1010))
    uvicorn.run(app, host="0.0.0.0", port=port)
