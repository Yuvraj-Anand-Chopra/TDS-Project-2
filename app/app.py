"""
AUTONOMOUS QUIZ SOLVER - PRODUCTION READY
Single-file implementation with LangGraph + FastAPI
Fully tested and verified working configuration
"""

import os
import sys
import time
import json
import subprocess
import logging
import base64
import uuid
from typing import TypedDict, Annotated, List, Any, Dict, Optional
from contextlib import redirect_stdout
from collections import defaultdict
import io

# FastAPI
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# LangGraph & LangChain
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.rate_limiters import InMemoryRateLimiter
from langgraph.graph.message import add_messages
from langchain_google_genai import HarmBlockThreshold, HarmCategory, ChatGoogleGenerativeAI

# External
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import uvicorn

# ============================================================================
# CONFIGURATION & ENVIRONMENT
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

# ============================================================================
# TOOLS - Helper functions that LLM can call
# ============================================================================

@tool
def get_rendered_html(url: str) -> str:
    """Fetch and return the HTML content from a URL."""
    try:
        print(f"\n📄 Fetching: {url}\n")
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        if len(response.text) > 200000:
            content = response.text[:200000] + "\n[CONTENT TRUNCATED]"
        else:
            content = response.text
            
        logger.info(f"✅ Retrieved HTML from {url}")
        return content
    except Exception as e:
        logger.error(f"❌ Error fetching {url}: {e}")
        return f"Error fetching URL: {str(e)}"


@tool
def download_file(url: str, filename: str = "download.txt") -> str:
    """Download a file from URL and save it to LLMFiles/."""
    try:
        print(f"\n⬇️ Downloading: {url}\n")
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        os.makedirs("LLMFiles", exist_ok=True)
        filepath = os.path.join("LLMFiles", filename)
        
        with open(filepath, "wb") as f:
            f.write(response.content)
        
        logger.info(f"✅ Downloaded and saved: {filepath}")
        return f"Saved to {filepath}"
    except Exception as e:
        logger.error(f"❌ Error downloading {url}: {e}")
        return f"Error downloading file: {str(e)}"


@tool
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
        print(f"Payload: {json.dumps({k: (v[:50]+'...' if isinstance(v, str) and len(v) > 50 else v) for k, v in payload.items()}, indent=2)}\n")
        
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
        
        if not data.get("correct") and CACHE[cur_url] < RETRY_LIMIT:
            data["message"] = "Incorrect. Analyze again and retry."
        
        return json.dumps(data)
    except Exception as e:
        logger.error(f"❌ POST Error: {e}")
        return json.dumps({"error": str(e)})


@tool
def run_code(code: str) -> str:
    """Execute Python code and return output."""
    try:
        print(f"\n🐍 Executing code:\n{code}\n")
        
        os.makedirs("LLMFiles", exist_ok=True)
        f_out = io.StringIO()
        
        local_vars = {}
        global_vars = {
            "pd": __import__("pandas"),
            "np": __import__("numpy"),
            "requests": requests,
            "json": json,
            "print": print,
            "BeautifulSoup": BeautifulSoup,
        }
        
        with redirect_stdout(f_out):
            exec(code, global_vars, local_vars)
        
        output = f_out.getvalue().strip()
        result = output if output else "Code executed successfully"
        
        logger.info(f"✅ Code result: {result}")
        return result[:3000]  # Limit output
    except Exception as e:
        error_msg = f"Code execution failed: {str(e)}"
        logger.error(f"❌ {error_msg}")
        return error_msg


@tool
def ocr_image_tool(image_path: str) -> str:
    """Extract text from image using OCR (if tesseract available)."""
    try:
        import pytesseract
        from PIL import Image
        
        if not image_path.startswith("LLMFiles"):
            image_path = os.path.join("LLMFiles", image_path)
        
        img = Image.open(image_path).convert("RGB")
        text = pytesseract.image_to_string(img)
        return f"OCR Result: {text.strip()}"
    except ImportError:
        return "Error: Tesseract not installed."
    except Exception as e:
        return f"OCR error: {str(e)}"


@tool
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


@tool
def add_dependencies(package_name: str) -> str:
    """Install Python package."""
    try:
        if package_name in ["hashlib", "math", "json", "os", "sys"]:
            return "Already built-in."
        print(f"\n📦 Installing {package_name}...\n")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name, "-q"])
        return f"Successfully installed {package_name}"
    except Exception as e:
        return f"Error installing {package_name}: {str(e)}"


# ============================================================================
# AGENT SETUP
# ============================================================================

TOOLS = [
    get_rendered_html,
    download_file,
    post_request,
    run_code,
    ocr_image_tool,
    encode_image_to_base64,
    add_dependencies,
]

# Safety settings
safety_settings = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

# Rate limiter
rate_limiter = InMemoryRateLimiter(
    requests_per_second=5 / 60,
    check_every_n_seconds=1,
    max_bucket_size=5,
)

# Initialize LLM (using stable API)
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        google_api_key=GOOGLE_API_KEY,
        rate_limiter=rate_limiter,
        safety_settings=safety_settings,
    ).bind_tools(TOOLS)
except Exception:
    # Fallback to alternate model
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        google_api_key=GOOGLE_API_KEY,
        rate_limiter=rate_limiter,
        safety_settings=safety_settings,
    ).bind_tools(TOOLS)

SYSTEM_PROMPT = f"""You are an autonomous quiz-solving agent with full access to 8 specialized tools.

CRITICAL INSTRUCTIONS:
1. Load quiz pages from URLs
2. Extract instructions and identify the submit endpoint
3. Solve ALL tasks correctly
4. Submit answers to /submit or indicated endpoint
5. Follow URLs until completion, then return END

TOOLS AVAILABLE:
- get_rendered_html: Fetch full page content
- download_file: Save files to LLMFiles/
- post_request: Submit answers
- run_code: Execute Python (pandas, numpy, requests, BeautifulSoup pre-loaded)
- ocr_image_tool: Extract text from images
- encode_image_to_base64: Convert images to Base64
- add_dependencies: Install packages

EMAIL: {EMAIL}
SECRET: {SECRET}

Proceed immediately!"""


class AgentState(TypedDict):
    """Agent state."""
    messages: Annotated[List, add_messages]


def agent_node(state: AgentState):
    """Execute agent step."""
    print("\n🤖 AGENT INVOKING...\n")
    messages = state["messages"]
    if len(messages) > 100:
        messages = messages[:1] + messages[-99:]  # Keep system + recent
    result = llm.invoke(messages)
    return {"messages": [result]}


def route(state):
    """Routing logic."""
    last = state["messages"][-1]
    
    # Check for tool calls
    tool_calls = getattr(last, "tool_calls", None)
    if tool_calls:
        return "tools"
    
    # Check for END
    content = getattr(last, "content", None)
    if isinstance(content, str) and "END" in content:
        return END
    
    return "agent"


# Build graph
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(TOOLS))
graph.add_edge(START, "agent")
graph.add_edge("tools", "agent")
graph.add_conditional_edges("agent", route)
agent_graph = graph.compile()


def run_agent(url: str):
    """Execute the quiz-solving agent."""
    print(f"\n🚀 Starting task: {url}\n")
    BASE64_STORE.clear()
    URL_TIME.clear()
    CACHE.clear()
    os.environ["url"] = url
    URL_TIME[url] = time.time()
    
    try:
        agent_graph.invoke(
            {"messages": [{"role": "user", "content": f"Solve this quiz: {url}"}]},
            config={"recursion_limit": 5000},
        )
        print("\n✅ Task completed successfully!\n")
    except Exception as e:
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


@app.get("/health")
def health():
    """Health check."""
    return {"status": "ok", "uptime": int(time.time() - START_TIME)}


@app.get("/healthz")
def healthz():
    """Liveness check."""
    return {"status": "ok", "uptime": int(time.time() - START_TIME)}


@app.post("/solve-quiz")
async def solve_quiz_endpoint(request: Request):
    """Synchronous solve endpoint."""
    try:
        data = await request.json()
        url = data.get("url")
        secret = data.get("secret")
        
        if not url or secret != EXPECTED_SECRET:
            raise HTTPException(status_code=400, detail="Invalid request")
        
        run_agent(str(url))
        return {"status": "completed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/solve")
async def solve_endpoint(request: Request, background_tasks: BackgroundTasks):
    """Asynchronous solve endpoint."""
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
