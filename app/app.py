"""
OPTIMIZED LLM ANALYSIS QUIZ SOLVER
Single-file, lightweight implementation for Render deployment
- Minimal dependencies: FastAPI, Requests, BeautifulSoup, LangGraph core
- Direct Gemini API calls via langchain.chat_models.init_chat_model
- Modular tool design with clean separation
- 3-minute timeout handling
- Proper error handling and retry logic
"""

import os
import sys
import json
import time
import base64
import uuid
import subprocess
import logging
from typing import TypedDict, Annotated, List, Any, Dict, Optional
from collections import defaultdict
from dotenv import load_dotenv

# ============================================================================
# ESSENTIAL IMPORTS ONLY
# ============================================================================

# FastAPI
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# LangGraph minimal setup
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.messages import HumanMessage, trim_messages

# LangChain lightweight model init
from langchain.chat_models import init_chat_model

# External utilities
import requests
from bs4 import BeautifulSoup
import uvicorn

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

# State management - lightweight
BASE64_STORE = {}
URL_TIME = {}
CACHE = defaultdict(int)
RETRY_LIMIT = 4
MAX_TOKENS = 50000
RECURSION_LIMIT = 5000

# ============================================================================
# TOOLS - Lightweight implementations
# ============================================================================

@tool
def get_rendered_html(url: str) -> str:
    """Fetch and return HTML content from a URL."""
    try:
        print(f"\n📄 Fetching: {url}\n")
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Truncate large content
        content = response.text
        if len(content) > 200000:
            content = content[:200000] + "\n[CONTENT TRUNCATED]"
        
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
            if delay < 180:
                data["message"] = "Incorrect. Analyze again and retry."
                data["url"] = cur_url  # Retry same question
                os.environ["offset"] = str(URL_TIME.get(cur_url, time.time()))
            else:
                data["url"] = next_url  # Move to next
        
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
        if package_name in ["hashlib", "math", "json", "os", "sys", "base64"]:
            return "Already built-in."
        
        print(f"\n📦 Installing {package_name}...\n")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name, "-q"])
        return f"Successfully installed {package_name}"
    except Exception as e:
        return f"Error installing {package_name}: {str(e)}"


# OCR tool - optional, lazy loaded
@tool
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


# ============================================================================
# AGENT SETUP
# ============================================================================

TOOLS = [
    get_rendered_html,
    download_file,
    post_request,
    run_code,
    encode_image_to_base64,
    add_dependencies,
    ocr_image_tool,
]

# Rate limiter (4 requests per minute max)
rate_limiter = InMemoryRateLimiter(
    requests_per_second=4 / 60,
    check_every_n_seconds=1,
    max_bucket_size=4,
)

# Initialize LLM with lightweight init_chat_model
try:
    llm = init_chat_model(
        model_provider="google_genai",
        model="gemini-2.0-flash-exp",
        api_key=GOOGLE_API_KEY,
        rate_limiter=rate_limiter,
    ).bind_tools(TOOLS)
except Exception as e:
    logger.warning(f"⚠️ Primary model failed: {e}. Trying gemini-1.5-pro")
    llm = init_chat_model(
        model_provider="google_genai",
        model="gemini-1.5-pro",
        api_key=GOOGLE_API_KEY,
        rate_limiter=rate_limiter,
    ).bind_tools(TOOLS)


SYSTEM_PROMPT = f"""You are an autonomous quiz-solving agent with 7 specialized tools.

CRITICAL INSTRUCTIONS:

1. Load quiz pages from URLs using get_rendered_html
2. Extract instructions and identify the submit endpoint
3. Solve ALL tasks correctly using available tools
4. Submit answers ONLY to the correct endpoint with post_request
5. Follow new URLs until completion, then return END

TOOLS AVAILABLE:
- get_rendered_html: Fetch full page content
- download_file: Save files to LLMFiles/
- post_request: Submit answers
- run_code: Execute Python (has pandas, numpy, requests, BeautifulSoup pre-available)
- encode_image_to_base64: Convert images to Base64
- add_dependencies: Install packages
- ocr_image_tool: Extract text from images

RULES:
- For base64 generation of images, ALWAYS use encode_image_to_base64 tool, NEVER create your own
- Never hallucinate URLs or fields
- Never shorten endpoints
- Always inspect server responses
- Never stop early
- Email: {EMAIL}
- Secret: {SECRET}

Proceed immediately!"""


# ============================================================================
# LANGGRAPH STATE & NODES
# ============================================================================

class AgentState(TypedDict):
    """Agent state for message passing."""
    messages: Annotated[List, add_messages]


def handle_malformed_node(state: AgentState):
    """Handle malformed JSON responses from LLM."""
    print("--- DETECTED MALFORMED JSON. ASKING AGENT TO RETRY ---")
    return {
        "messages": [
            HumanMessage(
                content="SYSTEM ERROR: Your last tool call had invalid JSON. Please rewrite it and try again. Ensure you escape newlines and quotes correctly."
            )
        ]
    }


def agent_node(state: AgentState):
    """Execute agent step with timeout handling."""
    
    # TIME HANDLING
    cur_time = time.time()
    cur_url = os.getenv("url")
    prev_time = URL_TIME.get(cur_url)
    offset = os.getenv("offset", "0")
    
    if prev_time is not None:
        diff = cur_time - prev_time
        if diff >= 180 or (offset != "0" and (cur_time - float(offset)) > 90):
            print(f"Timeout exceeded ({diff}s) — submitting wrong answer")
            fail_instruction = """You have exceeded the time limit for this task (over 180 seconds).
            
Immediately call the post_request tool and submit a WRONG answer for the CURRENT quiz."""
            
            fail_msg = HumanMessage(content=fail_instruction)
            result = llm.invoke(state["messages"] + [fail_msg])
            return {"messages": [result]}
    
    # MESSAGE TRIMMING
    trimmed_messages = trim_messages(
        messages=state["messages"],
        max_tokens=MAX_TOKENS,
        strategy="last",
        include_system=True,
        start_on="human",
        token_counter=llm,
    )
    
    # Ensure human message exists
    has_human = any(msg.type == "human" for msg in trimmed_messages)
    if not has_human:
        print("WARNING: Context trimmed too far. Injecting reminder.")
        current_url = os.getenv("url", "Unknown URL")
        reminder = HumanMessage(
            content=f"Context cleared due to length. Continue processing URL: {current_url}"
        )
        trimmed_messages.append(reminder)
    
    print(f"--- INVOKING AGENT (Context: {len(trimmed_messages)} items) ---")
    result = llm.invoke(trimmed_messages)
    return {"messages": [result]}


def route(state):
    """Route logic: determine next step."""
    last = state["messages"][-1]
    
    # Check for malformed calls
    if hasattr(last, "response_metadata"):
        if last.response_metadata.get("finish_reason") == "MALFORMED_FUNCTION_CALL":
            return "handle_malformed"
    
    # Check for tool calls
    tool_calls = getattr(last, "tool_calls", None)
    if tool_calls:
        print("Route → tools")
        return "tools"
    
    # Check for END
    content = getattr(last, "content", None)
    if isinstance(content, str) and content.strip() == "END":
        return END
    
    if isinstance(content, list) and content and isinstance(content[0], dict):
        if content[0].get("text", "").strip() == "END":
            return END
    
    print("Route → agent")
    return "agent"


# Build graph
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(TOOLS))
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

agent_graph = graph.compile()


def run_agent(url: str):
    """Execute the quiz-solving agent."""
    print(f"\n🚀 Starting task: {url}\n")
    
    # Clear state
    BASE64_STORE.clear()
    URL_TIME.clear()
    CACHE.clear()
    
    os.environ["url"] = url
    os.environ["offset"] = "0"
    URL_TIME[url] = time.time()
    
    try:
        initial_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": url},
        ]
        
        agent_graph.invoke(
            {"messages": initial_messages},
            config={"recursion_limit": RECURSION_LIMIT},
        )
        
        print("\n✅ Task completed successfully!\n")
    except Exception as e:
        print(f"\n❌ Task failed: {e}\n")
        logger.error(f"Agent error: {e}")


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
    """Health check endpoint."""
    return {"status": "ok", "uptime": int(time.time() - START_TIME)}


@app.get("/healthz")
def healthz():
    """Liveness probe for Render."""
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
        raise HTTPException(status_code=400, detail="Invalid JSON")
    
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
    uvicorn.run(app, host="0.0.0.0", port=port)
