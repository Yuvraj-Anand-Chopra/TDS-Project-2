"""
AUTONOMOUS QUIZ SOLVER - Production Ready
Features:
- File System Management (LLMFiles)
- Safe Code Execution with Context Trimming
- Robust POST Requests with Retry Logic
- Safety Settings for Gemini API
- Background Task Processing
"""

import os
import sys
import time
import json
import logging
import subprocess
import requests
import io
import hashlib
import base64
from typing import TypedDict, Annotated, List, Any, Dict, Optional
from collections import defaultdict
from contextlib import redirect_stdout
from urllib.parse import urljoin

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import uvicorn

from langchain_core.tools import tool
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_google_genai import HarmBlockThreshold, HarmCategory

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

# ============================================================================
# CONFIGURATION
# ============================================================================

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EMAIL = os.getenv("EMAIL", "")
SECRET = os.getenv("SECRET", "")
EXPECTED_SECRET = os.getenv("SECRET", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

RECURSION_LIMIT = 5000
MAX_TOKENS = 60000
RETRY_LIMIT = 4

BASE64_STORE = {}
URL_TIME = {}
CACHE = defaultdict(int)

# ============================================================================
# TOOLS
# ============================================================================


@tool
def get_rendered_html(url: str) -> str:
    """
    Fetch and return the HTML content from a URL.
    Returns a JSON string.
    """
    print(f"\nFetching and rendering: {url}")
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        content = response.text

        if len(content) > 300000:
            print("Warning: HTML too large, truncating...")
            content = content[:300000] + "... [TRUNCATED DUE TO SIZE]"

        imgs = []
        if BeautifulSoup:
            try:
                soup = BeautifulSoup(content, "html.parser")
                imgs = [urljoin(url, img["src"]) for img in soup.find_all("img", src=True)]
            except Exception:
                pass

        # FIX: Return JSON string
        return json.dumps({"html": content, "images": imgs, "url": url})
    except Exception as e:
        return json.dumps({"error": f"Error fetching page: {str(e)}"})


@tool
def download_file(url: str, filename: str) -> str:
    """
    Download a file from a URL and save it to the 'LLMFiles/' directory.
    """
    try:
        print(f"\nDownloading file from: {url}")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        os.makedirs("LLMFiles", exist_ok=True)
        path = os.path.join("LLMFiles", filename)

        with open(path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print(f"Saved to {path}")
        return f"File saved successfully at: {path}. Use this path for processing."
    except Exception as e:
        return f"Error downloading file: {str(e)}"


@tool
def run_code(code: str) -> str:
    """
    Execute Python code.
    Pre-imported: pandas, numpy, requests, hashlib, json, os, bs4, base64, re, math.
    Returns a JSON string.
    """
    try:
        print(f"\nExecuting Code:\n{code[:200]}...")

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
            "BeautifulSoup": BeautifulSoup,
            "os": os,
            "hashlib": hashlib,
            "base64": base64,
            "math": __import__("math"),
            "re": __import__("re"),
        }

        with redirect_stdout(f_out):
            try:
                original_cwd = os.getcwd()
                os.chdir("LLMFiles")
                exec(code, global_vars, local_vars)
            except Exception as e:
                f_err.write(str(e))
            finally:
                os.chdir(original_cwd)

        stdout = f_out.getvalue().strip()
        stderr = f_err.getvalue().strip()

        # FIX: Return JSON string
        result = {
            "stdout": stdout if len(stdout) < 10000 else stdout[:10000] + "...truncated",
            "stderr": stderr,
            "return_code": 0 if not stderr else 1,
        }
        return json.dumps(result)

    except Exception as e:
        return json.dumps({"stdout": "", "stderr": str(e), "return_code": -1})


@tool
def post_request(url: str, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> str:
    """
    Send an HTTP POST request to submit an answer.
    Returns response as a JSON string.
    """
    headers = headers or {"Content-Type": "application/json"}

    # Handle Base64 placeholder replacement
    ans = payload.get("answer")
    if isinstance(ans, str) and ans.startswith("BASE64_KEY:"):
        key = ans.split(":", 1)[1]
        if key in BASE64_STORE:
            payload["answer"] = BASE64_STORE[key]
            print("Swapped placeholder with actual Base64 data.")

    try:
        log_payload = payload.copy()
        if isinstance(log_payload.get("answer"), str) and len(log_payload["answer"]) > 100:
            log_payload["answer"] = log_payload["answer"][:50] + "..."
        print(f"\nSending Answer to {url}...\nPayload: {json.dumps(log_payload, indent=2)}")

        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()

        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")

        # FIX: Return JSON string
        return json.dumps(data)

    except Exception as e:
        logger.error(f"POST Request Error: {e}")
        return json.dumps({"error": str(e)})


@tool
def add_dependencies(package_name: str) -> str:
    """
    Install Python packages dynamically using pip.
    """
    try:
        print(f"\nInstalling {package_name}...")
        if package_name in ["hashlib", "math", "json", "os", "sys"]:
            return "Already installed."

        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name, "-q"])
        return f"Successfully installed {package_name}"
    except Exception as e:
        return f"Installation failed: {e}"


# ============================================================================
# AGENT SETUP
# ============================================================================

ALL_TOOLS = [get_rendered_html, download_file, run_code, post_request, add_dependencies]

safety_settings = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

rate_limiter = InMemoryRateLimiter(
    requests_per_second=15 / 60,
    check_every_n_seconds=1,
    max_bucket_size=15,
)

llm = init_chat_model(
    model_provider="google_genai",
    model="gemini-2.5-flash",
    rate_limiter=rate_limiter,
    safety_settings=safety_settings,
).bind_tools(ALL_TOOLS)

SYSTEM_PROMPT = f"""You are an elite autonomous agent designed to solve web challenges.

YOUR GOAL:
1. Access the given URL.
2. Analyze the page content (HTML, Text, Images).
3. Solve the specific puzzle (Math, Scraping, Coding).
4. Submit the answer to the provided endpoint.
5. Repeat until you receive completion message.

CRITICAL RULES:
- **Files**: All files are downloaded to 'LLMFiles/'. Always prepend 'LLMFiles/' when reading files in Python.
- **Libraries**: pandas, numpy, hashlib, requests, bs4 are pre-installed.
- **Submission**: Always use `post_request`.
- **Identity**: 
    - Email: {EMAIL}
    - Secret: {SECRET}

Start by fetching the page content. Good luck."""


class AgentState(TypedDict):
    messages: Annotated[List, add_messages]


def agent_node(state: AgentState):
    """Main agent node."""
    print(f"--- INVOKING AGENT (Context: {len(state['messages'])} msgs) ---")
    result = llm.invoke(state["messages"])
    return {"messages": [result]}


def route(state):
    """Routing logic."""
    last = state["messages"][-1]

    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"

    content = last.content if hasattr(last, "content") else ""
    if isinstance(content, str) and ("Tasks completed" in content or "END" in content):
        return END

    return "agent"


graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(ALL_TOOLS))

graph.add_edge(START, "agent")
graph.add_edge("tools", "agent")
graph.add_conditional_edges("agent", route, {"tools": "tools", "agent": "agent", END: END})

agent_runner = graph.compile()

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

SERVER_START_TIME = time.time()


@app.get("/healthz")
def healthz():
    return {"status": "ok", "uptime": int(time.time() - SERVER_START_TIME), "model": "gemini-2.5-flash"}


@app.get("/health")
def health():
    return {"status": "ok", "model": "gemini-2.5-flash"}


async def background_solve(url: str):
    """Background task to run the agent loop."""
    try:
        print(f"Starting background task for: {url}")
        BASE64_STORE.clear()
        URL_TIME.clear()
        CACHE.clear()

        initial_msg = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Start the challenge at: {url}"},
        ]

        agent_runner.invoke({"messages": initial_msg}, config={"recursion_limit": RECURSION_LIMIT})
        print("Task Finished Successfully.")
    except Exception as e:
        print(f"Task Failed: {e}")


@app.post("/solve-quiz")
async def solve_quiz_endpoint(request: Request):
    """Synchronous endpoint."""
    try:
        data = await request.json()
        url = data.get("url")
        secret = data.get("secret")

        if not url or not secret:
            raise HTTPException(status_code=400, detail="Missing url or secret")

        if secret != EXPECTED_SECRET:
            raise HTTPException(status_code=403, detail="Invalid secret")

        result = {}
        try:
            BASE64_STORE.clear()
            URL_TIME.clear()
            CACHE.clear()

            initial_msg = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Start the challenge at: {url}"},
            ]

            agent_runner.invoke({"messages": initial_msg}, config={"recursion_limit": RECURSION_LIMIT})
            result = {"status": "completed", "message": "All quiz tasks solved successfully"}
        except Exception as e:
            result = {"status": "error", "message": str(e)}

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/solve")
async def solve_endpoint(request: Request, background_tasks: BackgroundTasks):
    """Asynchronous endpoint."""
    try:
        data = await request.json()
        url = data.get("url")
        secret = data.get("secret")

        if not url or not secret:
            raise HTTPException(status_code=400, detail="Missing url or secret")

        if secret != EXPECTED_SECRET:
            raise HTTPException(status_code=403, detail="Invalid secret")

        background_tasks.add_task(background_solve, url)

        return JSONResponse(status_code=200, content={"status": "ok", "message": "Agent started"})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.getenv("PORT", 1010))
    uvicorn.run(app, host="0.0.0.0", port=port)
