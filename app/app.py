"""
AUTONOMOUS QUIZ SOLVER - Production Ready (Stable Version)
- LangGraph 0.0.69 (stable)
- LangChain 0.1.10 (compatible)
- Safe tool output handling
- No serialization errors
"""

import os
import sys
import time
import json
import logging
import subprocess
import requests
import io
from typing import TypedDict, Annotated, List, Any, Dict, Optional
from contextlib import redirect_stdout

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
# SETUP
# ============================================================================

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EMAIL = os.getenv("EMAIL", "")
SECRET = os.getenv("SECRET", "")
EXPECTED_SECRET = os.getenv("SECRET", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

RECURSION_LIMIT = 5000
MAX_TOKENS = 50000

# ============================================================================
# TOOLS
# ============================================================================


@tool
def get_rendered_html(url: str) -> str:
    """Fetch and return HTML content from a URL."""
    print(f"\nFetching: {url}")
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        content = response.text

        if len(content) > 300000:
            content = content[:300000] + "... [TRUNCATED]"

        return content[:100000] if len(content) > 100000 else content
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def download_file(url: str, filename: str) -> str:
    """Download a file and save it to LLMFiles/."""
    try:
        print(f"\nDownloading: {url}")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        os.makedirs("LLMFiles", exist_ok=True)
        path = os.path.join("LLMFiles", filename)

        with open(path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        return f"Saved to {path}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def run_code(code: str) -> str:
    """Execute Python code safely."""
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
            "hashlib": __import__("hashlib"),
            "base64": __import__("base64"),
            "math": __import__("math"),
            "re": __import__("re"),
        }

        if BeautifulSoup:
            global_vars["BeautifulSoup"] = BeautifulSoup

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
        output = stdout if not stderr else f"ERROR: {stderr}"
        return output[:5000] if len(output) > 5000 else output

    except Exception as e:
        return f"Execution failed: {str(e)}"


@tool
def post_request(url: str, payload: Dict[str, Any]) -> str:
    """Send HTTP POST request and return response."""
    try:
        print(f"\nSending to: {url}")
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        return json.dumps(response.json())
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def add_dependencies(package_name: str) -> str:
    """Install Python packages."""
    try:
        print(f"\nInstalling: {package_name}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name, "-q"])
        return f"Installed {package_name}"
    except Exception as e:
        return f"Failed: {str(e)}"


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
    requests_per_second=9 / 60,
    check_every_n_seconds=1,
    max_bucket_size=9,
)

llm = init_chat_model(
    model_provider="google_genai",
    model="gemini-2.5-flash",
    rate_limiter=rate_limiter,
    safety_settings=safety_settings,
).bind_tools(ALL_TOOLS)

SYSTEM_PROMPT = f"""You are an autonomous agent solving web challenges.

GOAL:
1. Access the URL provided
2. Analyze content (HTML, text, data)
3. Solve the puzzle (math, scraping, coding)
4. Submit answer
5. Continue until completion

RULES:
- Files downloaded to LLMFiles/ (use this path in Python)
- Pre-installed: pandas, numpy, hashlib, requests, bs4
- Always use post_request to submit
- Email: {EMAIL}
- Secret: {SECRET}

Start by fetching the page."""


class AgentState(TypedDict):
    messages: Annotated[List, add_messages]


def agent_node(state: AgentState):
    """Main agent."""
    result = llm.invoke(state["messages"])
    return {"messages": [result]}


def route(state):
    """Route logic."""
    last = state["messages"][-1]

    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"

    content = getattr(last, "content", "")
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
# FASTAPI
# ============================================================================

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SERVER_START = time.time()


@app.get("/healthz")
def healthz():
    return {"status": "ok", "uptime": int(time.time() - SERVER_START)}


@app.get("/health")
def health():
    return {"status": "ok"}


async def run_agent_task(url: str):
    """Run agent in background."""
    try:
        print(f"Starting: {url}")
        initial_msg = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Start: {url}"},
        ]
        agent_runner.invoke(
            {"messages": initial_msg},
            config={"recursion_limit": RECURSION_LIMIT},
        )
        print("Completed successfully")
    except Exception as e:
        print(f"Failed: {e}")


@app.post("/solve-quiz")
async def solve_quiz(request: Request):
    """Solve quiz synchronously."""
    try:
        data = await request.json()
        url = data.get("url")
        secret = data.get("secret")

        if not url or secret != EXPECTED_SECRET:
            raise HTTPException(status_code=400, detail="Invalid request")

        try:
            initial_msg = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Start: {url}"},
            ]
            agent_runner.invoke(
                {"messages": initial_msg},
                config={"recursion_limit": RECURSION_LIMIT},
            )
            return {"status": "completed"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/solve")
async def solve_async(request: Request, background_tasks: BackgroundTasks):
    """Solve quiz asynchronously."""
    try:
        data = await request.json()
        url = data.get("url")
        secret = data.get("secret")

        if not url or secret != EXPECTED_SECRET:
            raise HTTPException(status_code=400, detail="Invalid request")

        background_tasks.add_task(run_agent_task, url)
        return {"status": "ok"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.getenv("PORT", 1010))
    uvicorn.run(app, host="0.0.0.0", port=port)
