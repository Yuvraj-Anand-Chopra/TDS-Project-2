"""
CONSOLIDATED MAIN.PY - Single File Execution
All logic from agent.py, tools.py, and main.py merged into one file.
Execution: python app.py or uv run app.py
"""

# ============================================================================
# IMPORTS - All dependencies at the top
# ============================================================================

import os
import sys
import time
import json
import subprocess
import logging
from typing import TypedDict, Annotated, List, Any, Dict, Optional
from contextlib import redirect_stdout
import io

# FastAPI
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# LangGraph
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode

# LangChain
from langchain_core.tools import tool
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import init_chat_model
from langgraph.graph.message import add_messages

# Google GenAI Types for Safety Settings
from langchain_google_genai import HarmBlockThreshold, HarmCategory

# Other
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import uvicorn

# ============================================================================
# CONFIGURATION & ENVIRONMENT
# ============================================================================

load_dotenv()

EMAIL = os.getenv("EMAIL")
SECRET = os.getenv("SECRET")
EXPECTED_SECRET = os.getenv("SECRET")
# Ensure we check for both keys to be safe
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# TOOL FUNCTIONS - Helper functions that LLM can call
# ============================================================================

@tool
def get_rendered_html(url: str) -> str:
    """
    Fetch and return the HTML content from a URL.
    Use this to get the page content before making decisions.
    """
    try:
        print(f"\nFetching and rendering: {url}\n")
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        logger.info(f"Retrieved HTML from {url}, length: {len(response.text)}")
        return response.text
    except Exception as e:
        logger.error(f"Error fetching {url}: {e}")
        return f"Error fetching URL: {str(e)}"

@tool
def download_file(url: str, filename: str = None) -> str:
    """
    Download a file (or any resource) from a URL and return its content.
    Useful for getting utils.js, data files, JSON endpoints, etc.
    """
    try:
        print(f"\nFetching and rendering: {url}\n")
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Save to file if filename provided
        saved_path = "memory"
        if filename:
            os.makedirs("LLMFiles", exist_ok=True)
            filepath = os.path.join("LLMFiles", filename)
            with open(filepath, "wb") as f:
                f.write(response.content)
            logger.info(f"Downloaded and saved: {filepath}")
            saved_path = filepath
        
        logger.info(f"Downloaded from {url}, length: {len(response.text)}")
        
        # FIX: Explicitly tell the LLM the PATH where it was saved
        return (f"File downloaded successfully. "
                f"Saved at path: '{saved_path}'. "
                f"Content length: {len(response.text)}. "
                f"IMPORTANT: When reading this file in Python, you MUST use the path '{saved_path}'. "
                "Do not download this again. Immediately use 'run_code' to read and analyze it.")
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        return f"Error downloading file: {str(e)}"

@tool
def post_request(url: str, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> str:
    """
    Send an HTTP POST request to submit an answer.
    Always use this to submit answers to the /submit endpoint.
    Returns the response as a JSON string.
    """
    headers = headers or {"Content-Type": "application/json"}
    try:
        print(f"\nSending Answer\n{json.dumps(payload, indent=4)}\n to url: {url}\n")
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        delay = data.get("delay", 0)
        delay = delay if isinstance(delay, (int, float)) else 0
        correct = data.get("correct", False)
        
        if not correct and delay and delay < 180:
            if "url" in data:
                del data["url"]
        
        if delay and delay >= 180:
            data = {"url": data.get("url")}
        
        print(f"Got the response:\n{json.dumps(data, indent=4)}\n")
        return json.dumps(data)
        
    except requests.HTTPError as e:
        err_resp = e.response
        try:
            err_data = err_resp.json()
        except ValueError:
            err_data = err_resp.text
        logger.error(f"HTTP Error Response: {err_data}")
        return json.dumps({"error": str(err_data)})
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return json.dumps({"error": str(e)})

@tool
def run_code(code: str) -> str:
    """
    Execute Python code and return the output.
    Use this when you need to calculate values, parse data, or process information.
    """
    try:
        import pandas as pd
        import numpy as np
        
        print(f"\nExecuting code:\n{code}\n")
        
        os.makedirs("LLMFiles", exist_ok=True)
        filename = "runner.py"
        filepath = os.path.join("LLMFiles", filename)
        
        with open(filepath, "w") as f:
            f.write(code)
        
        f_out = io.StringIO()
        local_vars = {}
        # FIX: Added 'os' to global vars so it can check paths
        global_vars = {
            "pd": pd,
            "numpy": np,
            "np": np,
            "requests": requests,
            "json": json,
            "print": print,
            "BeautifulSoup": BeautifulSoup,
            "os": os
        }
        
        with redirect_stdout(f_out):
            exec(code, global_vars, local_vars)
        
        output = f_out.getvalue().strip()
        result = output if output else "Code executed successfully"
        logger.info(f"Code result: {result}")
        return result
        
    except Exception as e:
        error_msg = f"Code execution failed: {str(e)}"
        logger.error(error_msg)
        return error_msg

@tool
def add_dependencies(package_name: str) -> str:
    """
    Install a Python package if needed.
    Use this if the code requires additional libraries.
    """
    try:
        print(f"\nInstalling {package_name}...\n")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name, "-q"])
        result = f"Successfully installed {package_name}"
        logger.info(result)
        return result
    except Exception as e:
        error_msg = f"Error installing {package_name}: {str(e)}"
        logger.error(error_msg)
        return error_msg

# ============================================================================
# LANGGRAPH SETUP - Agent state and configuration
# ============================================================================

class AgentState(TypedDict):
    """State object that holds conversation messages"""
    messages: Annotated[List, add_messages]

# Define tools that the agent can use
TOOLS = [run_code, get_rendered_html, download_file, post_request, add_dependencies]

# Rate limiter for Gemini API
rate_limiter = InMemoryRateLimiter(
    requests_per_second=9/60,
    check_every_n_seconds=1,
    max_bucket_size=9
)

# Safety settings to prevent "FinishReason: 10" errors on CSV/Text data
safety_settings = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

# Initialize LLM with tool binding
llm = init_chat_model(
    model_provider="google_genai",
    model="gemini-2.5-flash",
    rate_limiter=rate_limiter,
    safety_settings=safety_settings
).bind_tools(TOOLS)

# System prompt - Instructions for the LLM
SYSTEM_PROMPT = f"""You are an autonomous quiz-solving agent.

Your job is to:
1. Load the quiz page from the given URL.
2. Extract ALL instructions, required parameters, submission rules, and the submit endpoint.
3. Solve the task exactly as required.
4. Submit the answer ONLY to the endpoint specified on the current page.

STRICT RULES — FOLLOW EXACTLY:

FILE SYSTEM RULES (CRITICAL):
- All files downloaded via 'download_file' are saved in the 'LLMFiles' subdirectory.
- When you write Python code to read these files, you MUST use the path 'LLMFiles/filename'.
- Example: `pd.read_csv('LLMFiles/data.csv')` or `open('LLMFiles/script.js')`.
- If a file read fails, use `os.listdir('LLMFiles')` to check the actual filenames.

DATA HANDLING RULES:
- If you download a CSV/Data file, DO NOT download it again. Immediately read it using `run_code`.
- Do not assume column names (e.g. 'value'). Always print `df.columns` first to verify.
- If you encounter a "KeyError", it means the column name is different. Check the printed columns and retry with the correct name.

GENERAL RULES:
- NEVER stop early. Continue solving tasks until no new URL is provided.
- NEVER hallucinate URLs, endpoints, fields, values, or JSON structure.
- ALWAYS use the tools provided to fetch, scrape, download, render HTML, or send requests.

TIME LIMIT RULES:
- Each task has a hard 3-minute limit.
- If your answer is wrong retry again.

STOPPING CONDITION:
- Only return "END" when a server response explicitly contains NO new URL.

ADDITIONAL INFORMATION YOU MUST INCLUDE WHEN REQUIRED:
- Email: {EMAIL}
- Secret: {SECRET}

YOUR JOB:
- Follow pages exactly.
- Extract data reliably.
- Never guess.
- Submit correct answers.
- Continue until no new URL.
- Then respond with: END"""

# Create prompt with system message
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="messages")
])

llm_with_prompt = prompt | llm

# ============================================================================
# AGENT NODE & ROUTING - Core agent logic
# ============================================================================

def agent_node(state: AgentState):
    """The main agent node - calls LLM to decide next action"""
    result = llm_with_prompt.invoke({"messages": state["messages"]})
    return {"messages": state["messages"] + [result]}

def route(state):
    """Decide whether to call tools, continue agent, or end"""
    last = state["messages"][-1]
    
    # Check for tool calls
    tool_calls = None
    if hasattr(last, "tool_calls"):
        tool_calls = getattr(last, "tool_calls", None)
    elif isinstance(last, dict):
        tool_calls = last.get("tool_calls")
    
    if tool_calls:
        return "tools"
    
    # Check for END signal
    content = None
    if hasattr(last, "content"):
        content = getattr(last, "content", None)
    elif isinstance(last, dict):
        content = last.get("content")
    
    if isinstance(content, str) and content.strip() == "END":
        return END
    
    if isinstance(content, list) and len(content) > 0:
        if isinstance(content[0], dict) and content[0].get("text", "").strip() == "END":
            return END
    
    return "agent"

# ============================================================================
# BUILD LANGGRAPH
# ============================================================================

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(TOOLS))

graph.add_edge(START, "agent")
graph.add_edge("tools", "agent")
graph.add_conditional_edges("agent", route)

agent_graph = graph.compile()

# ============================================================================
# AGENT EXECUTION FUNCTION
# ============================================================================

def run_agent(url: str) -> dict:
    """Execute the quiz-solving agent"""
    try:
        print("Verified starting the task...")
        result = agent_graph.invoke(
            {"messages": [{"role": "user", "content": str(url)}]},
            config={"recursion_limit": 5000},
        )
        print("Tasks completed successfully")
        return {"status": "completed", "message": "All quiz tasks solved successfully"}
    except Exception as e:
        print(f"Error: {e}")
        return {"status": "error", "message": str(e)}

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

# ============================================================================
# FASTAPI ENDPOINTS
# ============================================================================

@app.get("/health")
def health():
    """Health check endpoint"""
    return {"status": "ok", "model": "gemini-2.5-flash"}

@app.get("/healthz")
def healthz():
    """Simple liveness check"""
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME)
    }

@app.post("/solve-quiz")
async def solve_quiz_endpoint(request: Request):
    """Synchronous endpoint - blocks until quiz is solved"""
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    
    if not data:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    
    url = data.get("url")
    secret = data.get("secret")
    email = data.get("email")
    
    if not url or not secret:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    
    if secret != EXPECTED_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")
    
    result = run_agent(str(url))
    return result

@app.post("/solve")
async def solve_endpoint(request: Request, background_tasks: BackgroundTasks):
    """Asynchronous endpoint - runs in background"""
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    
    if not data:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    
    url = data.get("url")
    secret = data.get("secret")
    email = data.get("email")
    
    if not url or not secret:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    
    if secret != EXPECTED_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")
    
    print("Verified starting the task...")
    background_tasks.add_task(run_agent, url)
    
    return JSONResponse(status_code=200, content={"status": "ok"})

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 1010)))
