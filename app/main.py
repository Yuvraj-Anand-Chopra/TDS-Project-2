import os
import logging
import json
import re
import base64
import io
from contextlib import redirect_stdout
from typing import Dict
from urllib.parse import urlparse

import requests
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv
from bs4 import BeautifulSoup

load_dotenv()
logging.basicConfig(level=logging.INFO)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class QuizRequest(BaseModel):
    email: str
    secret: str
    url: HttpUrl

class TaskSolver:
    def __init__(self):
        self.model = genai.GenerativeModel("gemini-2.0-flash")
        self.headers = {"User-Agent": "Mozilla/5.0"}

    def decode_html(self, html):
        try:
            matches = re.findall(r'atob\([`"\']([A-Za-z0-9+/=\s]+)[`"\']\)', html)
            parts = []
            for m in matches:
                try: parts.append(base64.b64decode(re.sub(r'\s+', '', m)).decode('utf-8'))
                except: pass
            if parts: return "\n".join(parts) + "\n" + html
            return html
        except: return html

    def clean_text(self, html):
        try:
            soup = BeautifulSoup(html, 'html.parser')
            for s in soup(['script', 'style']): s.decompose()
            text = soup.get_text(separator="\n")
            lines = [l.strip() for l in text.split('\n') if l.strip()]
            filtered = []
            for l in lines:
                q = l.lower()
                if "post to json" in q: continue
                if "cutoff:" in q: continue
                if "email=" in q or "anything you want" in q: continue
                if len(l) < 5 and not l.isdigit(): continue
                filtered.append(l)
            return "\n".join(filtered) or "Question"
        except: return html

    def run_code(self, code):
        try:
            f = io.StringIO()
            import pandas as pd, numpy as np
            g = {"pd": pd, "numpy": np, "np": np, "requests": requests, "json": __import__('json'), "print": print}
            with redirect_stdout(f): exec(code, g, {})
            return f.getvalue().strip() or "Success"
        except Exception as e: return str(e)

    def get_answer(self, q):
        prompt = f"Answer: {q}\nGive ONLY answer or output Python code without any markdown or formatting."
        try:
            res = self.model.generate_content(prompt).text.strip()
            if "python" in res.lower():
                start = res.lower().find("python")
                code = res[start+6:].strip()
                if code.startswith(":"): code = code[1:].lstrip()
                return self.run_code(code)
            return res
        except: return "Error"

    def solve(self, url, email, secret):
        try:
            r = requests.get(url, headers=self.headers, timeout=30)
            html = self.decode_html(r.text)
            if "/demo" in url and "scrape" not in url and "audio" not in url:
                ans = "anything you want"
            else:
                ans = self.get_answer(self.clean_text(html))
            final = ans
            try:
                sval = str(ans)
                if sval.replace('.','',1).isdigit(): final = float(sval) if '.' in sval else int(sval)
            except: pass
            u = urlparse(url)
            sub = f"{u.scheme}://{u.netloc}/submit"
            pl = {"email": email, "secret": secret, "url": url, "answer": final}
            return requests.post(sub, json=pl, timeout=30).json()
        except Exception as e:
            return {"error": str(e)}

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
solver = TaskSolver()

@app.post("/solve-quiz")
async def endpoint(req: Request):
    try:
        d = await req.json()
        data = QuizRequest(**d)
    except: raise HTTPException(400)
    if os.getenv("SECRET") and data.secret != os.getenv("SECRET"):
        raise HTTPException(403)
    url = str(data.url)
    res = {}
    for _ in range(10):
        res = solver.solve(url, data.email, data.secret)
        if res.get("correct") and res.get("url"):
            url = res["url"]
            continue
        break
    return res

@app.get("/health")
def health(): return {"status": "ok"}
