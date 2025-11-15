import sys
import asyncio

# Fix for Windows asyncio subprocess issue with Python 3.13
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import logging
from datetime import datetime
import uuid

from app.config import settings
from app.models import QuizRequest, QuizResponse, HealthCheckResponse
from app.processor import QuizProcessor
from app.security import verify_secret

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application starting up")
    logger.info(f"Environment: {settings.ENV}")
    logger.info(f"Gemini Model: {settings.GEMINI_MODEL}")
    yield
    logger.info("Application shutting down")

app = FastAPI(
    title="LLM Analysis Quiz Solver",
    description="Automated quiz solving using LLM and data analysis",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "LLM Analysis Quiz Solver API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )

@app.post("/solve-quiz", response_model=QuizResponse)
async def solve_quiz(request: QuizRequest):
    task_id = str(uuid.uuid4())[:8]
    logger.info(f"Starting quiz task {task_id} | email={request.email} | url={request.url}")
    
    try:
        if not verify_secret(request.secret):
            logger.warning(f"Task {task_id}: Invalid secret provided")
            raise HTTPException(status_code=403, detail="Invalid secret")
        
        processor = QuizProcessor()
        result = await processor.solve_quiz(request.url, task_id)
        
        logger.info(f"Task {task_id}: Quiz completed | correct={result.get('correct')}")
        return QuizResponse(**result)
        
    except HTTPException as he:
        logger.error(f"HTTP Exception: {he.status_code} - {he.detail}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing quiz: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail="An unexpected error occurred while processing the quiz"
        )

@app.post("/test-connection")
async def test_connection():
    try:
        from app.llm_helper import GeminiHelper
        helper = GeminiHelper()
        response = await helper.generate_response("Say 'Hello! Connection successful!'")
        return {"status": "success", "message": response}
    except Exception as e:
        logger.error(f"Connection test failed: {str(e)}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
