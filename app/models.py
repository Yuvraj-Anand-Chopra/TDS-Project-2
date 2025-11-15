from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Any

class QuizRequest(BaseModel):
    """Quiz task request model"""
    email: EmailStr
    secret: str = Field(..., max_length=100)
    url: str = Field(..., description="URL to the quiz page")

    class Config:
        json_schema_extra = {
            "example": {
                "email": "student@vit.ac.in",
                "secret": "my-secret",
                "url": "https://quiz.example.com/q123"
            }
        }

class QuizResponse(BaseModel):
    """Quiz task response model"""
    correct: bool
    url: Optional[str] = None
    reason: Optional[str] = None

class HealthCheckResponse(BaseModel):
    """Health check response model"""
    status: str
    timestamp: str
    version: str = "1.0.0"

class ErrorResponse(BaseModel):
    """Error response model"""
    detail: str
    path: Optional[str] = None
    status_code: int
