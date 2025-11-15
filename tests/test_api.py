import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.config import settings
from app.security import verify_secret

client = TestClient(app)

class TestAPI:
    """API endpoint tests"""

    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        assert "LLM Analysis Quiz Solver" in response.json()["message"]

    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_solve_quiz_invalid_secret(self):
        """Test quiz endpoint with invalid secret"""
        payload = {
            "email": "test@vit.ac.in",
            "secret": "invalid-secret-xyz",
            "url": "https://example.com/quiz/123"
        }
        response = client.post("/solve-quiz", json=payload)
        assert response.status_code == 403
        assert "Invalid secret" in response.json()["detail"]

    def test_solve_quiz_missing_email(self):
        """Test quiz endpoint with missing email"""
        payload = {
            "secret": "test-secret",
            "url": "https://example.com/quiz/123"
        }
        response = client.post("/solve-quiz", json=payload)
        assert response.status_code == 422

class TestSecurity:
    """Security-related tests"""

    def test_verify_secret_valid(self):
        """Test secret verification with valid secret"""
        assert verify_secret(settings.SECRET)

    def test_verify_secret_invalid(self):
        """Test secret verification with invalid secret"""
        assert not verify_secret("invalid-secret")
