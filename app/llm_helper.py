import logging
import os
import google.generativeai as genai
from app.config import settings

logger = logging.getLogger(__name__)

class GeminiHelper:
    def __init__(self):
        """Initialize Gemini API"""
        try:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self.model = genai.GenerativeModel(settings.GEMINI_MODEL)
            logger.info(f"Gemini initialized with model: {settings.GEMINI_MODEL}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {str(e)}")
            raise
    
    async def generate_response(self, prompt: str, max_retries: int = 3) -> str:
        """Generate response from Gemini"""
        for attempt in range(max_retries):
            try:
                logger.info(f"Generating response from Gemini (attempt {attempt + 1}/{max_retries})")
                
                response = self.model.generate_content(
                    prompt,
                    generation_config={
                        "max_output_tokens": 1024,
                        "temperature": 0.7,
                    }
                )
                
                if response.text:
                    logger.info(f"Gemini response generated: {len(response.text)} chars")
                    return response.text.strip()
                else:
                    logger.warning(f"Empty response from Gemini")
                    return "No response"
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error(f"All {max_retries} attempts failed")
                    raise
                import asyncio
                await asyncio.sleep(0.5 * (attempt + 1))
    
    async def analyze_question(self, question: str) -> dict:
        """Analyze a quiz question"""
        prompt = f"""
Analyze this quiz question and extract:
1. The actual question
2. The type of answer needed (number, text, boolean, etc.)
3. Any hints or clues in the question

Question:
{question}

Respond in JSON format.
"""
        response = await self.generate_response(prompt)
        try:
            import json
            return json.loads(response)
        except:
            return {"question": question, "response": response}
    
    async def solve_problem(self, problem: str, context: str = "") -> str:
        """Solve a problem or answer a question"""
        prompt = f"""
Solve this problem and provide ONLY the answer:

Problem:
{problem}

{f"Context: {context}" if context else ""}

Provide ONLY the numerical or text answer, nothing else.
"""
        return await self.generate_response(prompt)
