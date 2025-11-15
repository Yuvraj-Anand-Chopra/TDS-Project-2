import logging
import asyncio
import json
import re
from typing import Optional, Dict, Any

from app.llm_helper import GeminiHelper
from app.scrapers import scraper
from app.request_handler import RequestHandler

logger = logging.getLogger(__name__)

class QuizProcessor:
    def __init__(self):
        self.gemini = GeminiHelper()
        self.scraper = scraper
    
    async def solve_quiz(self, url: str, task_id: str) -> Dict:
        """Main quiz solving logic"""
        logger.info(f"Task {task_id}: Starting quiz processing for {url}")
        
        try:
            await self.scraper.initialize()
            
            # Fetch quiz page
            quiz_data = await self.scraper.fetch_quiz_page(url)
            
            logger.info(f"Task {task_id}: Question parsed: {quiz_data['instructions'][:100]}")
            
            # Extract submit URL from instructions
            # Look for "/submit" pattern
            instructions = quiz_data['instructions']
            submit_url = None
            
            # Check if instructions contain absolute URL
            if 'https://' in instructions or 'http://' in instructions:
                match = re.search(r'(https?://[^\s]+/submit)', instructions)
                if match:
                    submit_url = match.group(1)
            
            # If no absolute URL, look for relative path
            if not submit_url:
                if '/submit' in instructions:
                    # Extract base URL from the current URL
                    from urllib.parse import urlparse
                    parsed = urlparse(url)
                    base_url = f"{parsed.scheme}://{parsed.netloc}"
                    submit_url = f"{base_url}/submit"
            
            # Fallback to URL + /submit
            if not submit_url:
                submit_url = f"{url}/submit"
            
            logger.info(f"Task {task_id}: Using submit URL: {submit_url}")
            
            # Use Gemini to understand the question
            gemini_prompt = f"""
You are a quiz solver. Analyze this quiz question and provide the answer.

Question:
{quiz_data['instructions']}

Provide ONLY the answer value, nothing else.
"""
            
            answer = await self.gemini.generate_response(gemini_prompt)
            logger.info(f"Task {task_id}: Generated answer: {str(answer)[:100]}")
            
            # Submit answer
            logger.info(f"Task {task_id}: Submitting answer to {submit_url}")
            
            payload = {
                "email": "24f2002642@ds.study.iitm.ac.in",
                "secret": "YuvrajChopra2024Secret",
                "url": url,
                "answer": answer
            }
            
            response = await RequestHandler.submit_json(submit_url, payload)
            logger.info(f"Task {task_id}: Server response: {response}")
            
            await self.scraper.close()
            
            return {
                "correct": response.get("correct", False),
                "answer": answer,
                "url": response.get("url"),
                "reason": response.get("reason")
            }
        
        except Exception as e:
            logger.error(f"Task {task_id}: Error in quiz processing: {str(e)}", exc_info=True)
            await self.scraper.close()
            raise
