import logging
import asyncio
import time
from typing import Callable, Any

logger = logging.getLogger(__name__)

class RequestHandler:
    @staticmethod
    async def retry_with_backoff(
        operation: Callable,
        max_retries: int = 3,
        initial_delay: float = 0.5,
        backoff_factor: float = 2.0
    ) -> Any:
        """Retry operation with exponential backoff"""
        last_exception = None
        delay = initial_delay
        
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Attempt {attempt}/{max_retries}")
                
                # Check if operation is async or sync
                result = operation()
                if asyncio.iscoroutine(result):
                    result = await result
                else:
                    # If sync, run in thread executor to avoid blocking
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, lambda: result)
                
                logger.info(f"Attempt {attempt} succeeded")
                return result
                
            except Exception as e:
                last_exception = e
                logger.warning(f"Attempt {attempt} failed. Error: {str(e)}")
                
                if attempt < max_retries:
                    wait_time = delay * (backoff_factor ** (attempt - 1))
                    logger.info(f"Retrying in {wait_time:.2f}s...")
                    await asyncio.sleep(wait_time)
        
        logger.error(f"All {max_retries} attempts failed")
        raise last_exception
    
    @staticmethod
    async def submit_json(url: str, payload: dict, timeout: int = 30) -> dict:
        """Submit JSON payload to URL"""
        import requests
        logger.info(f"Submitting to {url}")
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()
    
    @staticmethod
    async def fetch_json(url: str, timeout: int = 30) -> dict:
        """Fetch JSON from URL"""
        import requests
        logger.info(f"Fetching from {url}")
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.json()
