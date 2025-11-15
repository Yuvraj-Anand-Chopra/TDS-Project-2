import logging
import json
from typing import Any
from datetime import datetime

logger = logging.getLogger(__name__)

class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for special types"""

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)

def safe_json_dumps(obj: Any) -> str:
    """Safely convert object to JSON string"""
    try:
        return json.dumps(obj, cls=JSONEncoder)
    except Exception as e:
        logger.error(f"Error serializing to JSON: {str(e)}")
        return json.dumps({"error": str(obj)})

def format_log_message(message: str, **kwargs) -> str:
    """Format log message with context"""
    context = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
    if context:
        return f"{message} | {context}"
    return message

def validate_email(email: str) -> bool:
    """Simple email validation"""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def truncate_string(text: str, max_length: int = 100) -> str:
    """Truncate string to max length"""
    if len(text) > max_length:
        return text[:max_length-3] + "..."
    return text
