import logging
import requests
import json
import asyncio
from typing import Any, List, Dict, Optional
import re
import io

logger = logging.getLogger(__name__)

class DataAnalyzer:
    """Handles data extraction, transformation, and analysis"""
    
    @staticmethod
    async def download_file(url: str, timeout: int = 30) -> bytes:
        """Download file from URL with timeout"""
        try:
            logger.info(f"Downloading file from {url}")
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            logger.info(f"File downloaded successfully, size: {len(response.content)} bytes")
            return response.content
        except requests.RequestException as e:
            logger.error(f"Failed to download {url}: {str(e)}")
            raise
    
    @staticmethod
    def extract_pdf_table(pdf_content: bytes, page: int = 0, table_index: int = 0) -> Dict[str, Any]:
        """Extract table from PDF - simplified version without cryptography"""
        try:
            # Use pypdf which doesn't require cryptography
            import pypdf
            
            logger.info(f"Extracting data from PDF page {page}")
            
            pdf_file = io.BytesIO(pdf_content)
            reader = pypdf.PdfReader(pdf_file)
            
            if page >= len(reader.pages):
                logger.warning(f"Page {page} does not exist. PDF has {len(reader.pages)} pages")
                return {}
            
            pdf_page = reader.pages[page]
            text = pdf_page.extract_text()
            
            # Parse text for table data
            lines = text.split('\n')
            data = []
            for line in lines:
                if line.strip():
                    data.append(line.strip())
            
            logger.info(f"Extracted {len(data)} lines from PDF")
            return {"data": data, "text": text}
            
        except Exception as e:
            logger.error(f"Error extracting PDF: {str(e)}")
            # Fallback: return raw content info
            return {"size": len(pdf_content), "error": str(e)}
    
    @staticmethod
    def fetch_api_data(url: str, headers: Optional[Dict] = None, timeout: int = 30) -> Any:
        """Fetch data from API endpoint"""
        try:
            logger.info(f"Fetching API data from {url}")
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            logger.info(f"API data fetched successfully")
            return data
        except Exception as e:
            logger.error(f"Error fetching API data from {url}: {str(e)}")
            raise
    
    @staticmethod
    def aggregate_data(data: Dict, group_by: str, agg_column: str, 
                      agg_func: str = "sum") -> Dict[str, float]:
        """Aggregate data by column"""
        try:
            # Simple aggregation without pandas
            result = {}
            if isinstance(data, list):
                for item in data:
                    key = item.get(group_by)
                    value = item.get(agg_column, 0)
                    if key:
                        if key not in result:
                            result[key] = []
                        try:
                            result[key].append(float(value))
                        except (ValueError, TypeError):
                            pass
                
                # Apply aggregation function
                if agg_func == "sum":
                    result = {k: sum(v) for k, v in result.items()}
                elif agg_func == "mean":
                    result = {k: sum(v)/len(v) for k, v in result.items()}
                elif agg_func == "count":
                    result = {k: len(v) for k, v in result.items()}
            
            logger.info(f"Aggregated data: {group_by} by {agg_func}")
            return result
        except Exception as e:
            logger.error(f"Error aggregating data: {str(e)}")
            raise
    
    @staticmethod
    def calculate_statistics(data: List[float]) -> Dict[str, float]:
        """Calculate summary statistics"""
        try:
            if not data:
                return {}
            
            data_sorted = sorted(data)
            n = len(data)
            
            stats = {
                "mean": sum(data) / n,
                "median": data_sorted[n//2] if n % 2 else (data_sorted[n//2-1] + data_sorted[n//2]) / 2,
                "min": min(data),
                "max": max(data),
                "sum": sum(data),
                "count": n
            }
            
            # Standard deviation
            mean = stats["mean"]
            variance = sum((x - mean) ** 2 for x in data) / n
            stats["std"] = variance ** 0.5
            
            logger.info(f"Calculated statistics for {n} values")
            return stats
        except Exception as e:
            logger.error(f"Error calculating statistics: {str(e)}")
            raise
    
    @staticmethod
    def filter_data(data: List[Dict], conditions: Dict) -> List[Dict]:
        """Filter data based on conditions"""
        try:
            result = data.copy() if isinstance(data, list) else [data]
            for column, value in conditions.items():
                result = [item for item in result if item.get(column) == value]
            logger.info(f"Filtered data from {len(data)} to {len(result)} items")
            return result
        except Exception as e:
            logger.error(f"Error filtering data: {str(e)}")
            raise
    
    @staticmethod
    def extract_text_data(html_content: str) -> str:
        """Extract plain text from HTML content"""
        try:
            text = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
            text = re.sub(r'<[^>]+>', '', text)
            from html import unescape
            text = unescape(text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            raise
