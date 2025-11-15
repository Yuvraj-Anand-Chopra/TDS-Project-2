import logging
import requests
from bs4 import BeautifulSoup
from typing import Optional, Dict
import re
import base64

logger = logging.getLogger(__name__)

class WebScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.last_html = None
        self.last_url = None
    
    async def initialize(self):
        """Initialize scraper"""
        logger.info("Scraper initialized (requests-based)")
    
    async def fetch_quiz_page(self, url: str) -> Dict:
        """Fetch quiz page and extract all content"""
        try:
            logger.info(f"Fetching quiz page: {url}")
            self.last_url = url
            
            # Fetch page content
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            html_content = response.text
            
            # Parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Decode JavaScript base64 content
            scripts = soup.find_all('script')
            for script in scripts:
                if script.string and 'atob' in script.string:
                    # Extract base64 content
                    matches = re.findall(r'atob\([`"\']([A-Za-z0-9+/=\s]+)[`"\']\)', script.string)
                    for match in matches:
                        try:
                            clean_match = match.replace('\n', '').replace(' ', '').replace('\r', '')
                            decoded = base64.b64decode(clean_match).decode('utf-8')
                            logger.info(f"Decoded base64 content: {len(decoded)} bytes")
                            
                            # Insert decoded content
                            result_div = soup.find(id='result')
                            if result_div:
                                result_div.clear()
                                decoded_soup = BeautifulSoup(decoded, 'html.parser')
                                result_div.append(decoded_soup)
                        except Exception as e:
                            logger.warning(f"Failed to decode base64: {str(e)}")
            
            # Get final HTML
            self.last_html = str(soup)
            
            # Extract text from result div or body
            result_div = soup.find(id='result')
            if result_div:
                instructions = result_div.get_text('\n').strip()
            else:
                instructions = soup.get_text('\n').strip()
            
            # Extract submit URL
            submit_url = None
            submit_patterns = [
                r'POST.*?to\s+(https?://[^\s]+)',
                r'post.*?to\s+(https?://[^\s]+)',
                r'submit.*?to\s+(https?://[^\s]+)',
                r'(https?://[^\s]+/submit)',
            ]
            
            for pattern in submit_patterns:
                match = re.search(pattern, instructions, re.IGNORECASE)
                if match:
                    submit_url = match.group(1).strip()
                    break
            
            logger.info(f"Extracted quiz page - instructions length: {len(instructions)}")
            if submit_url:
                logger.info(f"Found submit URL: {submit_url}")
            
            return {
                'url': url,
                'html': self.last_html,
                'text': instructions,
                'instructions': instructions,
                'submit_url': submit_url
            }
            
        except Exception as e:
            logger.error(f"Error fetching quiz page {url}: {str(e)}")
            raise
    
    async def fetch_page_content(self, url: str, wait_for: Optional[str] = None, timeout: int = 30) -> str:
        """Fetch page content"""
        try:
            logger.info(f"Fetching page: {url}")
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()
            self.last_html = response.text
            self.last_url = url
            logger.info(f"Page fetched: {len(self.last_html)} bytes")
            return self.last_html
        except Exception as e:
            logger.error(f"Error fetching page {url}: {str(e)}")
            raise
    
    async def get_element_text(self, selector: str) -> str:
        """Get text content of element"""
        try:
            if not self.last_html:
                return ""
            soup = BeautifulSoup(self.last_html, 'html.parser')
            
            if selector.startswith('#'):
                element = soup.find(id=selector[1:])
            elif selector.startswith('.'):
                element = soup.find(class_=selector[1:])
            else:
                element = soup.find(selector)
            
            if element:
                return element.get_text().strip()
            return ""
        except Exception as e:
            logger.error(f"Error getting element text for {selector}: {str(e)}")
            return ""
    
    async def close(self):
        """Close scraper"""
        if self.session:
            self.session.close()
        logger.info("Scraper closed successfully")

# Global scraper instance
scraper = WebScraper()
