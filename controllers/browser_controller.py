from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import WebDriverException, TimeoutException
import time
import re
from urllib.parse import quote_plus
import logging
from config.settings import settings
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class BrowserController:
    def __init__(self):
        self.driver = None
        self.setup_driver()

    def setup_driver(self):
        """Setup browser driver with fallback options"""
        try:
            # Try Chrome first
            chrome_options = ChromeOptions()
            if settings.BROWSER_HEADLESS:
                chrome_options.add_argument("--headless=new")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--use-gl=desktop")
            chrome_options.add_argument("--window-size=1920,1200")
            chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

            self.driver = webdriver.Chrome(options=chrome_options)
            logger.info("Chrome driver initialized successfully")

        except Exception as e:
            logger.warning("Chrome driver failed: %s, trying Firefox...", e)
            try:
                # Fallback to Firefox
                firefox_options = FirefoxOptions()
                if settings.BROWSER_HEADLESS:
                    firefox_options.add_argument("--headless")
                firefox_options.add_argument("--no-sandbox")
                firefox_options.add_argument("--disable-dev-shm-usage")

                self.driver = webdriver.Firefox(options=firefox_options)
                logger.info("Firefox driver initialized successfully")

            except Exception as e2:
                logger.error("Both Chrome and Firefox drivers failed: %s", e2)
                self.driver = None

    async def research_topic(self, topic: str, depth: str = "comprehensive") -> Dict:
        """Research topic using free sources"""
        if not self.driver:
            return {"error": "Browser driver not available"}

        results = {
            "topic": topic,
            "sources": [],
            "summary": "",
            "key_points": [],
            "timestamp": datetime.now().isoformat()
        }

        # Free research sources
        search_queries = [
            ("Wikipedia", f"https://en.wikipedia.org/wiki/Special:Search?search={quote_plus(topic)}&go=Go"),
            ("Google Scholar", f"https://scholar.google.com/scholar?q={quote_plus(topic)}"),
            ("DuckDuckGo", f"https://duckduckgo.com/?q={quote_plus(topic)}&ia=web")
        ]

        for source_name, url in search_queries:
            try:
                logger.info("Searching {source_name} for: %s", topic)
                content = await self._search_source(url, source_name)
                if content:
                    results["sources"].append({
                        "name": source_name,
                        "url": url,
                        "content": content,
                        "relevance_score": self._calculate_relevance(content, topic)
                    })

                # Add delay between requests
                time.sleep(2)

            except Exception as e:
                logger.error("Error searching {source_name}: %s", e)
                results["sources"].append({
                    "name": source_name,
                    "url": url,
                    "error": str(e)
                })

        # Generate summary from collected sources
        results["summary"] = self._generate_summary(results["sources"], topic)
        results["key_points"] = self._extract_key_points(results["sources"])

        return results

    async def _search_source(self, url: str, source_name: str) -> Optional[str]:
        """Search specific source and extract content"""
        try:
            self.driver.get(url)

            # Wait for page to load
            WebDriverWait(self.driver, 10).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )

            if "wikipedia" in url.lower():
                return self._extract_wikipedia_content()
            elif "scholar.google" in url.lower():
                return self._extract_scholar_content()
            elif "duckduckgo" in url.lower():
                return self._extract_duckduckgo_content()
            else:
                return self._extract_generic_content()

        except TimeoutException:
            logger.warning("Timeout loading %s", url)
            return None
        except Exception as e:
            logger.error("Error extracting content from {url}: %s", e)
            return None

    def _extract_wikipedia_content(self) -> str:
        """Extract content from Wikipedia"""
        try:
            # Try to find the main article content
            content_selectors = [
                "#mw-content-text p",
                ".mw-parser-output p",
                "#bodyContent p"
            ]

            for selector in content_selectors:
                try:
                    paragraphs = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if paragraphs:
                        content = " ".join([p.text.strip() for p in paragraphs[:5] if p.text.strip()])
                        if len(content) > 100:  # Ensure we got substantial content
                            return content[:2000]  # Limit content length
                except Exception:
                    continue

            # Fallback: get any text content
            body = self.driver.find_element(By.TAG_NAME, "body")
            return body.text[:1000] if body.text else "No content found"

        except Exception as e:
            logger.error("Error extracting Wikipedia content: %s", e)
            return "Wikipedia extraction failed"

    def _extract_scholar_content(self) -> str:
        """Extract content from Google Scholar"""
        try:
            # Look for search results
            results = self.driver.find_elements(By.CSS_SELECTOR, ".gs_ri")
            content_pieces = []

            for result in results[:3]:  # Top 3 results
                try:
                    title_elem = result.find_element(By.CSS_SELECTOR, ".gs_rt a")
                    title = title_elem.text.strip()

                    snippet_elem = result.find_element(By.CSS_SELECTOR, ".gs_rs")
                    snippet = snippet_elem.text.strip()

                    if title and snippet:
                        content_pieces.append(f"Title: {title}\nSummary: {snippet}")

                except Exception:
                    continue

            return "\n\n".join(content_pieces) if content_pieces else "No Scholar results found"

        except Exception as e:
            logger.error("Error extracting Scholar content: %s", e)
            return "Scholar extraction failed"

    def _extract_duckduckgo_content(self) -> str:
        """Extract content from DuckDuckGo search results"""
        try:
            # Look for search results
            results = self.driver.find_elements(By.CSS_SELECTOR, "[data-result]")
            content_pieces = []

            for result in results[:5]:  # Top 5 results
                try:
                    title_elem = result.find_element(By.CSS_SELECTOR, ".result__title a")
                    title = title_elem.text.strip()

                    snippet_elem = result.find_element(By.CSS_SELECTOR, ".result__snippet")
                    snippet = snippet_elem.text.strip()

                    if title and snippet:
                        content_pieces.append(f"{title}: {snippet}")

                except Exception:
                    continue

            return "\n\n".join(content_pieces) if content_pieces else "No search results found"

        except Exception as e:
            logger.error("Error extracting DuckDuckGo content: %s", e)
            return "Search extraction failed"

    def _extract_generic_content(self) -> str:
        """Generic content extraction for unknown pages"""
        try:
            # Try to find main content areas
            content_selectors = [
                "main", "article", ".content", "#content",
                ".post-content", ".entry-content", "p"
            ]

            for selector in content_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        content = " ".join([elem.text.strip() for elem in elements[:3]])
                        if len(content) > 100:
                            return content[:1500]
                except Exception:
                    continue

            return "Generic content extraction failed"

        except Exception as e:
            logger.error("Error with generic extraction: %s", e)
            return "Content extraction failed"

    def _calculate_relevance(self, content: str, topic: str) -> float:
        """Simple relevance scoring based on keyword matches"""
        if not content or not topic:
            return 0.0

        topic_words = topic.lower().split()
        content_lower = content.lower()

        matches = sum(1 for word in topic_words if word in content_lower)
        return matches / len(topic_words) if topic_words else 0.0

    def _generate_summary(self, sources: List[Dict], topic: str) -> str:
        """Generate summary from research sources"""
        valid_sources = [s for s in sources if "content" in s and s["content"]]

        if not valid_sources:
            return f"No reliable information found about {topic}"

        # Combine content from all sources
        combined_content = "\n\n".join([s["content"] for s in valid_sources])

        # Simple extractive summary (first sentences from each source)
        sentences = []
        for source in valid_sources:
            content = source["content"]
            # Extract first meaningful sentence
            first_sentence = content.split('.')[0] + '.'
            if len(first_sentence) > 20:  # Ensure it's substantial
                sentences.append(first_sentence)

        return " ".join(sentences[:3]) if sentences else "Summary generation failed"

    def _extract_key_points(self, sources: List[Dict]) -> List[str]:
        """Extract key points from sources"""
        key_points = []

        for source in sources:
            if "content" not in source:
                continue

            content = source["content"]

            # Look for numbered lists, bullet points, or sentences with key indicators
            key_indicators = ["important", "key", "main", "significant", "major", "primary"]

            sentences = content.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if any(indicator in sentence.lower() for indicator in key_indicators):
                    if 10 < len(sentence) < 200:  # Reasonable length
                        key_points.append(sentence + ".")

                if len(key_points) >= 5:  # Limit number of key points
                    break

        return list(set(key_points))[:5]  # Remove duplicates and limit

    async def navigate_to_url(self, url: str) -> Dict:
        """Navigate to a specific URL"""
        if not self.driver:
            return {"error": "Browser driver not available"}

        try:
            self.driver.get(url)
            WebDriverWait(self.driver, 10).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )

            return {
                "url": url,
                "title": self.driver.title,
                "status": "success"
            }
        except Exception as e:
            return {"error": f"Failed to navigate to {url}: {str(e)}"}

    async def extract_page_content(self) -> Dict:
        """Extract content from current page"""
        if not self.driver:
            return {"error": "Browser driver not available"}

        try:
            return {
                "title": self.driver.title,
                "url": self.driver.current_url,
                "content": self._extract_generic_content(),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": f"Failed to extract content: {str(e)}"}

    def close(self):
        """Close browser driver"""
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass
