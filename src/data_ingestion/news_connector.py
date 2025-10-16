"""
News API connector for financial news ingestion
"""
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import time
from urllib.parse import urlencode

from .base_connector import BaseConnector, DataPoint
from config.config import config

class NewsConnector(BaseConnector):
    """Connector for NewsAPI financial news"""

    def __init__(self):
        super().__init__("NewsAPI", {"api_key": config.api.news_api_key})
        self.base_url = "https://newsapi.org/v2"
        self.rate_limit_delay = 1  # seconds between requests
        self.last_request_time = 0

    def connect(self) -> bool:
        """Test connection to NewsAPI"""
        try:
            response = requests.get(
                f"{self.base_url}/top-headlines",
                params={
                    "apiKey": self.config["api_key"],
                    "category": "business",
                    "pageSize": 1
                },
                timeout=10
            )

            if response.status_code == 200:
                self._is_connected = True
                self.logger.info("Successfully connected to NewsAPI")
                return True
            else:
                self.logger.error(f"NewsAPI connection failed:
    {response.status_code}")
                return False

        except Exception as e:
            self.logger.error(f"NewsAPI connection error: {str(e)}")
            return False

    def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()

    def fetch_financial_news(self,
                           tickers: Optional[List[str]] = None,
                           hours_back: int = 24,
                           sources: Optional[List[str]] = None) ->
    List[DataPoint]:
        """Fetch financial news articles"""
        if not self._is_connected:
            if not self.connect():
                return []

        articles = []
        tickers = tickers or config.monitored_tickers
        sources = sources or config.news_sources

        # Create search queries for each ticker
        queries = []
        for ticker in tickers[:5]:  # Limit to avoid API quota
            queries.append(f"{ticker} stock OR {ticker} earnings OR {ticker}
    financial")

        # Add general financial terms
        queries.append("stock market OR financial markets OR earnings OR IPO")

        for query in queries:
            self._rate_limit()

            params = {
                "apiKey": self.config["api_key"],
                "q": query,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 20,
                "from": (datetime.now() -
    timedelta(hours=hours_back)).isoformat()
            }

            if sources:
                params["sources"] = ",".join(sources)

            try:
                response = requests.get(
                    f"{self.base_url}/everything",
                    params=params,
                    timeout=15
                )

                if response.status_code == 200:
                    data = response.json()
                    for article in data.get("articles", []):
                        articles.append(self._convert_to_datapoint(article,
    query))
                else:
                    self.logger.warning(f"NewsAPI request failed:
    {response.status_code}")

            except Exception as e:
                self.logger.error(f"Error fetching news for query '{query}':
    {str(e)}")

        self.logger.info(f"Fetched {len(articles)} news articles")
        return articles

    def _convert_to_datapoint(self, article: Dict[str, Any], query: str) ->
    DataPoint:
        """Convert NewsAPI article to DataPoint"""
        # Extract ticker from query if possible
        ticker = None
        for t in config.monitored_tickers:
            if t in query.upper():
                ticker = t
                break

        # Calculate credibility score
        source_info = {
            "domain": article.get("source", {}).get("name", "").lower(),
            "author": article.get("author", ""),
            "url": article.get("url", "")
        }
        credibility = self.get_credibility_score(source_info)

        # Combine title and description for content
        content = f"{article.get('title', '')} {article.get('description',
    '')}"

        return DataPoint(
            source="NewsAPI",
            timestamp=datetime.fromisoformat(article["publishedAt"].replace("Z"
    , "+00:00")),
            content=content.strip(),
            ticker=ticker,
            credibility_score=credibility,
            metadata={
                "title": article.get("title", ""),
                "description": article.get("description", ""),
                "url": article.get("url", ""),
                "author": article.get("author", ""),
                "source_name": article.get("source", {}).get("name", ""),
                "query": query,
                "url_to_image": article.get("urlToImage", "")
            }
        )

    def fetch_data(self, **kwargs) -> List[DataPoint]:
        """Main fetch method"""
        return self.fetch_financial_news(**kwargs)

    def is_healthy(self) -> bool:
        """Check if NewsAPI connection is healthy"""
        return self._is_connected and self.connect()
