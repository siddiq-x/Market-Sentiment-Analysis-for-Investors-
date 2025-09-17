"""
Social media connector for Twitter/X financial sentiment data
"""
import tweepy
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import time
import re

from .base_connector import BaseConnector, DataPoint
from config.config import config

class SocialConnector(BaseConnector):
    """Connector for Twitter/X financial discussions"""
    
    def __init__(self):
        super().__init__("Twitter", {"bearer_token": config.api.twitter_bearer_token})
        self.client = None
        self.rate_limit_delay = 1  # seconds between requests
        self.last_request_time = 0
    
    def connect(self) -> bool:
        """Initialize Twitter API client"""
        try:
            if not self.config["bearer_token"]:
                self.logger.warning("Twitter Bearer Token not configured")
                return False
            
            self.client = tweepy.Client(
                bearer_token=self.config["bearer_token"],
                wait_on_rate_limit=True
            )
            
            # Test connection
            test_query = "stock market -is:retweet lang:en"
            tweets = self.client.search_recent_tweets(
                query=test_query,
                max_results=10,
                tweet_fields=['created_at', 'author_id', 'public_metrics']
            )
            
            if tweets.data:
                self._is_connected = True
                self.logger.info("Successfully connected to Twitter API")
                return True
            else:
                self.logger.error("Twitter API test failed - no data returned")
                return False
                
        except Exception as e:
            self.logger.error(f"Twitter API connection error: {str(e)}")
            return False
    
    def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
    def _extract_tickers(self, text: str) -> List[str]:
        """Extract stock tickers from text"""
        # Pattern for stock tickers ($ followed by 1-5 uppercase letters)
        ticker_pattern = r'\$([A-Z]{1,5})\b'
        tickers = re.findall(ticker_pattern, text.upper())
        
        # Also check for common ticker mentions without $
        for ticker in config.monitored_tickers:
            if ticker.upper() in text.upper():
                tickers.append(ticker)
        
        return list(set(tickers))  # Remove duplicates
    
    def _calculate_engagement_score(self, metrics: Dict[str, int]) -> float:
        """Calculate engagement score based on tweet metrics"""
        likes = metrics.get('like_count', 0)
        retweets = metrics.get('retweet_count', 0)
        replies = metrics.get('reply_count', 0)
        
        # Weighted engagement score
        score = (likes * 1) + (retweets * 3) + (replies * 2)
        
        # Normalize to 0-1 scale (log scale for viral content)
        import math
        normalized = min(1.0, math.log10(score + 1) / 4)  # log10(10000) ≈ 4
        
        return normalized
    
    def fetch_financial_tweets(self, 
                              tickers: Optional[List[str]] = None,
                              hours_back: int = 24,
                              max_tweets: int = 100) -> List[DataPoint]:
        """Fetch tweets about financial topics"""
        if not self._is_connected:
            if not self.connect():
                return []
        
        tickers = tickers or config.monitored_tickers[:10]  # Limit for API quota
        tweets_data = []
        
        # Create search queries
        queries = []
        
        # Ticker-specific queries
        for ticker in tickers[:5]:  # Limit to avoid API quota
            query = f"${ticker} OR {ticker} (stock OR price OR earnings OR buy OR sell) -is:retweet lang:en"
            queries.append((query, ticker))
        
        # General financial queries
        general_queries = [
            "stock market OR stocks OR trading -is:retweet lang:en",
            "bull market OR bear market OR market crash -is:retweet lang:en",
            "earnings OR dividend OR IPO -is:retweet lang:en"
        ]
        
        for query in general_queries:
            queries.append((query, None))
        
        for query, associated_ticker in queries:
            self._rate_limit()
            
            try:
                tweets = self.client.search_recent_tweets(
                    query=query,
                    max_results=min(max_tweets // len(queries), 100),
                    tweet_fields=['created_at', 'author_id', 'public_metrics', 'context_annotations'],
                    start_time=datetime.now() - timedelta(hours=hours_back)
                )
                
                if tweets.data:
                    for tweet in tweets.data:
                        tweets_data.append(self._convert_to_datapoint(tweet, associated_ticker))
                
            except Exception as e:
                self.logger.error(f"Error fetching tweets for query '{query}': {str(e)}")
        
        self.logger.info(f"Fetched {len(tweets_data)} tweets")
        return tweets_data
    
    def _convert_to_datapoint(self, tweet, associated_ticker: Optional[str] = None) -> DataPoint:
        """Convert tweet to DataPoint"""
        # Extract tickers from tweet text
        tickers = self._extract_tickers(tweet.text)
        primary_ticker = associated_ticker or (tickers[0] if tickers else None)
        
        # Calculate credibility based on engagement and account metrics
        engagement_score = self._calculate_engagement_score(tweet.public_metrics)
        base_credibility = 0.60  # Base credibility for social media
        
        # Adjust credibility based on engagement (higher engagement = higher credibility)
        credibility = min(0.85, base_credibility + (engagement_score * 0.25))
        
        return DataPoint(
            source="Twitter",
            timestamp=tweet.created_at,
            content=tweet.text,
            ticker=primary_ticker,
            credibility_score=credibility,
            metadata={
                "tweet_id": tweet.id,
                "author_id": tweet.author_id,
                "like_count": tweet.public_metrics.get('like_count', 0),
                "retweet_count": tweet.public_metrics.get('retweet_count', 0),
                "reply_count": tweet.public_metrics.get('reply_count', 0),
                "quote_count": tweet.public_metrics.get('quote_count', 0),
                "engagement_score": engagement_score,
                "extracted_tickers": tickers,
                "context_annotations": getattr(tweet, 'context_annotations', []),
                "data_type": "social_media"
            }
        )
    
    def fetch_data(self, **kwargs) -> List[DataPoint]:
        """Main fetch method"""
        return self.fetch_financial_tweets(**kwargs)
    
    def is_healthy(self) -> bool:
        """Check if Twitter connection is healthy"""
        return self._is_connected and self.client is not None


class RedditConnector(BaseConnector):
    """Connector for Reddit financial discussions (using public API)"""
    
    def __init__(self):
        super().__init__("Reddit", {})
        self.base_url = "https://www.reddit.com"
        self.rate_limit_delay = 2  # Reddit rate limiting
        self.last_request_time = 0
    
    def connect(self) -> bool:
        """Test connection to Reddit"""
        try:
            response = requests.get(
                f"{self.base_url}/r/investing/hot.json",
                headers={'User-Agent': 'MarketSentiment/1.0'},
                timeout=10
            )
            
            if response.status_code == 200:
                self._is_connected = True
                self.logger.info("Successfully connected to Reddit")
                return True
            else:
                self.logger.error(f"Reddit connection failed: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Reddit connection error: {str(e)}")
            return False
    
    def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
    def fetch_reddit_posts(self, 
                          subreddits: Optional[List[str]] = None,
                          max_posts: int = 50) -> List[DataPoint]:
        """Fetch posts from financial subreddits"""
        if not self._is_connected:
            if not self.connect():
                return []
        
        subreddits = subreddits or ['investing', 'stocks', 'SecurityAnalysis', 'ValueInvesting', 'wallstreetbets']
        posts_data = []
        
        for subreddit in subreddits:
            self._rate_limit()
            
            try:
                response = requests.get(
                    f"{self.base_url}/r/{subreddit}/hot.json",
                    headers={'User-Agent': 'MarketSentiment/1.0'},
                    params={'limit': max_posts // len(subreddits)},
                    timeout=15
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    for post in data.get('data', {}).get('children', []):
                        post_data = post.get('data', {})
                        posts_data.append(self._convert_reddit_to_datapoint(post_data, subreddit))
                
            except Exception as e:
                self.logger.error(f"Error fetching from r/{subreddit}: {str(e)}")
        
        self.logger.info(f"Fetched {len(posts_data)} Reddit posts")
        return posts_data
    
    def _convert_reddit_to_datapoint(self, post_data: Dict[str, Any], subreddit: str) -> DataPoint:
        """Convert Reddit post to DataPoint"""
        title = post_data.get('title', '')
        selftext = post_data.get('selftext', '')
        content = f"{title} {selftext}".strip()
        
        # Extract tickers
        tickers = self._extract_tickers(content)
        primary_ticker = tickers[0] if tickers else None
        
        # Calculate credibility based on subreddit and engagement
        subreddit_scores = {
            'investing': 0.75,
            'SecurityAnalysis': 0.80,
            'ValueInvesting': 0.78,
            'stocks': 0.70,
            'wallstreetbets': 0.45  # Lower credibility due to meme nature
        }
        
        base_credibility = subreddit_scores.get(subreddit, 0.60)
        upvote_ratio = post_data.get('upvote_ratio', 0.5)
        score = post_data.get('score', 0)
        
        # Adjust credibility based on community approval
        credibility = base_credibility * upvote_ratio
        if score > 100:
            credibility = min(0.85, credibility + 0.05)
        
        return DataPoint(
            source="Reddit",
            timestamp=datetime.fromtimestamp(post_data.get('created_utc', 0)),
            content=content,
            ticker=primary_ticker,
            credibility_score=credibility,
            metadata={
                "post_id": post_data.get('id'),
                "subreddit": subreddit,
                "author": post_data.get('author'),
                "score": post_data.get('score', 0),
                "upvote_ratio": post_data.get('upvote_ratio', 0),
                "num_comments": post_data.get('num_comments', 0),
                "url": post_data.get('url'),
                "extracted_tickers": tickers,
                "data_type": "social_media"
            }
        )
    
    def _extract_tickers(self, text: str) -> List[str]:
        """Extract stock tickers from text"""
        ticker_pattern = r'\$([A-Z]{1,5})\b'
        tickers = re.findall(ticker_pattern, text.upper())
        
        for ticker in config.monitored_tickers:
            if ticker.upper() in text.upper():
                tickers.append(ticker)
        
        return list(set(tickers))
    
    def fetch_data(self, **kwargs) -> List[DataPoint]:
        """Main fetch method"""
        return self.fetch_reddit_posts(**kwargs)
    
    def is_healthy(self) -> bool:
        """Check if Reddit connection is healthy"""
        return self._is_connected
