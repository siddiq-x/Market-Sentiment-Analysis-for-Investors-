"""
Pytest configuration and fixtures for the test suite.
"""
import os
from unittest.mock import MagicMock, patch
import pytest

# Add the project root to the Python path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
    '..')))

# Fixtures for testing


@pytest.fixture
def sample_news_data():
    """Sample news data for testing."""
    return {
        "title": "Tech Company Reports Record Earnings",
        "content": "The company reported a 20% increase in quarterly revenue.",
        "source": "Financial Times",
        "published_at": "2025-01-01T12:00:00Z"
    }


@pytest.fixture
def sample_tweet_data():
    """Sample tweet data for testing."""
    return {
        "id": "1234567890",
        "text": "$AAPL stock is soaring after the latest earnings report! #investing",
        "created_at": "2025-01-01T12:30:00Z",
        "author_id": "9876543210",
        "public_metrics": {
            "retweet_count": 42,
            "reply_count": 10,
            "like_count": 150,
            "quote_count": 5
        }
    }

@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing."""
    with patch.dict(os.environ, {
        "NEWS_API_KEY": "test_news_api_key",
        "TWITTER_BEARER_TOKEN": "test_twitter_token",
        "MONGODB_URI": "mongodb://test:test@localhost:27017/test_db"
    }):
        yield
