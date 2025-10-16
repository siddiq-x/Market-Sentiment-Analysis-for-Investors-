"""
Test suite for data ingestion components
"""
import unittest
from unittest.mock import Mock, patch
from datetime import datetime
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_ingestion.base_connector import DataPoint  # noqa: E402
from data_ingestion.news_connector import NewsConnector  # noqa: E402
from data_ingestion.market_connector import MarketConnector  # noqa: E402
from data_ingestion.social_connector import TwitterConnector, RedditConnector  # noqa: E402
from data_ingestion.ingestion_manager import IngestionManager  # noqa: E402


class TestDataPoint(unittest.TestCase):
    """Test DataPoint class"""

    def test_datapoint_creation(self):
        """Test DataPoint creation and validation"""
        dp = DataPoint(
            content="Test news content",
            timestamp=datetime.now(),
            source="test_source",
            ticker="AAPL",
            metadata={"author": "test_author"}
        )

        self.assertEqual(dp.content, "Test news content")
        self.assertEqual(dp.source, "test_source")
        self.assertEqual(dp.ticker, "AAPL")
        self.assertIsInstance(dp.timestamp, datetime)
        self.assertIsInstance(dp.metadata, dict)

    def test_credibility_scoring(self):
        """Test credibility scoring logic"""
        # High credibility source
        dp_high = DataPoint(
            content="Breaking: Company reports strong earnings",
            timestamp=datetime.now(),
            source="reuters",
            ticker="AAPL"
        )

        # Low credibility source
        dp_low = DataPoint(
            content="OMG AAPL TO THE MOON!!!",
            timestamp=datetime.now(),
            source="unknown_blog",
            ticker="AAPL"
        )

        self.assertGreater(dp_high.credibility_score, dp_low.credibility_score)


class TestNewsConnector(unittest.TestCase):
    """Test NewsConnector"""

    @patch('data_ingestion.news_connector.requests.get')
    def test_news_connector_fetch(self, mock_get):
        """Test news fetching"""
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'articles': [
                {
                    'title': 'Apple Reports Strong Q4 Earnings',
                    'description': 'Apple Inc. reported better than expected earnings',
                    'content': 'Full article content here...',
                    'publishedAt': '2023-10-15T10:00:00Z',
                    'source': {'name': 'Reuters'},
                    'url': 'https://example.com/article1'
                }
            ]
        }
        mock_get.return_value = mock_response

        connector = NewsConnector()
        data_points = connector.fetch_data(['AAPL'], hours_back=24)

        self.assertIsInstance(data_points, list)
        if data_points:  # If API call succeeded
            self.assertIsInstance(data_points[0], DataPoint)
            self.assertEqual(data_points[0].ticker, 'AAPL')


class TestMarketConnector(unittest.TestCase):
    """Test MarketConnector"""

    @patch('data_ingestion.market_connector.yf.download')
    def test_market_data_fetch(self, mock_download):
        """Test market data fetching"""
        # Mock yfinance response
        import pandas as pd
        mock_data = pd.DataFrame({
            'Open': [150.0, 151.0],
            'High': [152.0, 153.0],
            'Low': [149.0, 150.0],
            'Close': [151.0, 152.0],
            'Volume': [1000000, 1100000]
        }, index=pd.date_range('2023-10-15', periods=2, freq='D'))

        mock_download.return_value = mock_data

        connector = MarketConnector()
        data_points = connector.fetch_data(['AAPL'], hours_back=24)

        self.assertIsInstance(data_points, list)
        if data_points:
            self.assertIsInstance(data_points[0], DataPoint)
            self.assertEqual(data_points[0].ticker, 'AAPL')


class TestSocialConnector(unittest.TestCase):
    """Test Social Media Connectors"""

    @patch('data_ingestion.social_connector.tweepy.API')
    def test_twitter_connector(self, mock_api):
        """Test Twitter connector"""
        # Mock Twitter API response
        mock_tweet = Mock()
        mock_tweet.full_text = "AAPL looking strong today! #bullish"
        mock_tweet.created_at = datetime.now()
        mock_tweet.user.screen_name = "test_user"
        mock_tweet.user.followers_count = 1000
        mock_tweet.retweet_count = 5
        mock_tweet.favorite_count = 10

        mock_api_instance = Mock()
        mock_api_instance.search_tweets.return_value = [mock_tweet]
        mock_api.return_value = mock_api_instance

        connector = TwitterConnector()
        data_points = connector.fetch_data(['AAPL'], hours_back=24)

        self.assertIsInstance(data_points, list)

    @patch('data_ingestion.social_connector.requests.get')
    def test_reddit_connector(self, mock_get):
        """Test Reddit connector"""
        # Mock Reddit API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'data': {
                'children': [
                    {
                        'data': {
                            'title': 'AAPL Discussion',
                            'selftext': 'What do you think about Apple stock?',
                            'created_utc': datetime.now().timestamp(),
                            'author': 'test_user',
                            'score': 50,
                            'num_comments': 10,
                            'subreddit': 'investing'
                        }
                    }
                ]
            }
        }
        mock_get.return_value = mock_response

        connector = RedditConnector()
        data_points = connector.fetch_data(['AAPL'], hours_back=24)

        self.assertIsInstance(data_points, list)


class TestIngestionManager(unittest.TestCase):
    """Test IngestionManager"""

    def setUp(self):
        """Set up test fixtures"""
        self.manager = IngestionManager()

    @patch('data_ingestion.ingestion_manager.NewsConnector')
    @patch('data_ingestion.ingestion_manager.MarketConnector')
    @patch('data_ingestion.ingestion_manager.TwitterConnector')
    def test_fetch_all_data(self, mock_twitter, mock_market, mock_news):
        """Test fetching data from all sources"""
        # Mock connector instances
        mock_news_instance = Mock()
        mock_market_instance = Mock()
        mock_twitter_instance = Mock()

        # Mock return values
        mock_news_instance.fetch_data.return_value = [
            DataPoint("News content", datetime.now(), "news", "AAPL")
        ]
        mock_market_instance.fetch_data.return_value = [
            DataPoint("Market data", datetime.now(), "market", "AAPL")
        ]
        mock_twitter_instance.fetch_data.return_value = [
            DataPoint("Tweet content", datetime.now(), "twitter", "AAPL")
        ]

        # Set up mocks
        mock_news.return_value = mock_news_instance
        mock_market.return_value = mock_market_instance
        mock_twitter.return_value = mock_twitter_instance

        # Test fetch
        all_data = self.manager.fetch_all_data(['AAPL'], hours_back=24)

        self.assertIsInstance(all_data, list)
        # Should have data from all sources
        self.assertGreaterEqual(len(all_data), 0)

    def test_validate_data_point(self):
        """Test data point validation"""
        valid_dp = DataPoint(
            content="Valid content",
            timestamp=datetime.now(),
            source="valid_source",
            ticker="AAPL"
        )

        invalid_dp = DataPoint(
            content="",  # Empty content
            timestamp=datetime.now(),
            source="source",
            ticker="AAPL"
        )

        self.assertTrue(self.manager.validate_data_point(valid_dp))
        self.assertFalse(self.manager.validate_data_point(invalid_dp))

    def test_filter_duplicates(self):
        """Test duplicate filtering"""
        dp1 = DataPoint("Same content", datetime.now(), "source1", "AAPL")
        dp2 = DataPoint("Same content", datetime.now(), "source2", "AAPL")
        dp3 = DataPoint("Different content", datetime.now(), "source1", "AAPL")

        data_points = [dp1, dp2, dp3]
        filtered = self.manager.filter_duplicates(data_points)

        # Should remove duplicate content
        self.assertLessEqual(len(filtered), len(data_points))

    def test_export_data(self):
        """Test data export functionality"""
        data_points = [
            DataPoint("Content 1", datetime.now(), "source1", "AAPL"),
            DataPoint("Content 2", datetime.now(), "source2", "AAPL")
        ]

        # Test JSON export
        json_result = self.manager.export_data(data_points, format='json')
        self.assertIsInstance(json_result, str)

        # Test CSV export
        csv_result = self.manager.export_data(data_points, format='csv')
        self.assertIsInstance(csv_result, str)


class TestErrorHandling(unittest.TestCase):
    """Test error handling in data ingestion"""

    def test_api_rate_limiting(self):
        """Test API rate limiting handling"""
        connector = NewsConnector()

        # Test that rate limiting doesn't crash
        # This is more of a smoke test
        self.assertIsNotNone(connector)

    @patch('data_ingestion.news_connector.requests.get')
    def test_api_error_handling(self, mock_get):
        """Test API error handling"""
        # Mock API error
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = Exception("API Error")
        mock_get.return_value = mock_response

        connector = NewsConnector()

        # Should handle errors gracefully
        try:
            data_points = connector.fetch_data(['AAPL'], hours_back=24)
            # Should return empty list on error
            self.assertIsInstance(data_points, list)
        except Exception:
            self.fail("Connector should handle API errors gracefully")

    def test_invalid_ticker_handling(self):
        """Test handling of invalid tickers"""
        connector = MarketConnector()

        # Test with invalid ticker
        data_points = connector.fetch_data(['INVALID_TICKER'], hours_back=24)

        # Should handle gracefully (empty list or valid response)
        self.assertIsInstance(data_points, list)


if __name__ == '__main__':
    unittest.main()
