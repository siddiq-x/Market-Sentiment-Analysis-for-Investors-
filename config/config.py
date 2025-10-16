"""
Configuration management for Market Sentiment Analysis System
"""
import os
from dataclasses import dataclass
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class APIConfig:
    """API configuration settings"""
    news_api_key: str = os.getenv('NEWS_API_KEY', '')
    twitter_bearer_token: str = os.getenv('TWITTER_BEARER_TOKEN', '')
    alpha_vantage_key: str = os.getenv('ALPHA_VANTAGE_API_KEY', '')
    fred_api_key: str = os.getenv('FRED_API_KEY', '')


@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    elasticsearch_host: str = os.getenv('ELASTICSEARCH_HOST', 'localhost')
    elasticsearch_port: int = int(os.getenv('ELASTICSEARCH_PORT', '9200'))
    redis_host: str = os.getenv('REDIS_HOST', 'localhost')
    redis_port: int = int(os.getenv('REDIS_PORT', '6379'))
    mongodb_uri: str = os.getenv(
        'MONGODB_URI',
        'mongodb://localhost:27017/sentiment_analysis'
    )


@dataclass
class KafkaConfig:
    """Kafka configuration settings"""
    bootstrap_servers: str = os.getenv(
        'KAFKA_BOOTSTRAP_SERVERS',
        'localhost:9092'
    )
    topic_news: str = os.getenv('KAFKA_TOPIC_NEWS', 'financial_news')
    topic_social: str = os.getenv('KAFKA_TOPIC_SOCIAL', 'social_media')
    topic_market: str = os.getenv('KAFKA_TOPIC_MARKET', 'market_data')


@dataclass
class ModelConfig:
    """Model configuration settings"""
    finbert_model_path: str = os.getenv('FINBERT_MODEL_PATH', 'models/finbert')
    sentiment_threshold: float = float(os.getenv('SENTIMENT_THRESHOLD', '0.1'))
    prediction_confidence_threshold: float = float(
        os.getenv('PREDICTION_CONFIDENCE_THRESHOLD', '0.7')
    )
    batch_size: int = int(os.getenv('BATCH_SIZE', '32'))
    max_sequence_length: int = 512


@dataclass
class DashboardConfig:
    """Dashboard configuration settings"""
    host: str = os.getenv('FLASK_HOST', '0.0.0.0')
    port: int = int(os.getenv('FLASK_PORT', '8080'))
    debug: bool = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'


@dataclass
class ProcessingConfig:
    """Processing configuration settings"""
    max_workers: int = int(os.getenv('MAX_WORKERS', '4'))
    processing_interval: int = int(os.getenv('PROCESSING_INTERVAL', '60'))
    log_level: str = os.getenv('LOG_LEVEL', 'INFO')


class Config:
    """Main configuration class"""

    def __init__(self):
        self.api = APIConfig()
        self.database = DatabaseConfig()
        self.kafka = KafkaConfig()
        self.model = ModelConfig()
        self.dashboard = DashboardConfig()
        self.processing = ProcessingConfig()

        # Financial tickers to monitor
        self.monitored_tickers = [
            'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
            'SPY', 'QQQ', 'IWM', 'GLD', 'BTC-USD', 'ETH-USD'
        ]

        # News sources configuration
        self.news_sources = [
            'reuters', 'bloomberg', 'cnbc', 'marketwatch', 'yahoo-finance',
            'financial-times', 'the-wall-street-journal'
        ]

        # Social media keywords
        self.social_keywords = [
            'stock', 'market', 'trading', 'investment', 'earnings', 'dividend',
            'bull', 'bear', 'rally', 'crash', 'volatility'
        ]

    def validate(self) -> List[str]:
        """
        Validate configuration and return list of missing required settings.

        Returns:
            List[str]: List of missing required settings with descriptions.
        """
        missing = []

        if not self.api.news_api_key:
            missing.append(
                'NEWS_API_KEY - Required for fetching news from newsapi.org')
        if not self.api.twitter_bearer_token:
            missing.append(
                'TWITTER_BEARER_TOKEN - Required for fetching tweets')
        if not self.api.alpha_vantage_key:
            missing.append(
                'ALPHA_VANTAGE_API_KEY - Required for fetching market data')
        if not self.api.fred_api_key:
            missing.append(
                'FRED_API_KEY - Required for fetching economic indicators')

        return missing


# Global configuration instance
config = Config()
