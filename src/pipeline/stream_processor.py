"""
Real-time stream processing pipeline for sentiment and market data
"""
import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from queue import Queue, Empty
import redis
from collections import defaultdict, deque

from ..data_ingestion.base_connector import DataPoint
from ..data_ingestion.ingestion_manager import ingestion_manager
from ..preprocessing.text_cleaner import TextCleaner
from ..preprocessing.ner_extractor import FinancialNER
from ..preprocessing.bot_detector import BotDetector
from ..sentiment.ensemble_analyzer import EnsembleSentimentAnalyzer
from ..fusion.fusion_manager import FusionManager
from config.config import config


@dataclass
class StreamMessage:
    """Container for stream messages"""
    message_id: str
    topic: str
    data: Dict[str, Any]
    timestamp: datetime
    processed: bool = False


@dataclass
class ProcessingResult:
    """Container for processing results"""
    message_id: str
    original_data: DataPoint
    cleaned_text: Optional[str]
    entities: List[Dict[str, Any]]
    sentiment_result: Optional[Dict[str, Any]]
    fusion_prediction: Optional[Dict[str, Any]]
    processing_time_ms: float
    timestamp: datetime


class KafkaSimulator:
    """Simulates Kafka streaming for development/testing"""

    def __init__(self):
        self.logger = logging.getLogger("kafka_simulator")
        self.topics = defaultdict(deque)
        self.subscribers = defaultdict(list)
        self.running = False
        self._lock = threading.Lock()

    def produce(self, topic: str, message: Dict[str, Any]):
        """Produce message to topic"""
        with self._lock:
            stream_message = StreamMessage(
                message_id=f"{topic}_{int(time.time() *
    1000)}_{len(self.topics[topic])}",
                topic=topic,
                data=message,
                timestamp=datetime.now()
            )
            self.topics[topic].append(stream_message)

            # Notify subscribers
            for callback in self.subscribers[topic]:
                try:
                    callback(stream_message)
                except Exception as e:
                    self.logger.error(f"Error in subscriber callback:
    {str(e)}")

    def subscribe(self, topic: str, callback: Callable[[StreamMessage], None]):
        """Subscribe to topic"""
        with self._lock:
            self.subscribers[topic].append(callback)
            self.logger.info(f"Subscribed to topic: {topic}")

    def get_messages(self, topic: str, max_messages: int = 100) ->
    List[StreamMessage]:
        """Get messages from topic"""
        with self._lock:
            messages = []
            for _ in range(min(max_messages, len(self.topics[topic]))):
                if self.topics[topic]:
                    messages.append(self.topics[topic].popleft())
            return messages

class StreamProcessor:
    """Main stream processing engine"""

    def __init__(self,
                 batch_size: int = 10,
                 processing_interval: float = 1.0,
                 max_workers: int = 4):
        self.logger = logging.getLogger("stream_processor")

        self.batch_size = batch_size
        self.processing_interval = processing_interval
        self.max_workers = max_workers

        # Initialize components
        self.kafka_sim = KafkaSimulator()
        self.text_cleaner = TextCleaner()
        self.ner_extractor = FinancialNER()
        self.bot_detector = BotDetector()
        self.sentiment_analyzer = EnsembleSentimentAnalyzer()
        self.fusion_manager = FusionManager()

        # Processing queues
        self.raw_queue = Queue()
        self.processed_queue = Queue()

        # State tracking
        self.running = False
        self.processing_stats = {
            "messages_processed": 0,
            "processing_errors": 0,
            "avg_processing_time": 0.0,
            "start_time": None
        }

        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Redis for caching (optional)
        try:
            self.redis_client = redis.Redis(
                host=config.database.redis_host,
                port=config.database.redis_port,
                decode_responses=True
            )
            self.redis_available = True
        except:
            self.redis_client = None
            self.redis_available = False
            self.logger.warning("Redis not available, using in-memory caching")

        # Setup topic subscriptions
        self._setup_subscriptions()

        self.logger.info("Stream processor initialized")

    def _setup_subscriptions(self):
        """Setup Kafka topic subscriptions"""
        self.kafka_sim.subscribe(config.kafka.topic_news,
    self._handle_news_message)
        self.kafka_sim.subscribe(config.kafka.topic_social,
    self._handle_social_message)
        self.kafka_sim.subscribe(config.kafka.topic_market,
    self._handle_market_message)

    def start(self):
        """Start the stream processor"""
        if self.running:
            self.logger.warning("Stream processor already running")
            return

        self.running = True
        self.processing_stats["start_time"] = datetime.now()

        # Start processing threads
        processing_thread = threading.Thread(target=self._processing_loop)
        processing_thread.daemon = True
        processing_thread.start()

        # Start data ingestion simulation
        ingestion_thread = threading.Thread(target=self._simulate_data_ingestio
    n)
        ingestion_thread.daemon = True
        ingestion_thread.start()

        self.logger.info("Stream processor started")

    def stop(self):
        """Stop the stream processor"""
        self.running = False
        self.executor.shutdown(wait=True)
        self.logger.info("Stream processor stopped")

    def _simulate_data_ingestion(self):
        """Simulate continuous data ingestion"""
        while self.running:
            try:
                # Fetch data from connectors
                tickers = config.monitored_tickers[:5]  # Limit for simulation

                # Simulate news data
                if hasattr(ingestion_manager.connectors.get('news'),
    'fetch_data'):
                    try:
                        news_data = ingestion_manager.fetch_by_connector('news'
    , tickers=tickers, hours_back=1)
                        for data_point in news_data[-5:]:  # Last 5 items
                            self.kafka_sim.produce(config.kafka.topic_news,
    self._datapoint_to_dict(data_point))
                    except Exception as e:
                        self.logger.error(f"Error fetching news data:
    {str(e)}")

                # Simulate market data
                if hasattr(ingestion_manager.connectors.get('market'),
    'fetch_data'):
                    try:
                        market_data = ingestion_manager.fetch_by_connector('mar
    ket', tickers=tickers)
                        for data_point in market_data[-10:]:  # Last 10 items
                            self.kafka_sim.produce(config.kafka.topic_market,
    self._datapoint_to_dict(data_point))
                    except Exception as e:
                        self.logger.error(f"Error fetching market data:
    {str(e)}")

                # Simulate social media data
                if hasattr(ingestion_manager.connectors.get('twitter'),
    'fetch_data'):
                    try:
                        social_data = ingestion_manager.fetch_by_connector('twi
    tter', tickers=tickers, max_tweets=20)
                        for data_point in social_data[-3:]:  # Last 3 items
                            self.kafka_sim.produce(config.kafka.topic_social,
    self._datapoint_to_dict(data_point))
                    except Exception as e:
                        self.logger.error(f"Error fetching social data:
    {str(e)}")

                # Wait before next ingestion cycle
                time.sleep(30)  # Fetch new data every 30 seconds

            except Exception as e:
                self.logger.error(f"Error in data ingestion simulation:
    {str(e)}")
                time.sleep(5)

    def _datapoint_to_dict(self, data_point: DataPoint) -> Dict[str, Any]:
        """Convert DataPoint to dictionary for streaming"""
        return {
            "source": data_point.source,
            "timestamp": data_point.timestamp.isoformat(),
            "content": data_point.content,
            "ticker": data_point.ticker,
            "credibility_score": data_point.credibility_score,
            "metadata": data_point.metadata
        }

    def _handle_news_message(self, message: StreamMessage):
        """Handle news stream messages"""
        self.raw_queue.put(('news', message))

    def _handle_social_message(self, message: StreamMessage):
        """Handle social media stream messages"""
        self.raw_queue.put(('social', message))

    def _handle_market_message(self, message: StreamMessage):
        """Handle market data stream messages"""
        self.raw_queue.put(('market', message))

    def _processing_loop(self):
        """Main processing loop"""
        while self.running:
            try:
                # Collect batch of messages
                batch = []
                start_time = time.time()

                # Collect messages for batch processing
                while len(batch) < self.batch_size and (time.time() -
    start_time) < self.processing_interval:
                    try:
                        message_type, message = self.raw_queue.get(timeout=0.1)
                        batch.append((message_type, message))
                    except Empty:
                        continue

                if batch:
                    # Process batch
                    futures = []
                    for message_type, message in batch:
                        future = self.executor.submit(self._process_message,
    message_type, message)
                        futures.append(future)

                    # Collect results
                    for future in futures:
                        try:
                            result = future.result(timeout=10)  # 10 second
    timeout
                            if result:
                                self.processed_queue.put(result)
                                self.processing_stats["messages_processed"] +=
    1
                        except Exception as e:
                            self.logger.error(f"Error processing message:
    {str(e)}")
                            self.processing_stats["processing_errors"] += 1

                # Small delay to prevent CPU spinning
                time.sleep(0.1)

            except Exception as e:
                self.logger.error(f"Error in processing loop: {str(e)}")
                time.sleep(1)

    def _process_message(self, message_type: str, message: StreamMessage) ->
    Optional[ProcessingResult]:
        """Process individual message"""
        start_time = time.time()

        try:
            # Convert message data back to DataPoint
            data_point = self._dict_to_datapoint(message.data)

            # Skip processing for market data (no text to analyze)
            if message_type == 'market':
                return ProcessingResult(
                    message_id=message.message_id,
                    original_data=data_point,
                    cleaned_text=None,
                    entities=[],
                    sentiment_result=None,
                    fusion_prediction=None,
                    processing_time_ms=(time.time() - start_time) * 1000,
                    timestamp=datetime.now()
                )

            # Text preprocessing
            cleaned_text_obj = self.text_cleaner.clean_text(
                data_point.content,
                preserve_tickers=True,
                preserve_numbers=True
            )

            # Named entity recognition
            entities = self.ner_extractor.extract_entities(data_point.content)

            # Bot detection (for social media)
            if message_type == 'social':
                bot_detection = self.bot_detector.detect_bot(
                    data_point.content,
                    author_info=data_point.metadata.get('author_info'),
                    metadata=data_point.metadata
                )
                data_point.metadata['bot_detection'] = {
                    'is_bot_likely': bot_detection.is_bot_likely,
                    'confidence': bot_detection.confidence,
                    'reasons': bot_detection.reasons,
                    'credibility_adjustment':
    bot_detection.credibility_adjustment
                }

                # Skip further processing for likely bot content
                if bot_detection.is_bot_likely:
                    self.logger.debug(f"Skipping likely bot content:
    {data_point.content[:100]}...")
                    return None

            # Sentiment analysis
            sentiment_result = self.sentiment_analyzer.analyze_sentiment(
                cleaned_text_obj.cleaned,
                ticker=data_point.ticker,
                source=data_point.source
            )

            # Prepare entities for output
            entity_list = [{
                'text': e.text,
                'label': e.label,
                'confidence': e.confidence
            } for e in entities]

            # Create processing result
            processing_time_ms = (time.time() - start_time) * 1000

            return ProcessingResult(
                message_id=message.message_id,
                original_data=data_point,
                cleaned_text=cleaned_text_obj.cleaned,
                entities=entity_list,
                sentiment_result={
                    'sentiment': sentiment_result.sentiment,
                    'confidence': sentiment_result.confidence,
                    'ensemble_score': sentiment_result.ensemble_score,
                    'finbert_available': sentiment_result.finbert_result is
    not None,
                    'lexicon_available': sentiment_result.lexicon_result is
    not None
                },
                fusion_prediction=None,  # Will be added in batch processing
                processing_time_ms=processing_time_ms,
                timestamp=datetime.now()
            )

        except Exception as e:
            self.logger.error(f"Error processing message {message.message_id}:
    {str(e)}")
            return None

    def _dict_to_datapoint(self, data: Dict[str, Any]) -> DataPoint:
        """Convert dictionary back to DataPoint"""
        return DataPoint(
            source=data["source"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            content=data["content"],
            ticker=data.get("ticker"),
            credibility_score=data.get("credibility_score", 1.0),
            metadata=data.get("metadata", {})
        )

    def get_processed_results(self, max_results: int = 100) ->
    List[ProcessingResult]:
        """Get processed results from queue"""
        results = []
        for _ in range(max_results):
            try:
                result = self.processed_queue.get_nowait()
                results.append(result)
            except Empty:
                break
        return results

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        stats = self.processing_stats.copy()

        if stats["start_time"]:
            runtime = (datetime.now() - stats["start_time"]).total_seconds()
            stats["runtime_seconds"] = runtime
            stats["messages_per_second"] = stats["messages_processed"] /
    runtime if runtime > 0 else 0

        stats["queue_sizes"] = {
            "raw_queue": self.raw_queue.qsize(),
            "processed_queue": self.processed_queue.qsize()
        }

        stats["redis_available"] = self.redis_available
        stats["running"] = self.running

        return stats

    def get_recent_sentiment_data(self, ticker: str, hours_back: int = 24) ->
    List[Dict[str, Any]]:
        """Get recent sentiment data for a ticker from cache"""
        if not self.redis_available:
            return []

        try:
            # Get all sentiment keys for the ticker
            pattern = f"sentiment:{ticker}:*"
            keys = self.redis_client.keys(pattern)

            sentiment_data = []
            cutoff_time = datetime.now() - timedelta(hours=hours_back)

            for key in keys:
                data = json.loads(self.redis_client.get(key))
                timestamp = datetime.fromisoformat(data["timestamp"])

                if timestamp >= cutoff_time:
                    sentiment_data.append(data)

            # Sort by timestamp
            sentiment_data.sort(key=lambda x: x["timestamp"])

            return sentiment_data

        except Exception as e:
            self.logger.error(f"Error retrieving sentiment data: {str(e)}")
            return []

    def trigger_fusion_prediction(self, ticker: str) -> Optional[Dict[str,
    Any]]:
        """Trigger fusion prediction for a ticker"""
        try:
            # Get recent sentiment data
            sentiment_data = self.get_recent_sentiment_data(ticker,
    hours_back=24)

            if len(sentiment_data) < 5:  # Need minimum data points
                return None

            # This would integrate with the fusion manager
            # For now, return a mock prediction
            return {
                "ticker": ticker,
                "prediction": 0,  # neutral
                "confidence": 0.75,
                "timestamp": datetime.now().isoformat(),
                "data_points_used": len(sentiment_data)
            }

        except Exception as e:
            self.logger.error(f"Error in fusion prediction: {str(e)}")
            return None
