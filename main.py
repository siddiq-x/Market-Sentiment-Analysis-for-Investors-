"""
Main entry point for Market Sentiment Analysis System
"""
import argparse
import logging
import sys
import os
from datetime import datetime
import signal
import threading
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config.config import config
from src.data_ingestion.ingestion_manager import ingestion_manager
from src.pipeline.stream_processor import StreamProcessor
from src.dashboard.app import app, socketio
from src.sentiment.ensemble_analyzer import EnsembleSentimentAnalyzer
from src.fusion.fusion_manager import FusionManager
from src.utils.model_retrainer import ModelRetrainer

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.processing.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/market_sentiment.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("main")

class MarketSentimentSystem:
    """Main system orchestrator"""
    
    def __init__(self):
        self.logger = logging.getLogger("system")
        self.running = False
        
        # Initialize components
        self.stream_processor = None
        self.sentiment_analyzer = None
        self.fusion_manager = None
        self.model_retrainer = None
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("Market Sentiment Analysis System initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown()
        sys.exit(0)
    
    def validate_configuration(self):
        """Validate system configuration"""
        self.logger.info("Validating configuration...")
        
        # Check required directories
        os.makedirs('logs', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        os.makedirs('data', exist_ok=True)
        
        # Validate API configuration
        missing_config = config.validate()
        if missing_config:
            self.logger.warning(f"Missing configuration: {missing_config}")
            self.logger.warning("Some features may not work without proper API keys")
        
        # Validate data connectors
        validation_results = ingestion_manager.validate_configuration()
        if validation_results:
            self.logger.warning(f"Connector validation issues: {validation_results}")
        
        self.logger.info("Configuration validation completed")
    
    def initialize_components(self):
        """Initialize system components"""
        self.logger.info("Initializing components...")
        
        try:
            # Initialize sentiment analyzer
            self.sentiment_analyzer = EnsembleSentimentAnalyzer()
            self.logger.info("Sentiment analyzer initialized")
            
            # Initialize fusion manager
            self.fusion_manager = FusionManager()
            self.logger.info("Fusion manager initialized")
            
            # Initialize stream processor
            self.stream_processor = StreamProcessor()
            self.logger.info("Stream processor initialized")
            
            # Initialize model retrainer
            self.model_retrainer = ModelRetrainer()
            self.logger.info("Model retrainer initialized")
            
            # Test data connectors
            connection_status = ingestion_manager.connect_all()
            connected_count = sum(1 for status in connection_status.values() if status)
            self.logger.info(f"Connected to {connected_count}/{len(connection_status)} data sources")
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {str(e)}")
            raise
    
    def start_stream_processing(self):
        """Start stream processing"""
        if self.stream_processor:
            self.logger.info("Starting stream processing...")
            self.stream_processor.start()
            self.logger.info("Stream processing started")
        else:
            self.logger.error("Stream processor not initialized")
        
        # Start model retrainer if available
        if self.model_retrainer:
            self.model_retrainer.start()
            self.logger.info("Model retrainer started")
    
    def start_dashboard(self, host=None, port=None, debug=False):
        """Start web dashboard"""
        host = host or config.dashboard.host
        port = port or config.dashboard.port
        debug = debug or config.dashboard.debug
        
        self.logger.info(f"Starting dashboard on {host}:{port}")
        
        try:
            socketio.run(app, host=host, port=port, debug=debug)
        except Exception as e:
            self.logger.error(f"Error starting dashboard: {str(e)}")
            raise
    
    def run_batch_analysis(self, tickers=None, hours_back=24):
        """Run batch sentiment analysis"""
        tickers = tickers or config.monitored_tickers[:5]  # Limit for demo
        
        self.logger.info(f"Running batch analysis for {len(tickers)} tickers")
        
        try:
            # Fetch data
            all_data = ingestion_manager.fetch_all_data(
                tickers=tickers,
                hours_back=hours_back
            )
            
            if not all_data:
                self.logger.warning("No data fetched for analysis")
                return
            
            # Analyze sentiment for text data
            text_data = [dp for dp in all_data if dp.content and dp.content.strip()]
            
            if text_data:
                self.logger.info(f"Analyzing sentiment for {len(text_data)} text items")
                
                for data_point in text_data[:10]:  # Limit for demo
                    try:
                        result = self.sentiment_analyzer.analyze_sentiment(
                            data_point.content,
                            context={
                                "source": data_point.source,
                                "ticker": data_point.ticker
                            }
                        )
                        
                        self.logger.info(
                            f"Sentiment for {data_point.ticker or 'General'}: "
                            f"{result.sentiment} (confidence: {result.confidence:.3f})"
                        )
                        
                    except Exception as e:
                        self.logger.error(f"Error analyzing sentiment: {str(e)}")
            
            self.logger.info("Batch analysis completed")
            
        except Exception as e:
            self.logger.error(f"Error in batch analysis: {str(e)}")
    
    def run_training_demo(self, ticker="AAPL"):
        """Run a training demonstration"""
        self.logger.info(f"Running training demo for {ticker}")
        
        try:
            # This would normally use historical data
            # For demo, we'll just show the training process
            self.logger.info("Training demo completed - would train fusion model with historical data")
            
        except Exception as e:
            self.logger.error(f"Error in training demo: {str(e)}")
    
    def shutdown(self):
        """Shutdown system gracefully"""
        if self.running:
            self.logger.info("Shutting down system...")
            
            if self.stream_processor:
                self.stream_processor.stop()
            
            if self.model_retrainer:
                self.model_retrainer.stop()
            
            # Cleanup connectors
            ingestion_manager.cleanup()
            
            self.running = False
            self.logger.info("System shutdown completed")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Market Sentiment Analysis System")
    parser.add_argument("--mode", choices=["dashboard", "batch", "stream", "train"], 
                       default="dashboard", help="Operation mode")
    parser.add_argument("--host", default=None, help="Dashboard host")
    parser.add_argument("--port", type=int, default=None, help="Dashboard port")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--tickers", nargs="+", help="Tickers to analyze")
    parser.add_argument("--hours", type=int, default=24, help="Hours of data to analyze")
    parser.add_argument("--ticker", default="AAPL", help="Ticker for training demo")
    
    args = parser.parse_args()
    
    # Initialize system
    system = MarketSentimentSystem()
    
    try:
        # Validate configuration
        system.validate_configuration()
        
        # Initialize components
        system.initialize_components()
        
        # Run based on mode
        if args.mode == "dashboard":
            logger.info("Starting in dashboard mode")
            system.start_stream_processing()
            system.start_dashboard(host=args.host, port=args.port, debug=args.debug)
            
        elif args.mode == "batch":
            logger.info("Starting in batch analysis mode")
            system.run_batch_analysis(tickers=args.tickers, hours_back=args.hours)
            
        elif args.mode == "stream":
            logger.info("Starting in stream processing mode")
            system.start_stream_processing()
            
            # Keep running
            system.running = True
            try:
                while system.running:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
                
        elif args.mode == "train":
            logger.info("Starting in training mode")
            system.run_training_demo(ticker=args.ticker)
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"System error: {str(e)}")
        sys.exit(1)
    finally:
        system.shutdown()

if __name__ == "__main__":
    main()
