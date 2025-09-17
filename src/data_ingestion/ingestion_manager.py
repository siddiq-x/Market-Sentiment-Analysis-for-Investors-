"""
Data Ingestion Manager - Orchestrates all data connectors
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

from .base_connector import BaseConnector, DataPoint
from .news_connector import NewsConnector
from .market_connector import MarketConnector
from .social_connector import SocialConnector, RedditConnector
from config.config import config

class IngestionManager:
    """Manages all data ingestion connectors"""
    
    def __init__(self):
        self.logger = logging.getLogger("ingestion_manager")
        self.connectors: Dict[str, BaseConnector] = {}
        self._initialize_connectors()
        
    def _initialize_connectors(self):
        """Initialize all available connectors"""
        try:
            self.connectors['news'] = NewsConnector()
            self.connectors['market'] = MarketConnector()
            self.connectors['twitter'] = SocialConnector()
            self.connectors['reddit'] = RedditConnector()
            
            self.logger.info(f"Initialized {len(self.connectors)} connectors")
            
        except Exception as e:
            self.logger.error(f"Error initializing connectors: {str(e)}")
    
    def connect_all(self) -> Dict[str, bool]:
        """Connect all connectors and return status"""
        connection_status = {}
        
        for name, connector in self.connectors.items():
            try:
                status = connector.connect()
                connection_status[name] = status
                self.logger.info(f"Connector {name}: {'Connected' if status else 'Failed'}")
            except Exception as e:
                connection_status[name] = False
                self.logger.error(f"Error connecting {name}: {str(e)}")
        
        return connection_status
    
    def fetch_all_data(self, 
                      tickers: Optional[List[str]] = None,
                      hours_back: int = 24,
                      max_workers: int = 4) -> List[DataPoint]:
        """Fetch data from all connectors in parallel"""
        tickers = tickers or config.monitored_tickers
        all_data = []
        
        # Define fetch tasks for each connector
        fetch_tasks = [
            ('news', lambda: self.connectors['news'].fetch_data(
                tickers=tickers, hours_back=hours_back)),
            ('market', lambda: self.connectors['market'].fetch_data(
                tickers=tickers, period="1d", interval="1h")),
            ('twitter', lambda: self.connectors['twitter'].fetch_data(
                tickers=tickers, hours_back=hours_back, max_tweets=100)),
            ('reddit', lambda: self.connectors['reddit'].fetch_data(
                max_posts=50))
        ]
        
        # Execute tasks in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_connector = {
                executor.submit(task): name 
                for name, task in fetch_tasks 
                if name in self.connectors and self.connectors[name].is_healthy()
            }
            
            for future in as_completed(future_to_connector):
                connector_name = future_to_connector[future]
                try:
                    data = future.result(timeout=60)  # 60 second timeout
                    all_data.extend(data)
                    self.logger.info(f"Fetched {len(data)} items from {connector_name}")
                except Exception as e:
                    self.logger.error(f"Error fetching from {connector_name}: {str(e)}")
        
        # Sort by timestamp (newest first)
        all_data.sort(key=lambda x: x.timestamp, reverse=True)
        
        self.logger.info(f"Total data points collected: {len(all_data)}")
        return all_data
    
    def fetch_by_connector(self, connector_name: str, **kwargs) -> List[DataPoint]:
        """Fetch data from a specific connector"""
        if connector_name not in self.connectors:
            raise ValueError(f"Connector {connector_name} not found")
        
        connector = self.connectors[connector_name]
        if not connector.is_healthy():
            self.logger.warning(f"Connector {connector_name} is not healthy")
            return []
        
        return connector.fetch_data(**kwargs)
    
    def get_connector_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all connectors"""
        status = {}
        
        for name, connector in self.connectors.items():
            status[name] = {
                "connected": connector.is_healthy(),
                "name": connector.name,
                "config_keys": list(connector.config.keys())
            }
        
        return status
    
    def validate_configuration(self) -> Dict[str, List[str]]:
        """Validate configuration for all connectors"""
        validation_results = {}
        
        # Check main config
        missing_config = config.validate()
        if missing_config:
            validation_results['main_config'] = missing_config
        
        # Check connector-specific requirements
        connector_requirements = {
            'news': ['NEWS_API_KEY'],
            'twitter': ['TWITTER_BEARER_TOKEN'],
            'market': ['ALPHA_VANTAGE_API_KEY'],  # Optional for Yahoo Finance
            'reddit': []  # No API key required for public API
        }
        
        for connector_name, required_keys in connector_requirements.items():
            missing = []
            for key in required_keys:
                if not getattr(config.api, key.lower(), None):
                    missing.append(key)
            
            if missing:
                validation_results[connector_name] = missing
        
        return validation_results
    
    def export_data(self, data: List[DataPoint], format: str = "json") -> str:
        """Export data points to specified format"""
        if format.lower() == "json":
            export_data = []
            for dp in data:
                export_data.append({
                    "source": dp.source,
                    "timestamp": dp.timestamp.isoformat(),
                    "content": dp.content,
                    "ticker": dp.ticker,
                    "credibility_score": dp.credibility_score,
                    "metadata": dp.metadata
                })
            return json.dumps(export_data, indent=2)
        
        elif format.lower() == "csv":
            import pandas as pd
            df_data = []
            for dp in data:
                row = {
                    "source": dp.source,
                    "timestamp": dp.timestamp.isoformat(),
                    "content": dp.content,
                    "ticker": dp.ticker,
                    "credibility_score": dp.credibility_score
                }
                # Flatten metadata
                for key, value in dp.metadata.items():
                    row[f"metadata_{key}"] = value
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            return df.to_csv(index=False)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def cleanup(self):
        """Cleanup all connectors"""
        for name, connector in self.connectors.items():
            try:
                connector.disconnect()
                self.logger.info(f"Disconnected {name}")
            except Exception as e:
                self.logger.error(f"Error disconnecting {name}: {str(e)}")


# Singleton instance
ingestion_manager = IngestionManager()
