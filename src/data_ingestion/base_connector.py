"""
Base connector class for data ingestion
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from dataclasses import dataclass


@dataclass
class DataPoint:
    """Standard data point structure"""
    source: str
    timestamp: datetime
    content: str
    metadata: Dict[str, Any]
    ticker: Optional[str] = None
    credibility_score: float = 1.0


class BaseConnector(ABC):
    """Base class for all data connectors"""

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"connector.{name}")
        self._is_connected = False

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to data source"""

    @abstractmethod
    def fetch_data(self, **kwargs) -> List[DataPoint]:
        """Fetch data from the source"""

    @abstractmethod
    def is_healthy(self) -> bool:
        """Check if connection is healthy"""

    def disconnect(self):
        """Disconnect from data source"""
        self._is_connected = False
        self.logger.info(f"Disconnected from {self.name}")

    def get_credibility_score(self, source_info: Dict[str, Any]) -> float:
        """Calculate credibility score for a source"""
        # Base implementation - can be overridden
        domain_scores = {
            'reuters.com': 0.95,
            'bloomberg.com': 0.95,
            'wsj.com': 0.90,
            'cnbc.com': 0.85,
            'marketwatch.com': 0.80,
            'yahoo.com': 0.75,
            'twitter.com': 0.60,
            'reddit.com': 0.50
        }

        domain = source_info.get('domain', '').lower()
        return domain_scores.get(domain, 0.70)  # Default score
