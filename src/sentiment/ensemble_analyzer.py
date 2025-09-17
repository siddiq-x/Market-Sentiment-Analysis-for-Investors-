"""
Ensemble sentiment analyzer combining FinBERT and lexicon-based approaches
"""
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime
import numpy as np

from .finbert_analyzer import FinBERTAnalyzer, SentimentResult
from .lexicon_analyzer import FinancialLexiconAnalyzer, LexiconSentimentResult

@dataclass
class EnsembleSentimentResult:
    """Container for ensemble sentiment analysis results"""
    text: str
    sentiment: str
    confidence: float
    ensemble_score: float  # -1 to 1 scale
    finbert_result: SentimentResult
    lexicon_result: LexiconSentimentResult
    weights: Dict[str, float]
    timestamp: datetime
    metadata: Dict[str, Any]

class EnsembleSentimentAnalyzer:
    """Ensemble analyzer combining multiple sentiment analysis approaches"""
    
    def __init__(self, 
                 finbert_weight: float = 0.7,
                 lexicon_weight: float = 0.3,
                 enable_finbert: bool = True,
                 enable_lexicon: bool = True):
        """
        Initialize ensemble analyzer
        
        Args:
            finbert_weight: Weight for FinBERT results (0-1)
            lexicon_weight: Weight for lexicon results (0-1)
            enable_finbert: Whether to use FinBERT
            enable_lexicon: Whether to use lexicon analyzer
        """
        self.logger = logging.getLogger("ensemble_analyzer")
        
        # Normalize weights
        total_weight = finbert_weight + lexicon_weight
        if total_weight > 0:
            self.finbert_weight = finbert_weight / total_weight
            self.lexicon_weight = lexicon_weight / total_weight
        else:
            self.finbert_weight = 0.5
            self.lexicon_weight = 0.5
        
        self.enable_finbert = enable_finbert
        self.enable_lexicon = enable_lexicon
        
        # Initialize analyzers
        self.finbert_analyzer = None
        self.lexicon_analyzer = None
        
        if self.enable_finbert:
            try:
                self.finbert_analyzer = FinBERTAnalyzer()
                self.logger.info("FinBERT analyzer initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize FinBERT: {str(e)}")
                self.enable_finbert = False
        
        if self.enable_lexicon:
            try:
                self.lexicon_analyzer = FinancialLexiconAnalyzer()
                self.logger.info("Lexicon analyzer initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize lexicon analyzer: {str(e)}")
                self.enable_lexicon = False
        
        # Recalculate weights if some analyzers failed
        if not self.enable_finbert:
            self.finbert_weight = 0.0
            self.lexicon_weight = 1.0
        elif not self.enable_lexicon:
            self.finbert_weight = 1.0
            self.lexicon_weight = 0.0
        
        self.logger.info(f"Ensemble weights - FinBERT: {self.finbert_weight:.2f}, Lexicon: {self.lexicon_weight:.2f}")
    
    def analyze_sentiment(self, text: str, context: Optional[Dict[str, Any]] = None) -> EnsembleSentimentResult:
        """
        Analyze sentiment using ensemble approach
        
        Args:
            text: Text to analyze
            context: Additional context information
        """
        if not text or not text.strip():
            return self._create_empty_result(text)
        
        # Get results from individual analyzers
        finbert_result = None
        lexicon_result = None
        
        if self.enable_finbert and self.finbert_analyzer:
            try:
                finbert_result = self.finbert_analyzer.analyze_sentiment(text, context)
            except Exception as e:
                self.logger.error(f"FinBERT analysis failed: {str(e)}")
        
        if self.enable_lexicon and self.lexicon_analyzer:
            try:
                lexicon_result = self.lexicon_analyzer.analyze_sentiment(text, context)
            except Exception as e:
                self.logger.error(f"Lexicon analysis failed: {str(e)}")
        
        # Combine results
        ensemble_sentiment, ensemble_confidence, ensemble_score = self._combine_results(
            finbert_result, lexicon_result
        )
        
        # Create metadata
        metadata = {
            "finbert_enabled": self.enable_finbert,
            "lexicon_enabled": self.enable_lexicon,
            "finbert_available": finbert_result is not None,
            "lexicon_available": lexicon_result is not None,
            "context": context or {}
        }
        
        weights = {
            "finbert": self.finbert_weight,
            "lexicon": self.lexicon_weight
        }
        
        return EnsembleSentimentResult(
            text=text,
            sentiment=ensemble_sentiment,
            confidence=ensemble_confidence,
            ensemble_score=ensemble_score,
            finbert_result=finbert_result,
            lexicon_result=lexicon_result,
            weights=weights,
            timestamp=datetime.now(),
            metadata=metadata
        )
    
    def _create_empty_result(self, text: str) -> EnsembleSentimentResult:
        """Create empty result for invalid input"""
        return EnsembleSentimentResult(
            text=text,
            sentiment="neutral",
            confidence=0.0,
            ensemble_score=0.0,
            finbert_result=None,
            lexicon_result=None,
            weights={"finbert": self.finbert_weight, "lexicon": self.lexicon_weight},
            timestamp=datetime.now(),
            metadata={"error": "Empty or invalid text"}
        )
    
    def _combine_results(self, 
                        finbert_result: Optional[SentimentResult],
                        lexicon_result: Optional[LexiconSentimentResult]) -> Tuple[str, float, float]:
        """Combine results from different analyzers"""
        
        # Convert sentiment to numerical scores (-1 to 1)
        finbert_score = 0.0
        lexicon_score = 0.0
        finbert_confidence = 0.0
        lexicon_confidence = 0.0
        
        if finbert_result:
            finbert_score = self._sentiment_to_score(finbert_result.sentiment, finbert_result.scores)
            finbert_confidence = finbert_result.confidence
        
        if lexicon_result:
            lexicon_score = self._sentiment_to_score(lexicon_result.sentiment)
            lexicon_confidence = lexicon_result.confidence
        
        # Calculate weighted ensemble score
        if finbert_result and lexicon_result:
            # Both analyzers available
            ensemble_score = (finbert_score * self.finbert_weight + 
                            lexicon_score * self.lexicon_weight)
            ensemble_confidence = (finbert_confidence * self.finbert_weight + 
                                 lexicon_confidence * self.lexicon_weight)
        elif finbert_result:
            # Only FinBERT available
            ensemble_score = finbert_score
            ensemble_confidence = finbert_confidence
        elif lexicon_result:
            # Only lexicon available
            ensemble_score = lexicon_score
            ensemble_confidence = lexicon_confidence
        else:
            # No analyzers available
            ensemble_score = 0.0
            ensemble_confidence = 0.0
        
        # Convert back to sentiment category
        ensemble_sentiment = self._score_to_sentiment(ensemble_score)
        
        # Adjust confidence based on agreement between methods
        if finbert_result and lexicon_result:
            agreement = self._calculate_agreement(finbert_result.sentiment, lexicon_result.sentiment)
            ensemble_confidence *= agreement
        
        return ensemble_sentiment, ensemble_confidence, ensemble_score
    
    def _sentiment_to_score(self, sentiment: str, scores: Optional[Dict[str, float]] = None) -> float:
        """Convert sentiment category to numerical score (-1 to 1)"""
        if scores:
            # Use raw scores if available (more nuanced)
            positive = scores.get("positive", 0.0)
            negative = scores.get("negative", 0.0)
            return positive - negative
        else:
            # Simple mapping
            if sentiment == "positive":
                return 1.0
            elif sentiment == "negative":
                return -1.0
            else:
                return 0.0
    
    def _score_to_sentiment(self, score: float, threshold: float = 0.1) -> str:
        """Convert numerical score to sentiment category"""
        if score > threshold:
            return "positive"
        elif score < -threshold:
            return "negative"
        else:
            return "neutral"
    
    def _calculate_agreement(self, sentiment1: str, sentiment2: str) -> float:
        """Calculate agreement factor between two sentiment results"""
        if sentiment1 == sentiment2:
            return 1.0  # Perfect agreement
        elif (sentiment1 == "neutral" and sentiment2 != "neutral") or \
             (sentiment1 != "neutral" and sentiment2 == "neutral"):
            return 0.8  # Partial agreement (one neutral)
        else:
            return 0.6  # Disagreement (positive vs negative)
    
    def batch_analyze(self, 
                     texts: List[str], 
                     contexts: Optional[List[Dict[str, Any]]] = None) -> List[EnsembleSentimentResult]:
        """Analyze sentiment for multiple texts"""
        results = []
        contexts = contexts or [None] * len(texts)
        
        for i, text in enumerate(texts):
            context = contexts[i] if i < len(contexts) else None
            result = self.analyze_sentiment(text, context)
            results.append(result)
        
        return results
    
    def get_ensemble_summary(self, results: List[EnsembleSentimentResult]) -> Dict[str, Any]:
        """Get summary statistics of ensemble analysis results"""
        if not results:
            return {}
        
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        total_confidence = 0.0
        total_ensemble_score = 0.0
        
        finbert_available = 0
        lexicon_available = 0
        both_available = 0
        
        for result in results:
            sentiment_counts[result.sentiment] += 1
            total_confidence += result.confidence
            total_ensemble_score += result.ensemble_score
            
            if result.finbert_result:
                finbert_available += 1
            if result.lexicon_result:
                lexicon_available += 1
            if result.finbert_result and result.lexicon_result:
                both_available += 1
        
        total_results = len(results)
        
        return {
            "total_analyzed": total_results,
            "sentiment_distribution": {
                k: {"count": v, "percentage": (v/total_results)*100} 
                for k, v in sentiment_counts.items()
            },
            "average_confidence": total_confidence / total_results,
            "average_ensemble_score": total_ensemble_score / total_results,
            "dominant_sentiment": max(sentiment_counts.keys(), key=lambda k: sentiment_counts[k]),
            "analyzer_availability": {
                "finbert_available": finbert_available,
                "lexicon_available": lexicon_available,
                "both_available": both_available,
                "finbert_coverage": (finbert_available / total_results) * 100,
                "lexicon_coverage": (lexicon_available / total_results) * 100,
                "ensemble_coverage": (both_available / total_results) * 100
            },
            "weights": {
                "finbert": self.finbert_weight,
                "lexicon": self.lexicon_weight
            }
        }
    
    def update_weights(self, finbert_weight: float, lexicon_weight: float):
        """Update ensemble weights"""
        total_weight = finbert_weight + lexicon_weight
        if total_weight > 0:
            self.finbert_weight = finbert_weight / total_weight
            self.lexicon_weight = lexicon_weight / total_weight
            self.logger.info(f"Updated weights - FinBERT: {self.finbert_weight:.2f}, Lexicon: {self.lexicon_weight:.2f}")
        else:
            self.logger.warning("Invalid weights provided, keeping current weights")
    
    def get_analyzer_status(self) -> Dict[str, Any]:
        """Get status of individual analyzers"""
        return {
            "finbert": {
                "enabled": self.enable_finbert,
                "available": self.finbert_analyzer is not None,
                "weight": self.finbert_weight
            },
            "lexicon": {
                "enabled": self.enable_lexicon,
                "available": self.lexicon_analyzer is not None,
                "weight": self.lexicon_weight
            }
        }
