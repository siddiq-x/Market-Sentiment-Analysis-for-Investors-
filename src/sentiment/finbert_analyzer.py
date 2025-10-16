"""
FinBERT-based sentiment analysis for financial content
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import numpy as np
from datetime import datetime
import os

from config.config import config

@dataclass
class SentimentResult:
    """Container for sentiment analysis results"""
    text: str
    sentiment: str  # 'positive', 'negative', 'neutral'
    confidence: float
    scores: Dict[str, float]  # Raw scores for each class
    timestamp: datetime
    metadata: Dict[str, Any]

class FinBERTAnalyzer:
    """FinBERT-based sentiment analyzer for financial content"""

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.logger = logging.getLogger("finbert_analyzer")
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else
    "cpu")

        # Model and tokenizer will be loaded lazily
        self.model = None
        self.tokenizer = None
        self._model_loaded = False

        # Sentiment mappings
        self.label_mapping = {
            0: "negative",
            1: "neutral",
            2: "positive"
        }

        # Financial context adjustments
        self.financial_modifiers = {
            'strong_positive': ['rally', 'surge', 'soar', 'boom', 'bullish',
    'outperform'],
            'strong_negative': ['crash', 'plunge', 'collapse', 'bearish',
    'underperform', 'decline'],
            'uncertainty': ['volatile', 'uncertain', 'mixed', 'cautious',
    'wait-and-see']
        }

        self.logger.info(f"Initialized FinBERT analyzer with model:
    {model_name}")
        self.logger.info(f"Using device: {self.device}")

    def _load_model(self):
        """Load FinBERT model and tokenizer"""
        if self._model_loaded:
            return

        try:
            self.logger.info(f"Loading FinBERT model: {self.model_name}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(sel
    f.model_name)
            self.model.to(self.device)
            self.model.eval()

            self._model_loaded = True
            self.logger.info("FinBERT model loaded successfully")

        except Exception as e:
            self.logger.error(f"Error loading FinBERT model: {str(e)}")
            # Fallback to a simpler model
            try:
                self.logger.info("Attempting fallback to
    distilbert-base-uncased-finetuned-sst-2-english")
                self.model_name = "distilbert-base-uncased-finetuned-sst-2-engl
    ish"
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained
    (self.model_name)
                self.model.to(self.device)
                self.model.eval()

                # Adjust label mapping for different model
                self.label_mapping = {0: "negative", 1: "positive"}

                self._model_loaded = True
                self.logger.info("Fallback model loaded successfully")

            except Exception as fallback_error:
                self.logger.error(f"Fallback model loading failed:
    {str(fallback_error)}")
                raise RuntimeError("Failed to load any sentiment analysis
    model")

    def analyze_sentiment(self, text: str, context: Optional[Dict[str, Any]] =
    None) -> SentimentResult:
        """
        Analyze sentiment of financial text

        Args:
            text: Text to analyze
            context: Additional context (ticker, source, etc.)
        """
        if not text or not text.strip():
            return SentimentResult(
                text="",
                sentiment="neutral",
                confidence=0.0,
                scores={"negative": 0.33, "neutral": 0.34, "positive": 0.33},
                timestamp=datetime.now(),
                metadata={"error": "Empty text"}
            )

        # Load model if not already loaded
        if not self._model_loaded:
            self._load_model()

        # Preprocess text
        processed_text = self._preprocess_text(text)

        # Get raw sentiment scores
        raw_scores = self._get_raw_sentiment(processed_text)

        # Apply financial context adjustments
        adjusted_scores = self._apply_financial_context(processed_text,
    raw_scores, context)

        # Determine final sentiment and confidence
        sentiment, confidence = self._determine_sentiment(adjusted_scores)

        # Create metadata
        metadata = {
            "original_length": len(text),
            "processed_length": len(processed_text),
            "model_name": self.model_name,
            "device": str(self.device),
            "context": context or {}
        }

        return SentimentResult(
            text=text,
            sentiment=sentiment,
            confidence=confidence,
            scores=adjusted_scores,
            timestamp=datetime.now(),
            metadata=metadata
        )

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for FinBERT analysis"""
        # Basic cleaning while preserving financial context
        text = text.strip()

        # Limit length to model's max sequence length
        max_length = getattr(self.tokenizer, 'model_max_length', 512)
        if len(text) > max_length:
            # Try to keep the most important parts (beginning and end)
            half_length = (max_length - 50) // 2
            text = text[:half_length] + " ... " + text[-half_length:]

        return text

    def _get_raw_sentiment(self, text: str) -> Dict[str, float]:
        """Get raw sentiment scores from FinBERT"""
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

                # Apply softmax to get probabilities
                probabilities = torch.softmax(logits, dim=-1)
                scores = probabilities.cpu().numpy()[0]

            # Map to sentiment labels
            if len(scores) == 3:  # FinBERT format
                return {
                    "negative": float(scores[0]),
                    "neutral": float(scores[1]),
                    "positive": float(scores[2])
                }
            elif len(scores) == 2:  # Binary classification
                return {
                    "negative": float(scores[0]),
                    "positive": float(scores[1]),
                    "neutral": 0.0
                }
            else:
                raise ValueError(f"Unexpected number of classes:
    {len(scores)}")

        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {str(e)}")
            # Return neutral scores as fallback
            return {"negative": 0.33, "neutral": 0.34, "positive": 0.33}

    def _apply_financial_context(self,
                                text: str,
                                raw_scores: Dict[str, float],
                                context: Optional[Dict[str, Any]]) ->
    Dict[str, float]:
        """Apply financial context adjustments to raw scores"""
        adjusted_scores = raw_scores.copy()
        text_lower = text.lower()

        # Check for strong financial indicators
        strong_positive_count = sum(1 for word in
    self.financial_modifiers['strong_positive']
                                  if word in text_lower)
        strong_negative_count = sum(1 for word in
    self.financial_modifiers['strong_negative']
                                  if word in text_lower)
        uncertainty_count = sum(1 for word in
    self.financial_modifiers['uncertainty']
                              if word in text_lower)

        # Apply adjustments
        if strong_positive_count > 0:
            boost = min(0.2, strong_positive_count * 0.1)
            adjusted_scores["positive"] = min(1.0, adjusted_scores["positive"]
    + boost)
            adjusted_scores["negative"] = max(0.0, adjusted_scores["negative"]
    - boost/2)

        if strong_negative_count > 0:
            boost = min(0.2, strong_negative_count * 0.1)
            adjusted_scores["negative"] = min(1.0, adjusted_scores["negative"]
    + boost)
            adjusted_scores["positive"] = max(0.0, adjusted_scores["positive"]
    - boost/2)

        if uncertainty_count > 0:
            boost = min(0.15, uncertainty_count * 0.05)
            adjusted_scores["neutral"] = min(1.0, adjusted_scores["neutral"] +
    boost)
            adjusted_scores["positive"] = max(0.0, adjusted_scores["positive"]
    - boost/2)
            adjusted_scores["negative"] = max(0.0, adjusted_scores["negative"]
    - boost/2)

        # Context-based adjustments
        if context:
            source = context.get("source", "").lower()

            # Adjust based on source credibility
            if "reuters" in source or "bloomberg" in source:
                # High credibility sources - slight confidence boost
                max_score = max(adjusted_scores.values())
                for key in adjusted_scores:
                    if adjusted_scores[key] == max_score:
                        adjusted_scores[key] = min(1.0, adjusted_scores[key] +
    0.05)

            # Check for earnings-related content
            if any(term in text_lower for term in ["earnings", "quarterly",
    "revenue", "profit"]):
                # Earnings content tends to be more decisive
                max_key = max(adjusted_scores.keys(), key=lambda k:
    adjusted_scores[k])
                adjusted_scores[max_key] = min(1.0, adjusted_scores[max_key] +
    0.1)

        # Normalize scores to sum to 1
        total = sum(adjusted_scores.values())
        if total > 0:
            adjusted_scores = {k: v/total for k, v in adjusted_scores.items()}

        return adjusted_scores

    def _determine_sentiment(self, scores: Dict[str, float]) -> Tuple[str,
    float]:
        """Determine final sentiment and confidence from scores"""
        # Find the sentiment with highest score
        max_sentiment = max(scores.keys(), key=lambda k: scores[k])
        max_score = scores[max_sentiment]

        # Calculate confidence based on margin over second-highest
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) >= 2:
            confidence = max_score - sorted_scores[1]
        else:
            confidence = max_score

        # Apply threshold for neutral classification
        threshold = config.model.sentiment_threshold

        if max_sentiment != "neutral" and max_score < 0.5 + threshold:
            # If the winning sentiment is not strong enough, classify as neutral
            return "neutral", confidence

        return max_sentiment, confidence

    def batch_analyze(self,
                     texts: List[str],
                     contexts: Optional[List[Dict[str, Any]]] = None) ->
    List[SentimentResult]:
        """Analyze sentiment for multiple texts"""
        results = []
        contexts = contexts or [None] * len(texts)

        for i, text in enumerate(texts):
            context = contexts[i] if i < len(contexts) else None
            result = self.analyze_sentiment(text, context)
            results.append(result)

        return results

    def get_sentiment_summary(self, results: List[SentimentResult]) ->
    Dict[str, Any]:
        """Get summary statistics of sentiment analysis results"""
        if not results:
            return {}

        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        total_confidence = 0.0

        for result in results:
            sentiment_counts[result.sentiment] += 1
            total_confidence += result.confidence

        total_results = len(results)

        return {
            "total_analyzed": total_results,
            "sentiment_distribution": {
                k: {"count": v, "percentage": (v/total_results)*100}
                for k, v in sentiment_counts.items()
            },
            "average_confidence": total_confidence / total_results,
            "dominant_sentiment": max(sentiment_counts.keys(), key=lambda k:
    sentiment_counts[k]),
            "sentiment_score": (
                sentiment_counts["positive"] - sentiment_counts["negative"]
            ) / total_results  # Overall sentiment score (-1 to 1)
        }

    def save_model_cache(self, cache_dir: str):
        """Save model to local cache"""
        if not self._model_loaded:
            self._load_model()

        try:
            os.makedirs(cache_dir, exist_ok=True)
            self.model.save_pretrained(cache_dir)
            self.tokenizer.save_pretrained(cache_dir)
            self.logger.info(f"Model cached to {cache_dir}")
        except Exception as e:
            self.logger.error(f"Error caching model: {str(e)}")

    def load_from_cache(self, cache_dir: str) -> bool:
        """Load model from local cache"""
        try:
            if os.path.exists(cache_dir):
                self.tokenizer = AutoTokenizer.from_pretrained(cache_dir)
                self.model = AutoModelForSequenceClassification.from_pretrained
    (cache_dir)
                self.model.to(self.device)
                self.model.eval()
                self._model_loaded = True
                self.logger.info(f"Model loaded from cache: {cache_dir}")
                return True
        except Exception as e:
            self.logger.error(f"Error loading from cache: {str(e)}")

        return False
