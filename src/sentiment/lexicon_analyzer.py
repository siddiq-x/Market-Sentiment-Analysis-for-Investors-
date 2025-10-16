"""
Lexicon-based sentiment analysis for financial content
"""
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime
import re
import json
import os

@dataclass
class LexiconSentimentResult:
    """Container for lexicon-based sentiment results"""
    text: str
    sentiment: str
    confidence: float
    word_scores: Dict[str, float]
    matched_words: List[str]
    timestamp: datetime
    metadata: Dict[str, Any]

class FinancialLexiconAnalyzer:
    """Lexicon-based sentiment analyzer using financial dictionaries"""

    def __init__(self):
        self.logger = logging.getLogger("lexicon_analyzer")

        # Financial sentiment lexicons
        self.positive_words = {
            # Market movements
            'rally', 'surge', 'soar', 'boom', 'bullish', 'uptrend', 'breakout',
            'outperform', 'beat', 'exceed', 'strong', 'robust', 'solid',

            # Financial performance
            'profit', 'gain', 'growth', 'increase', 'rise', 'improve',
    'expand',
            'revenue', 'earnings', 'dividend', 'buyback', 'acquisition',

            # Analyst terms
            'upgrade', 'buy', 'overweight', 'outperform', 'positive',
    'bullish',
            'recommend', 'target', 'upside', 'opportunity',

            # General positive
            'excellent', 'outstanding', 'impressive', 'successful',
    'promising',
            'confident', 'optimistic', 'favorable', 'attractive'
        }

        self.negative_words = {
            # Market movements
            'crash', 'plunge', 'collapse', 'decline', 'fall', 'drop',
    'bearish',
            'downtrend', 'correction', 'selloff', 'underperform', 'miss',

            # Financial performance
            'loss', 'deficit', 'decrease', 'shrink', 'contract', 'weak',
    'poor',
            'disappointing', 'concern', 'risk', 'debt', 'bankruptcy',

            # Analyst terms
            'downgrade', 'sell', 'underweight', 'underperform', 'negative',
            'bearish', 'avoid', 'caution', 'warning', 'alert',

            # General negative
            'terrible', 'awful', 'disappointing', 'concerning', 'worrying',
            'uncertain', 'volatile', 'risky', 'problematic'
        }

        self.neutral_words = {
            'stable', 'steady', 'maintain', 'hold', 'neutral', 'unchanged',
            'flat', 'sideways', 'consolidate', 'range', 'mixed', 'balanced'
        }

        # Intensity modifiers
        self.intensifiers = {
            'very': 1.5, 'extremely': 2.0, 'highly': 1.8, 'significantly': 1.7,
            'substantially': 1.6, 'considerably': 1.5, 'remarkably': 1.8,
            'exceptionally': 2.0, 'tremendously': 1.9, 'dramatically': 1.8
        }

        self.diminishers = {
            'slightly': 0.5, 'somewhat': 0.6, 'rather': 0.7, 'fairly': 0.8,
            'moderately': 0.7, 'relatively': 0.8, 'marginally': 0.4,
            'barely': 0.3, 'hardly': 0.2, 'scarcely': 0.3
        }

        # Negation words
        self.negation_words = {
            'not', 'no', 'never', 'none', 'nothing', 'nowhere', 'neither',
            'nobody', 'cannot', 'cant', 'wont', 'wouldnt', 'shouldnt',
            'couldnt', 'doesnt', 'dont', 'didnt', 'isnt', 'arent', 'wasnt',
    'werent'
        }

        # Load additional lexicons if available
        self._load_additional_lexicons()

    def _load_additional_lexicons(self):
        """Load additional sentiment lexicons from files"""
        lexicon_dir = os.path.join(os.path.dirname(__file__), '..', '..',
    'data', 'lexicons')

        # Try to load custom financial lexicons
        try:
            if os.path.exists(lexicon_dir):
                for filename in os.listdir(lexicon_dir):
                    if filename.endswith('.json'):
                        filepath = os.path.join(lexicon_dir, filename)
                        with open(filepath, 'r') as f:
                            lexicon_data = json.load(f)

                        if 'positive' in lexicon_data:
                            self.positive_words.update(lexicon_data['positive']
    )
                        if 'negative' in lexicon_data:
                            self.negative_words.update(lexicon_data['negative']
    )
                        if 'neutral' in lexicon_data:
                            self.neutral_words.update(lexicon_data['neutral'])

                        self.logger.info(f"Loaded lexicon from {filename}")
        except Exception as e:
            self.logger.warning(f"Could not load additional lexicons:
    {str(e)}")

    def analyze_sentiment(self, text: str, context: Optional[Dict[str, Any]] =
    None) -> LexiconSentimentResult:
        """
        Analyze sentiment using financial lexicons

        Args:
            text: Text to analyze
            context: Additional context information
        """
        if not text or not text.strip():
            return LexiconSentimentResult(
                text="",
                sentiment="neutral",
                confidence=0.0,
                word_scores={},
                matched_words=[],
                timestamp=datetime.now(),
                metadata={"error": "Empty text"}
            )

        # Preprocess text
        words = self._preprocess_text(text)

        # Calculate sentiment scores
        word_scores, matched_words = self._calculate_word_scores(words)

        # Apply context and modifiers
        adjusted_scores = self._apply_modifiers(words, word_scores)

        # Determine final sentiment
        sentiment, confidence = self._determine_sentiment(adjusted_scores)

        # Create metadata
        metadata = {
            "word_count": len(words),
            "positive_words": len([w for w in matched_words if w in
    self.positive_words]),
            "negative_words": len([w for w in matched_words if w in
    self.negative_words]),
            "neutral_words": len([w for w in matched_words if w in
    self.neutral_words]),
            "context": context or {}
        }

        return LexiconSentimentResult(
            text=text,
            sentiment=sentiment,
            confidence=confidence,
            word_scores=adjusted_scores,
            matched_words=matched_words,
            timestamp=datetime.now(),
            metadata=metadata
        )

    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text and return list of words"""
        # Convert to lowercase and remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text.lower())

        # Split into words
        words = text.split()

        # Remove empty strings
        words = [word for word in words if word.strip()]

        return words

    def _calculate_word_scores(self, words: List[str]) -> Tuple[Dict[str,
    float], List[str]]:
        """Calculate sentiment scores for individual words"""
        word_scores = {}
        matched_words = []

        for word in words:
            score = 0.0

            if word in self.positive_words:
                score = 1.0
                matched_words.append(word)
            elif word in self.negative_words:
                score = -1.0
                matched_words.append(word)
            elif word in self.neutral_words:
                score = 0.0
                matched_words.append(word)

            if score != 0.0:
                word_scores[word] = score

        return word_scores, matched_words

    def _apply_modifiers(self, words: List[str], word_scores: Dict[str,
    float]) -> Dict[str, float]:
        """Apply intensity modifiers and negation"""
        adjusted_scores = word_scores.copy()

        for i, word in enumerate(words):
            if word in adjusted_scores:
                current_score = adjusted_scores[word]

                # Check for intensifiers/diminishers in previous 2 words
                for j in range(max(0, i-2), i):
                    prev_word = words[j]

                    if prev_word in self.intensifiers:
                        adjusted_scores[word] = current_score *
    self.intensifiers[prev_word]
                        break
                    elif prev_word in self.diminishers:
                        adjusted_scores[word] = current_score *
    self.diminishers[prev_word]
                        break

                # Check for negation in previous 3 words
                for j in range(max(0, i-3), i):
                    if words[j] in self.negation_words:
                        adjusted_scores[word] = -current_score
                        break

                # Ensure scores stay within bounds
                adjusted_scores[word] = max(-2.0, min(2.0,
    adjusted_scores[word]))

        return adjusted_scores

    def _determine_sentiment(self, word_scores: Dict[str, float]) ->
    Tuple[str, float]:
        """Determine overall sentiment from word scores"""
        if not word_scores:
            return "neutral", 0.0

        # Calculate aggregate scores
        positive_score = sum(score for score in word_scores.values() if score
    > 0)
        negative_score = abs(sum(score for score in word_scores.values() if
    score < 0))
        total_words = len(word_scores)

        # Calculate net sentiment
        net_sentiment = positive_score - negative_score

        # Normalize by number of sentiment words
        if total_words > 0:
            normalized_sentiment = net_sentiment / total_words
        else:
            normalized_sentiment = 0.0

        # Determine sentiment category
        if normalized_sentiment > 0.1:
            sentiment = "positive"
            confidence = min(1.0, abs(normalized_sentiment))
        elif normalized_sentiment < -0.1:
            sentiment = "negative"
            confidence = min(1.0, abs(normalized_sentiment))
        else:
            sentiment = "neutral"
            confidence = 1.0 - abs(normalized_sentiment)

        return sentiment, confidence

    def batch_analyze(self,
                     texts: List[str],
                     contexts: Optional[List[Dict[str, Any]]] = None) ->
    List[LexiconSentimentResult]:
        """Analyze sentiment for multiple texts"""
        results = []
        contexts = contexts or [None] * len(texts)

        for i, text in enumerate(texts):
            context = contexts[i] if i < len(contexts) else None
            result = self.analyze_sentiment(text, context)
            results.append(result)

        return results

    def get_lexicon_coverage(self, text: str) -> Dict[str, Any]:
        """Get statistics about lexicon coverage in text"""
        words = self._preprocess_text(text)
        total_words = len(words)

        if total_words == 0:
            return {"coverage": 0.0, "matched_words": 0, "total_words": 0}

        matched_count = 0
        sentiment_distribution = {"positive": 0, "negative": 0, "neutral": 0}

        for word in words:
            if word in self.positive_words:
                matched_count += 1
                sentiment_distribution["positive"] += 1
            elif word in self.negative_words:
                matched_count += 1
                sentiment_distribution["negative"] += 1
            elif word in self.neutral_words:
                matched_count += 1
                sentiment_distribution["neutral"] += 1

        coverage = matched_count / total_words

        return {
            "coverage": coverage,
            "matched_words": matched_count,
            "total_words": total_words,
            "sentiment_distribution": sentiment_distribution
        }

    def add_custom_words(self,
                        positive: Optional[List[str]] = None,
                        negative: Optional[List[str]] = None,
                        neutral: Optional[List[str]] = None):
        """Add custom words to lexicons"""
        if positive:
            self.positive_words.update(positive)
            self.logger.info(f"Added {len(positive)} positive words")

        if negative:
            self.negative_words.update(negative)
            self.logger.info(f"Added {len(negative)} negative words")

        if neutral:
            self.neutral_words.update(neutral)
            self.logger.info(f"Added {len(neutral)} neutral words")

    def save_lexicons(self, filepath: str):
        """Save current lexicons to file"""
        lexicon_data = {
            "positive": list(self.positive_words),
            "negative": list(self.negative_words),
            "neutral": list(self.neutral_words),
            "intensifiers": self.intensifiers,
            "diminishers": self.diminishers,
            "negation_words": list(self.negation_words)
        }

        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(lexicon_data, f, indent=2)
            self.logger.info(f"Lexicons saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving lexicons: {str(e)}")

    def get_lexicon_stats(self) -> Dict[str, int]:
        """Get statistics about loaded lexicons"""
        return {
            "positive_words": len(self.positive_words),
            "negative_words": len(self.negative_words),
            "neutral_words": len(self.neutral_words),
            "intensifiers": len(self.intensifiers),
            "diminishers": len(self.diminishers),
            "negation_words": len(self.negation_words),
            "total_sentiment_words": len(self.positive_words) +
    len(self.negative_words) + len(self.neutral_words)
        }
