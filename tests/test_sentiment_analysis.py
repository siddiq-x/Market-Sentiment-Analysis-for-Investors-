# Add src to path before any imports
import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'src')))

"""
Test suite for sentiment analysis components.

This module contains unit tests for the sentiment analysis functionality,
including tests for the FinBERT analyzer, Lexicon analyzer, and the
ensemble analyzer that combines multiple sentiment analysis approaches.

The tests cover:
- Basic functionality of each analyzer
- Edge cases and error handling
- Integration between different components
- Performance characteristics
"""
import unittest  # noqa: E402
from unittest.mock import Mock, patch  # noqa: E402

from sentiment.ensemble_analyzer import EnsembleSentimentAnalyzer  # noqa: E402
from sentiment.finbert_analyzer import FinBERTAnalyzer  # noqa: E402
from sentiment.lexicon_analyzer import LexiconAnalyzer  # noqa: E402


class TestSentimentAnalysis(unittest.TestCase):
    """
    Test suite for core sentiment analysis functionality.

    This class contains tests for the basic functionality of the sentiment
    analysis components, including both individual analyzers and their
    integration in the ensemble analyzer.
    """

    def setUp(self):
        """Set up test fixtures"""
        self.sample_texts = [
            "Apple stock is performing exceptionally well this quarter",
            "Tesla faces significant challenges in the current market",
            "The market remains neutral on Microsoft's latest earnings",
            "NVIDIA shows strong growth potential",
            "Economic uncertainty affects all major indices"
        ]

    @patch('sentiment.finbert_analyzer.AutoTokenizer')
    @patch('sentiment.finbert_analyzer.AutoModelForSequenceClassification')
    def test_finbert_analyzer_initialization(self, mock_model, mock_tokenizer):
        """
        Test FinBERT analyzer initialization and model loading.

        Verifies that the FinBERT analyzer correctly initializes and loads
        the required pre-trained models and tokenizers.
        """
        # Mock the model and tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.return_value = {
            'input_ids': [1, 2, 3],
            'attention_mask': [1, 1, 1]
        }
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = Mock()
        mock_model_instance.return_value = {
            'logits': [[0.1, 0.2, 0.7]]  # Mock logits for 3 classes
        }
        mock_model.from_pretrained.return_value = mock_model_instance

        # Test initialization
        analyzer = FinBERTAnalyzer()

        # Verify model and tokenizer were loaded
        self.assertIsNotNone(analyzer)
        mock_tokenizer.from_pretrained.assert_called_once()
        mock_model.from_pretrained.assert_called_once()

        # Test model prediction
        result = analyzer.analyze("Test text")
        self.assertIn('label', result)
        self.assertIn('score', result)
        self.assertIn(result['label'], ['positive', 'negative', 'neutral'])

    def test_lexicon_analyzer_initialization(self):
        """
        Test lexicon analyzer initialization and basic properties.

        Verifies that the lexicon analyzer is properly initialized with
        the required word lists and can be instantiated correctly.
        """
        analyzer = LexiconAnalyzer()

        self.assertIsNotNone(analyzer)
        self.assertIsInstance(analyzer.positive_words, set)
        self.assertIsInstance(analyzer.negative_words, set)
        self.assertGreater(len(analyzer.positive_words), 0)
        self.assertGreater(len(analyzer.negative_words), 0)

        # Test that word lists are properly loaded
        self.assertIn('bullish', analyzer.positive_words)
        self.assertIn('bearish', analyzer.negative_words)
        self.assertGreater(len(analyzer.negative_words), 0)

    def test_lexicon_analyzer_sentiment_scoring(self):
        """Test lexicon-based sentiment scoring"""
        analyzer = LexiconAnalyzer()

        # Test positive sentiment
        positive_text = "excellent profit growth bullish optimistic"
        result = analyzer.analyze_sentiment(positive_text)
        self.assertGreater(result.ensemble_score, 0)

        # Test negative sentiment
        negative_text = "terrible loss bearish pessimistic decline"
        result = analyzer.analyze_sentiment(negative_text)
        self.assertLess(result.ensemble_score, 0)

    def test_lexicon_analyzer_modifiers(self):
        """Test sentiment modifiers in lexicon analyzer"""
        analyzer = LexiconAnalyzer()

        # Test intensifier
        text_with_intensifier = "very excellent performance"
        result = analyzer.analyze_sentiment(text_with_intensifier)

        # Test negation
        text_with_negation = "not good performance"
        result_negated = analyzer.analyze_sentiment(text_with_negation)

        # Negated sentiment should be different from positive
        self.assertNotEqual(
            result.ensemble_score,
            result_negated.ensemble_score
        )

    @patch('sentiment.finbert_analyzer.FinBERTAnalyzer')
    @patch('sentiment.lexicon_analyzer.LexiconAnalyzer')
    def test_ensemble_analyzer_initialization(
            self, mock_lexicon, mock_finbert):
        """Test ensemble analyzer initialization"""
        # Mock the analyzers
        mock_finbert_instance = Mock()
        mock_lexicon_instance = Mock()
        mock_finbert.return_value = mock_finbert_instance
        mock_lexicon.return_value = mock_lexicon_instance

        analyzer = EnsembleSentimentAnalyzer()

        self.assertIsNotNone(analyzer)
        self.assertEqual(analyzer.finbert_weight, 0.7)
        self.assertEqual(analyzer.lexicon_weight, 0.3)

    @patch('sentiment.finbert_analyzer.FinBERTAnalyzer')
    @patch('sentiment.lexicon_analyzer.LexiconAnalyzer')
    def test_ensemble_analyzer_sentiment_combination(
            self, mock_lexicon, mock_finbert):
        """Test ensemble sentiment combination"""
        # Mock analyzer results
        mock_finbert_instance = Mock()
        mock_lexicon_instance = Mock()

        # Mock FinBERT result
        finbert_result = Mock()
        finbert_result.sentiment = 1
        finbert_result.confidence = 0.8
        finbert_result.ensemble_score = 0.6

        # Mock Lexicon result
        lexicon_result = Mock()
        lexicon_result.sentiment = 1
        lexicon_result.confidence = 0.7
        lexicon_result.ensemble_score = 0.4

        mock_finbert_instance.analyze_sentiment.return_value = finbert_result
        mock_lexicon_instance.analyze_sentiment.return_value = lexicon_result

        mock_finbert.return_value = mock_finbert_instance
        mock_lexicon.return_value = mock_lexicon_instance

        analyzer = EnsembleSentimentAnalyzer()
        result = analyzer.analyze_sentiment("test text")

        # Check that ensemble combines results
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, 'sentiment'))
        self.assertTrue(hasattr(result, 'confidence'))
        self.assertTrue(hasattr(result, 'ensemble_score'))

    def test_batch_processing(self):
        """Test batch processing capabilities"""
        analyzer = LexiconAnalyzer()

        results = analyzer.analyze_batch(self.sample_texts)

        self.assertEqual(len(results), len(self.sample_texts))
        for result in results:
            self.assertTrue(hasattr(result, 'sentiment'))
            self.assertTrue(hasattr(result, 'confidence'))
            self.assertIsInstance(result.sentiment, int)
            self.assertIsInstance(result.confidence, float)

    def test_context_awareness(self):
        """Test context-aware sentiment analysis"""
        analyzer = LexiconAnalyzer()

        # Test with financial context
        context = {"source": "financial_news", "ticker": "AAPL"}
        result = analyzer.analyze_sentiment("strong earnings", context=context)

        self.assertIsNotNone(result)
        # Context should influence the analysis
        self.assertGreater(result.confidence, 0)


class TestSentimentEdgeCases(unittest.TestCase):
    """
    Test edge cases and error handling in sentiment analysis.

    This class contains tests that verify the behavior of the sentiment
    analysis components when faced with unusual or unexpected inputs.
    """

    def setUp(self):
        self.analyzer = LexiconAnalyzer()

    def test_empty_text(self):
        """
        Test handling of empty text input.

        Verifies that the analyzer properly handles empty strings by returning
        a neutral sentiment with zero score.
        """
        analyzer = EnsembleSentimentAnalyzer()

        # Test empty string
        result = analyzer.analyze("")
        self.assertEqual(result['label'], 'neutral')
        self.assertEqual(result['score'], 0.0)

        # Test whitespace-only string
        result = analyzer.analyze("   \n\t")
        self.assertEqual(result['label'], 'neutral')
        self.assertEqual(result['score'], 0.0)

        # Test None input
        with self.assertRaises(ValueError):
            analyzer.analyze(None)

    def test_very_long_text(self):
        """Test handling of very long text"""
        long_text = "good " * 1000  # Very long text
        result = self.analyzer.analyze_sentiment(long_text)
        self.assertIsNotNone(result)

    def test_special_characters(self):
        """Test handling of special characters"""
        special_text = "$$$ AAPL 📈 🚀 #bullish @trader"
        result = self.analyzer.analyze_sentiment(special_text)
        self.assertIsNotNone(result)

    def test_mixed_sentiment(self):
        """Test handling of mixed sentiment"""
        mixed_text = "good news but bad execution"
        result = self.analyzer.analyze_sentiment(mixed_text)
        self.assertIsNotNone(result)
        # Should handle conflicting sentiments
        self.assertIsInstance(result.sentiment, int)

    def test_non_english_text(self):
        """Test handling of non-English text"""
        # This should gracefully handle non-English content
        non_english = "très bon marché financier"
        result = self.analyzer.analyze_sentiment(non_english)
        self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main()
