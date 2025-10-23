"""
Text cleaning and preprocessing utilities
"""

import re
import string
from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/stopwords")
    nltk.data.find("corpora/wordnet")
    nltk.data.find("taggers/averaged_perceptron_tagger")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)


@dataclass
class CleanedText:
    """Container for cleaned text data"""

    original: str
    cleaned: str
    tokens: List[str]
    sentences: List[str]
    metadata: Dict[str, Any]


class TextCleaner:
    """Advanced text cleaning for financial content"""

    def __init__(self):
        self.logger = logging.getLogger("text_cleaner")
        try:
            self.lemmatizer = WordNetLemmatizer()
        except Exception as e:
            self.logger.warning(f"WordNet lemmatizer failed to initialize: {e}")
            self.lemmatizer = None

        # Financial stop words (in addition to standard ones)
        self.financial_stopwords = {
            "stock",
            "share",
            "market",
            "trading",
            "trader",
            "investor",
            "investment",
            "financial",
            "finance",
            "money",
            "dollar",
            "price",
            "company",
            "corp",
            "inc",
            "ltd",
            "llc",
        }

        # Standard English stop words
        try:
            self.stop_words = set(stopwords.words("english"))
        except Exception:
            self.stop_words = set()

        # Financial abbreviations and their expansions
        self.financial_abbreviations = {
            "ipo": "initial public offering",
            "ceo": "chief executive officer",
            "cfo": "chief financial officer",
            "eps": "earnings per share",
            "pe": "price to earnings",
            "pb": "price to book",
            "roe": "return on equity",
            "roa": "return on assets",
            "ebitda": "earnings before interest taxes depreciation amortization",
            "yoy": "year over year",
            "qoq": "quarter over quarter",
            "atm": "at the money",
            "otm": "out of the money",
            "itm": "in the money",
        }

        # Compile regex patterns
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile commonly used regex patterns"""
        # URLs
        self.url_pattern = re.compile(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        )

        # Email addresses
        self.email_pattern = re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
        )

        # Stock tickers ($AAPL, AAPL)
        self.ticker_pattern = re.compile(r"\$?([A-Z]{1,5})\b")

        # Numbers with currency symbols
        self.currency_pattern = re.compile(r"[\$£€¥₹]\s*[\d,]+\.?\d*[BMK]?")

        # Percentages
        self.percentage_pattern = re.compile(r"\d+\.?\d*\s*%")

        # Social media handles
        self.handle_pattern = re.compile(r"@[A-Za-z0-9_]+")

        # Hashtags
        self.hashtag_pattern = re.compile(r"#[A-Za-z0-9_]+")

        # Multiple whitespace
        self.whitespace_pattern = re.compile(r"\s+")

        # HTML tags
        self.html_pattern = re.compile(r"<[^>]+>")

    def clean_text(
        self,
        text: str,
        preserve_tickers: bool = True,
        preserve_numbers: bool = True,
        remove_stopwords: bool = False,
        lemmatize: bool = True,
    ) -> CleanedText:
        """
        Comprehensive text cleaning for financial content

        Args:
            text: Input text to clean
            preserve_tickers: Keep stock tickers in text
            preserve_numbers: Keep numerical values
            remove_stopwords: Remove stop words
            lemmatize: Apply lemmatization
        """
        if not text or not isinstance(text, str):
            return CleanedText("", "", [], [], {})

        original_text = text
        metadata = {
            "original_length": len(text),
            "preserve_tickers": preserve_tickers,
            "preserve_numbers": preserve_numbers,
            "remove_stopwords": remove_stopwords,
            "lemmatize": lemmatize,
        }

        # Step 1: Extract important entities before cleaning
        tickers = self._extract_tickers(text) if preserve_tickers else []
        currencies = self._extract_currencies(text) if preserve_numbers else []
        percentages = self._extract_percentages(text) if preserve_numbers else []

        # Step 2: Remove HTML tags
        text = self.html_pattern.sub(" ", text)

        # Step 3: Remove URLs (but keep domain info in metadata)
        urls = self.url_pattern.findall(text)
        text = self.url_pattern.sub(" [URL] ", text)

        # Step 4: Remove email addresses
        emails = self.email_pattern.findall(text)
        text = self.email_pattern.sub(" [EMAIL] ", text)

        # Step 5: Handle social media elements
        handles = self.handle_pattern.findall(text)
        hashtags = self.hashtag_pattern.findall(text)
        text = self.handle_pattern.sub(" ", text)
        text = self.hashtag_pattern.sub(" ", text)

        # Step 6: Expand financial abbreviations
        text = self._expand_abbreviations(text)

        # Step 7: Normalize whitespace and punctuation
        text = self.whitespace_pattern.sub(" ", text)
        text = text.strip()

        # Step 8: Convert to lowercase (but preserve ticker case)
        if preserve_tickers and tickers:
            # Temporarily replace tickers with placeholders
            ticker_placeholders = {}
            for i, ticker in enumerate(tickers):
                placeholder = f"TICKER{i}PLACEHOLDER"
                ticker_placeholders[placeholder] = ticker
                text = re.sub(
                    rf"\b{re.escape(ticker)}\b", placeholder, text, flags=re.IGNORECASE
                )

        text = text.lower()

        # Restore tickers
        if preserve_tickers and tickers:
            for placeholder, ticker in ticker_placeholders.items():
                text = text.replace(placeholder.lower(), ticker)

        # Step 9: Remove extra punctuation (but keep sentence structure)
        text = re.sub(r"[^\w\s\.\!\?\$\%\-]", " ", text)
        text = self.whitespace_pattern.sub(" ", text).strip()

        # Step 10: Tokenization
        try:
            tokens = word_tokenize(text)
            sentences = sent_tokenize(text)
        except Exception:
            tokens = text.split()
            sentences = [text]

        # Step 11: Remove stopwords if requested
        if remove_stopwords:
            tokens = [
                token
                for token in tokens
                if token.lower() not in self.stop_words
                and token.lower() not in self.financial_stopwords
            ]

        # Step 12: Lemmatization
        if lemmatize:
            tokens = self._lemmatize_tokens(tokens)

        # Reconstruct cleaned text
        cleaned_text = " ".join(tokens)

        # Update metadata
        metadata.update(
            {
                "cleaned_length": len(cleaned_text),
                "token_count": len(tokens),
                "sentence_count": len(sentences),
                "extracted_tickers": tickers,
                "extracted_currencies": currencies,
                "extracted_percentages": percentages,
                "extracted_urls": urls,
                "extracted_emails": emails,
                "extracted_handles": handles,
                "extracted_hashtags": hashtags,
            }
        )

        return CleanedText(
            original=original_text,
            cleaned=cleaned_text,
            tokens=tokens,
            sentences=sentences,
            metadata=metadata,
        )

    def _extract_tickers(self, text: str) -> List[str]:
        """Extract stock tickers from text"""
        matches = self.ticker_pattern.findall(text.upper())
        # Filter out common false positives
        false_positives = {
            "THE",
            "AND",
            "FOR",
            "ARE",
            "BUT",
            "NOT",
            "YOU",
            "ALL",
            "CAN",
            "HER",
            "WAS",
            "ONE",
            "OUR",
            "HAD",
            "BY",
            "UP",
            "DO",
            "NO",
            "IF",
            "MY",
            "ON",
            "AS",
            "WE",
            "HE",
            "BE",
            "TO",
            "OF",
            "IT",
            "IS",
            "IN",
            "AT",
            "OR",
        }
        return [
            ticker
            for ticker in matches
            if ticker not in false_positives and len(ticker) <= 5
        ]

    def _extract_currencies(self, text: str) -> List[str]:
        """Extract currency amounts from text"""
        return self.currency_pattern.findall(text)

    def _extract_percentages(self, text: str) -> List[str]:
        """Extract percentages from text"""
        return self.percentage_pattern.findall(text)

    def _expand_abbreviations(self, text: str) -> str:
        """Expand financial abbreviations"""
        words = text.split()
        expanded_words = []

        for word in words:
            clean_word = word.lower().strip(string.punctuation)
            if clean_word in self.financial_abbreviations:
                expanded_words.append(self.financial_abbreviations[clean_word])
            else:
                expanded_words.append(word)

        return " ".join(expanded_words)

    def _lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Apply lemmatization to tokens"""
        try:
            # Get POS tags for better lemmatization
            pos_tags = pos_tag(tokens)
            lemmatized = []

            if self.lemmatizer is None:
                # Fallback to original tokens if lemmatizer is not available
                return tokens

            for token, pos in pos_tags:
                # Convert POS tag to WordNet format
                wordnet_pos = self._get_wordnet_pos(pos)
                if wordnet_pos:
                    lemmatized.append(self.lemmatizer.lemmatize(token, wordnet_pos))
                else:
                    lemmatized.append(self.lemmatizer.lemmatize(token))

            return lemmatized
        except Exception:
            # Fallback to simple lemmatization or original tokens
            if self.lemmatizer is None:
                return tokens
            return [self.lemmatizer.lemmatize(token) for token in tokens]

    def _get_wordnet_pos(self, treebank_tag: str) -> Optional[str]:
        """Convert TreeBank POS tag to WordNet POS tag"""
        if treebank_tag.startswith("J"):
            return "a"  # adjective
        elif treebank_tag.startswith("V"):
            return "v"  # verb
        elif treebank_tag.startswith("N"):
            return "n"  # noun
        elif treebank_tag.startswith("R"):
            return "r"  # adverb
        else:
            return None

    def batch_clean(self, texts: List[str], **kwargs) -> List[CleanedText]:
        """Clean multiple texts in batch"""
        return [self.clean_text(text, **kwargs) for text in texts]

    def get_cleaning_stats(self, cleaned_texts: List[CleanedText]) -> Dict[str, Any]:
        """Get statistics about cleaning process"""
        if not cleaned_texts:
            return {}

        total_original = sum(
            ct.metadata.get("original_length", 0) for ct in cleaned_texts
        )
        total_cleaned = sum(
            ct.metadata.get("cleaned_length", 0) for ct in cleaned_texts
        )
        total_tokens = sum(ct.metadata.get("token_count", 0) for ct in cleaned_texts)
        total_sentences = sum(
            ct.metadata.get("sentence_count", 0) for ct in cleaned_texts
        )

        all_tickers = set()
        for ct in cleaned_texts:
            all_tickers.update(ct.metadata.get("extracted_tickers", []))

        return {
            "total_texts": len(cleaned_texts),
            "total_original_chars": total_original,
            "total_cleaned_chars": total_cleaned,
            "compression_ratio": (
                total_cleaned / total_original if total_original > 0 else 0
            ),
            "total_tokens": total_tokens,
            "total_sentences": total_sentences,
            "avg_tokens_per_text": total_tokens / len(cleaned_texts),
            "unique_tickers_found": len(all_tickers),
            "tickers": list(all_tickers),
        }
