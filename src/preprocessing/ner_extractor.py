"""
Named Entity Recognition for financial content
"""
import spacy
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import re
from collections import defaultdict

# Try to load spaCy model, download if not available
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "spacy", "download",
    "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


@dataclass
class FinancialEntity:
    """Container for financial entities"""
    text: str
    label: str
    start: int
    end: int
    confidence: float
    context: str


class FinancialNER:
    """Named Entity Recognition specialized for financial content"""

    def __init__(self):
        self.logger = logging.getLogger("financial_ner")
        self.nlp = nlp

        # Add custom financial patterns
        self._add_financial_patterns()

        # Financial entity mappings
        self.financial_labels = {
            'TICKER': 'Stock Ticker',
            'COMPANY': 'Company Name',
            'CURRENCY': 'Currency Amount',
            'PERCENTAGE': 'Percentage',
            'FINANCIAL_METRIC': 'Financial Metric',
            'MARKET_EVENT': 'Market Event',
            'FINANCIAL_INSTRUMENT': 'Financial Instrument',
            'EXCHANGE': 'Stock Exchange',
            'SECTOR': 'Industry Sector',
            'ANALYST_RATING': 'Analyst Rating'
        }

        # Known financial entities
        self.known_entities = self._load_financial_entities()

    def _add_financial_patterns(self):
        """Add custom financial patterns to spaCy pipeline"""
        from spacy.matcher import Matcher

        self.matcher = Matcher(self.nlp.vocab)

        # Stock ticker patterns
        ticker_pattern = [{"TEXT": {"REGEX": r"^\$?[A-Z]{1,5}$"}}]
        self.matcher.add("TICKER", [ticker_pattern])

        # Currency patterns
        currency_patterns = [
            [{"TEXT": {"REGEX": r"^\$\d+\.?\d*[BMK]?$"}}],
            [{"TEXT": {"REGEX": r"^\d+\.?\d*[BMK]?\s*(dollars?|USD|billion|million|thousand)$"}}]
        ]
        self.matcher.add("CURRENCY", currency_patterns)

        # Percentage patterns
        percentage_pattern = [{"TEXT": {"REGEX": r"^\d+\.?\d*%$"}}]
        self.matcher.add("PERCENTAGE", [percentage_pattern])

        # Financial metrics patterns
        metric_patterns = [
            [{"LOWER": {"IN": ["eps", "pe", "pb", "roe", "roa", "ebitda",
    "revenue", "profit", "margin"]}}],
            [{"LOWER": "earnings"}, {"LOWER": "per"}, {"LOWER": "share"}],
            [{"LOWER": "price"}, {"LOWER": "to"}, {"LOWER": "earnings"}],
            [{"LOWER": "market"}, {"LOWER": "cap"}],
            [{"LOWER": "book"}, {"LOWER": "value"}]
        ]
        self.matcher.add("FINANCIAL_METRIC", metric_patterns)

        # Market events
        event_patterns = [
            [{"LOWER": {"IN": ["ipo", "merger", "acquisition", "buyout",
    "spinoff"]}}],
            [{"LOWER": "earnings"}, {"LOWER": "report"}],
            [{"LOWER": "dividend"}, {"LOWER": "announcement"}],
            [{"LOWER": "stock"}, {"LOWER": "split"}],
            [{"LOWER": "market"}, {"LOWER": {"IN": ["crash", "rally",
    "correction", "bubble"]}}]
        ]
        self.matcher.add("MARKET_EVENT", event_patterns)

        # Financial instruments
        instrument_patterns = [
            [{"LOWER": {"IN": ["stock", "bond", "option", "future", "etf",
    "mutual", "reit"]}}],
            [{"LOWER": "call"}, {"LOWER": "option"}],
            [{"LOWER": "put"}, {"LOWER": "option"}],
            [{"LOWER": "treasury"}, {"LOWER": {"IN": ["bill", "note",
    "bond"]}}]
        ]
        self.matcher.add("FINANCIAL_INSTRUMENT", instrument_patterns)

        # Analyst ratings
        rating_patterns = [
            [{"LOWER": {"IN": ["buy", "sell", "hold", "strong", "weak",
    "outperform", "underperform"]}}],
            [{"LOWER": "price"}, {"LOWER": "target"}],
            [{"LOWER": {"IN": ["upgrade", "downgrade", "initiate",
    "coverage"]}}]
        ]
        self.matcher.add("ANALYST_RATING", rating_patterns)

    def _load_financial_entities(self) -> Dict[str, List[str]]:
        """Load known financial entities"""
        return {
            'exchanges': [
                'NYSE', 'NASDAQ', 'AMEX', 'LSE', 'TSE', 'HKSE', 'SSE', 'BSE',
                'New York Stock Exchange', 'London Stock Exchange', 'Tokyo Stock Exchange'],
            'sectors': [
                'Technology', 'Healthcare', 'Financial', 'Energy', 'Consumer',
    'Industrial',
                'Materials', 'Utilities', 'Real Estate', 'Communication Services',
                'Biotechnology', 'Pharmaceuticals', 'Banking', 'Insurance',
    'Oil & Gas'
            ],
            'financial_institutions': [
                'Federal Reserve', 'Fed', 'SEC', 'CFTC', 'FINRA', 'FDIC',
                'Goldman Sachs', 'Morgan Stanley', 'JPMorgan', 'Bank of America',
                'Wells Fargo', 'Citigroup', 'BlackRock', 'Vanguard', 'Fidelity'
            ],
            'market_indices': [
                'S&P 500', 'Dow Jones', 'NASDAQ', 'Russell 2000', 'VIX',
                'FTSE 100', 'Nikkei', 'DAX', 'CAC 40', 'Hang Seng'
            ]
        }

    def extract_entities(self, text: str, include_context: bool = True) -> List[FinancialEntity]:
        """Extract financial entities from text"""
        if not text:
            return []

        entities = []
        doc = self.nlp(text)

        # Extract standard spaCy entities
        for ent in doc.ents:
            if self._is_financial_entity(ent):
                context = (
                    self._get_context(text, ent.start_char, ent.end_char)
                    if include_context
                    else ""
                )

                entity = FinancialEntity(
                    text=ent.text,
                    label=self._map_spacy_label(ent.label_),
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=0.8,  # Default confidence for spaCy entities
                    context=context
                )
                entities.append(entity)

        # Extract custom pattern matches
        matches = self.matcher(doc)
        for match_id, start, end in matches:
            span = doc[start:end]
            label = self.nlp.vocab.strings[match_id]

            # Avoid duplicates
            if not any(e.start <= span.start_char < e.end or e.start <
    span.end_char <= e.end for e in entities):
                context = self._get_context(text, span.start_char,
    span.end_char) if include_context else ""

                entity = FinancialEntity(
                    text=span.text,
                    label=label,
                    start=span.start_char,
                    end=span.end_char,
                    confidence=0.9,  # Higher confidence for pattern matches
                    context=context
                )
                entities.append(entity)

        # Extract additional financial entities using regex and known lists
        additional_entities = self._extract_additional_entities(text,
    include_context)
        entities.extend(additional_entities)

        # Sort by position and remove overlaps
        entities = self._remove_overlapping_entities(entities)

        return entities

    def _is_financial_entity(self, ent) -> bool:
        """Check if spaCy entity is financially relevant"""
        financial_labels = {'ORG', 'MONEY', 'PERCENT', 'CARDINAL', 'DATE',
    'GPE'}
        return ent.label_ in financial_labels

    def _map_spacy_label(self, label: str) -> str:
        """Map spaCy labels to financial labels"""
        mapping = {
            'ORG': 'COMPANY',
            'MONEY': 'CURRENCY',
            'PERCENT': 'PERCENTAGE',
            'CARDINAL': 'FINANCIAL_METRIC',
            'DATE': 'DATE',
            'GPE': 'LOCATION'
        }
        return mapping.get(label, label)

    def _get_context(self, text: str, start: int, end: int, window: int = 50) -> str:
        """Get context around an entity"""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end].strip()

    def _extract_additional_entities(self, text: str, include_context: bool) -> List[FinancialEntity]:
        """Extract additional financial entities using custom logic"""
        entities = []

        # Extract stock tickers
        ticker_pattern = r'\$?([A-Z]{1,5})\b'
        for match in re.finditer(ticker_pattern, text):
            ticker = match.group(1)
            # Filter out common false positives
            if self._is_valid_ticker(ticker):
                context = (
                    self._get_context(text, match.start(), match.end())
                    if include_context
                    else ""
                )

                entity = FinancialEntity(
                    text=match.group(0),
                    label='TICKER',
                    start=match.start(),
                    end=match.end(),
                    confidence=0.85,
                    context=context
                )
                entities.append(entity)

        # Extract known financial institutions and exchanges
        for category, entity_list in self.known_entities.items():
            for known_entity in entity_list:
                pattern = r'\b' + re.escape(known_entity) + r'\b'
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    context = (
                        self._get_context(text, match.start(), match.end())
                        if include_context
                        else ""
                    )

                    entity = FinancialEntity(
                        text=match.group(0),
                        label=category.upper().rstrip('S'),  # Remove plural
                        start=match.start(),
                        end=match.end(),
                        confidence=0.95,
                        context=context
                    )
                    entities.append(entity)

        return entities

    def _is_valid_ticker(self, ticker: str) -> bool:
        """Check if a potential ticker is valid"""
        # Common false positives
        false_positives = {
            'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN',
            'HER', 'WAS', 'ONE', 'OUR', 'HAD', 'BY', 'UP', 'DO', 'NO', 'IF',
            'MY', 'ON', 'AS', 'WE', 'HE', 'BE', 'TO', 'OF', 'IT', 'IS', 'IN',
            'AT', 'OR', 'SO', 'AN', 'GO', 'US', 'AM', 'GET', 'NEW', 'NOW',
    'WAY'
        }

        return (len(ticker) <= 5 and
                ticker not in false_positives and
                ticker.isalpha() and
                ticker.isupper())

    def _remove_overlapping_entities(self, entities: List[FinancialEntity]) -> List[FinancialEntity]:
        """Remove overlapping entities, keeping the one with higher confidence"""
        if not entities:
            return []

        # Sort by start position
        entities.sort(key=lambda x: (x.start, -x.confidence))

        filtered = []
        for entity in entities:
            # Check for overlap with already filtered entities
            overlap = False
            for filtered_entity in filtered:
                if (
                    entity.start < filtered_entity.end and
                    entity.end > filtered_entity.start):
                    overlap = True
                    break

            if not overlap:
                filtered.append(entity)

        return filtered

    def get_entity_summary(self, entities: List[FinancialEntity]) -> Dict[str,
    Any]:
        """Get summary statistics of extracted entities"""
        if not entities:
            return {}

        entity_counts = defaultdict(int)
        entity_texts = defaultdict(set)

        for entity in entities:
            entity_counts[entity.label] += 1
            entity_texts[entity.label].add(entity.text.lower())

        return {
            "total_entities": len(entities),
            "entity_types": len(entity_counts),
            "counts_by_type": dict(entity_counts),
            "unique_entities_by_type": {
                label: len(texts) for label, texts in entity_texts.items()
            },
            "most_common_type": max(entity_counts.items(), key=lambda x:
    x[1])[0] if entity_counts else None,
            "average_confidence": sum(e.confidence for e in entities) /
    len(entities)
        }

    def extract_ticker_mentions(self, text: str) -> List[Tuple[str, int, str]]:
        """Extract ticker mentions with frequency and context"""
        entities = self.extract_entities(text)
        ticker_mentions = []

        ticker_counts = defaultdict(int)
        ticker_contexts = defaultdict(list)

        for entity in entities:
            if entity.label == 'TICKER':
                ticker = entity.text.replace('$', '').upper()
                ticker_counts[ticker] += 1
                ticker_contexts[ticker].append(entity.context)

        for ticker, count in ticker_counts.items():
            # Get the most informative context
            best_context = (
                max(ticker_contexts[ticker], key=len)
                if ticker_contexts[ticker]
                else ""
            )
            ticker_mentions.append((ticker, count, best_context))

        # Sort by frequency
        ticker_mentions.sort(key=lambda x: x[1], reverse=True)

        return ticker_mentions

    def batch_extract(self, texts: List[str]) -> List[List[FinancialEntity]]:
        """Extract entities from multiple texts"""
        return [self.extract_entities(text) for text in texts]
