#!/usr/bin/env python3
"""
Apply all the fixes from the checkpoint that were already proven to work.
This applies exact string replacements for all broken multi-line strings.
"""
from pathlib import Path

fixes = {
    'src/data_ingestion/market_connector.py': [
        ('super().__init__("MarketData", {"alpha_vantage_key":\n    config.api.alpha_vantage_key})', 
         'super().__init__("MarketData", {"alpha_vantage_key": config.api.alpha_vantage_key})'),
        ('self.logger.info("Successfully connected to market data\n    sources")',
         'self.logger.info("Successfully connected to market data sources")'),
    ],
    'src/data_ingestion/news_connector.py': [
        ('queries.append(f"{ticker} stock OR {ticker} earnings OR {ticker}\n    financial")',
         'queries.append(f"{ticker} stock OR {ticker} earnings OR {ticker} financial")'),
        ('content = f"{article.get(\'title\', \'\')} {article.get(\'description\',\n    \'\')}"',
         'content = f"{article.get(\'title\', \'\')} {article.get(\'description\', \'\')}"'),
        ('timestamp=datetime.fromisoformat(article["publishedAt"].replace("Z"\n    , "+00:00")),',
         'timestamp=datetime.fromisoformat(article["publishedAt"].replace("Z", "+00:00")),'),
    ],
    'src/data_ingestion/social_connector.py': [
        ('super().__init__("Twitter", {"bearer_token":\n    config.api.twitter_bearer_token})',
         'super().__init__("Twitter", {"bearer_token": config.api.twitter_bearer_token})'),
        ('tickers = tickers or config.monitored_tickers[:10]  # Limit for API\n    quota',
         'tickers = tickers or config.monitored_tickers[:10]  # Limit for API quota'),
        ('query = f"${ticker} OR {ticker} (stock OR price OR earnings OR buy\n    OR sell) -is:retweet lang:en"',
         'query = f"${ticker} OR {ticker} (stock OR price OR earnings OR buy OR sell) -is:retweet lang:en"'),
        ('tweet_fields=[\'created_at\', \'author_id\', \'public_metrics\',\n    \'context_annotations\'],',
         'tweet_fields=[\'created_at\', \'author_id\', \'public_metrics\', \'context_annotations\'],'),
        ('engagement_score = self._calculate_engagement_score(tweet.public_metric\n    s)',
         'engagement_score = self._calculate_engagement_score(tweet.public_metrics)'),
        ('posts_data.append(self._convert_reddit_to_datapoint(pos\n    t_data, subreddit))',
         'posts_data.append(self._convert_reddit_to_datapoint(post_data, subreddit))'),
    ],
    'src/preprocessing/text_cleaner.py': [
        ('self.url_pattern = re.compile(\n    r\'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\\\(\\\\),]|(?:%[0-9a-fA-F\n    ][0-9a-fA-F]))+\'\n)',
         'self.url_pattern = re.compile(\n    r\'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\\\(\\\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+\'\n)'),
        ('self.email_pattern = re.compile(r\'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[\\n    A-Z|a-z]{2,}\\b\')',
         'self.email_pattern = re.compile(r\'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b\')'),
        ('percentages = self._extract_percentages(text) if preserve_numbers else\n    []',
         'percentages = self._extract_percentages(text) if preserve_numbers else []'),
        ('text = re.sub(rf\'\\b{re.escape(ticker)}\\b\', placeholder, text,\n    flags=re.IGNORECASE)',
         'text = re.sub(rf\'\\b{re.escape(ticker)}\\b\', placeholder, text, flags=re.IGNORECASE)'),
        ('return [ticker for ticker in matches if ticker not in false_positives\n    and len(ticker) <= 5]',
         'return [ticker for ticker in matches if ticker not in false_positives and len(ticker) <= 5]'),
        ('lemmatized.append(self.lemmatizer.lemmatize(token,\n    wordnet_pos))',
         'lemmatized.append(self.lemmatizer.lemmatize(token, wordnet_pos))'),
        ('total_original = sum(ct.metadata.get("original_length", 0) for ct in\n    cleaned_texts)',
         'total_original = sum(ct.metadata.get("original_length", 0) for ct in cleaned_texts)'),
        ('"compression_ratio": total_cleaned / total_original if\n    total_original > 0 else 0,',
         '"compression_ratio": total_cleaned / total_original if total_original > 0 else 0,'),
    ],
    'src/sentiment/ensemble_analyzer.py': [
        ('self.logger.info(f"Ensemble weights - FinBERT:\n    {self.finbert_weight:.2f}, Lexicon: {self.lexicon_weight:.2f}")',
         'self.logger.info(f"Ensemble weights - FinBERT: {self.finbert_weight:.2f}, Lexicon: {self.lexicon_weight:.2f}")'),
        ('finbert_result = self.finbert_analyzer.analyze_sentiment(text,\n    context)',
         'finbert_result = self.finbert_analyzer.analyze_sentiment(text, context)'),
        ('ensemble_sentiment, ensemble_confidence, ensemble_score =\n    self._combine_results(\n            finbert_result, lexicon_result\n        )',
         'ensemble_sentiment, ensemble_confidence, ensemble_score = self._combine_results(\n            finbert_result, lexicon_result\n        )'),
        ('weights={"finbert": self.finbert_weight, "lexicon":\n    self.lexicon_weight},',
         'weights={"finbert": self.finbert_weight, "lexicon": self.lexicon_weight},'),
        ('finbert_score = self._sentiment_to_score(finbert_result.sentiment,\n    finbert_result.scores)',
         'finbert_score = self._sentiment_to_score(finbert_result.sentiment, finbert_result.scores)'),
        ('agreement = self._calculate_agreement(finbert_result.sentiment,\n    lexicon_result.sentiment)',
         'agreement = self._calculate_agreement(finbert_result.sentiment, lexicon_result.sentiment)'),
        ('self.logger.info(f"Updated weights - FinBERT:\n    {self.finbert_weight:.2f}, Lexicon: {self.lexicon_weight:.2f}")',
         'self.logger.info(f"Updated weights - FinBERT: {self.finbert_weight:.2f}, Lexicon: {self.lexicon_weight:.2f}")'),
        ('self.logger.warning("Invalid weights provided, keeping current\n    weights")',
         'self.logger.warning("Invalid weights provided, keeping current weights")'),
    ],
    'src/sentiment/finbert_analyzer.py': [
        ('self.device = torch.device("cuda" if torch.cuda.is_available() else\n    "cpu")',
         'self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")'),
        ('\'strong_positive\': [\'rally\', \'surge\', \'soar\', \'boom\', \'bullish\',\n    \'outperform\'],',
         '\'strong_positive\': [\'rally\', \'surge\', \'soar\', \'boom\', \'bullish\', \'outperform\'],'),
        ('\'strong_negative\': [\'crash\', \'plunge\', \'collapse\', \'bearish\',\n    \'underperform\', \'decline\'],',
         '\'strong_negative\': [\'crash\', \'plunge\', \'collapse\', \'bearish\', \'underperform\', \'decline\'],'),
        ('\'uncertainty\': [\'volatile\', \'uncertain\', \'mixed\', \'cautious\',\n    \'wait-and-see\']',
         '\'uncertainty\': [\'volatile\', \'uncertain\', \'mixed\', \'cautious\', \'wait-and-see\']'),
        ('self.model = AutoModelForSequenceClassification.from_pretrained(sel\n    f.model_name)',
         'self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)'),
        ('self.logger.info("Attempting fallback to\n    distilbert-base-uncased-finetuned-sst-2-english")',
         'self.logger.info("Attempting fallback to distilbert-base-uncased-finetuned-sst-2-english")'),
        ('self.model_name = "distilbert-base-uncased-finetuned-sst-2-engl\n    ish"',
         'self.model_name = "distilbert-base-uncased-finetuned-sst-2-english"'),
        ('raise RuntimeError("Failed to load any sentiment analysis\n    model")',
         'raise RuntimeError("Failed to load any sentiment analysis model")'),
        ('adjusted_scores = self._apply_financial_context(processed_text,\n    raw_scores, context)',
         'adjusted_scores = self._apply_financial_context(processed_text, raw_scores, context)'),
        ('adjusted_scores["positive"] = min(1.0, adjusted_scores["positive"]\n    + boost)',
         'adjusted_scores["positive"] = min(1.0, adjusted_scores["positive"] + boost)'),
        ('if any(term in text_lower for term in ["earnings", "quarterly",\n    "revenue", "profit"]):',
         'if any(term in text_lower for term in ["earnings", "quarterly", "revenue", "profit"]):'),
        ('max_key = max(adjusted_scores.keys(), key=lambda k:\n    adjusted_scores[k])',
         'max_key = max(adjusted_scores.keys(), key=lambda k: adjusted_scores[k])'),
        ('self.model = AutoModelForSequenceClassification.from_pretrained\n    (cache_dir)',
         'self.model = AutoModelForSequenceClassification.from_pretrained(cache_dir)'),
    ],
}

base_dir = Path(__file__).parent
fixed_count = 0

for file_path, replacements in fixes.items():
    full_path = base_dir / file_path
    if not full_path.exists():
        print(f"⚠ File not found: {file_path}")
        continue
    
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original = content
        for old, new in replacements:
            if old in content:
                content = content.replace(old, new)
        
        if content != original:
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            fixed_count += 1
            print(f"✓ Fixed: {file_path}")
        else:
            print(f"- No changes needed: {file_path}")
    except Exception as e:
        print(f"✗ Error fixing {file_path}: {e}")

print(f"\n{'='*60}")
print(f"Successfully fixed {fixed_count} files")
print(f"{'='*60}")
