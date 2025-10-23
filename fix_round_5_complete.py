#!/usr/bin/env python3
"""Round 5 - Fix all remaining errors comprehensively"""
from pathlib import Path

base_dir = Path(__file__).parent

all_fixes = {
    "src/fusion/feature_engineer.py": [
        (
            "sentiment_features['credibility_score'] =\n    data.get('credibility_score', 0)",
            "sentiment_features['credibility_score'] = data.get('credibility_score', 0)",
        ),
        (
            "sentiment_features['sentiment_volume'] = data.get('sentiment_volume',\n    0)",
            "sentiment_features['sentiment_volume'] = data.get('sentiment_volume', 0)",
        ),
    ],
    "src/fusion/fusion_manager.py": [
        (
            'self.logger.info(f"Trained fusion model for {target_ticker}:\n    {train_metrics}")',
            'self.logger.info(f"Trained fusion model for {target_ticker}: {train_metrics}")',
        ),
    ],
    "src/fusion/lstm_model.py": [
        (
            "def predict(self, feature_set: FeatureSet) ->\n    List[PredictionResult]:",
            "def predict(self, feature_set: FeatureSet) -> List[PredictionResult]:",
        ),
    ],
    "src/pipeline/stream_processor.py": [
        (
            "market_data = ingestion_manager.fetch_by_connector('market\n    ')",
            "market_data = ingestion_manager.fetch_by_connector('market')",
        ),
    ],
    "src/preprocessing/bot_detector.py": [
        (
            "author_score, author_reasons =\n    self._check_author_info(author_info)",
            "author_score, author_reasons = self._check_author_info(author_info)",
        ),
    ],
    "src/preprocessing/ner_extractor.py": [
        (
            "def extract_entities(self, text: str) ->\n    Dict[str, List[Any]]:",
            "def extract_entities(self, text: str) -> Dict[str, List[Any]]:",
        ),
    ],
    "src/sentiment/lexicon_analyzer.py": [
        (
            "def _calculate_confidence(self, word_scores: Dict[str, float]) ->\n    float:",
            "def _calculate_confidence(self, word_scores: Dict[str, float]) -> float:",
        ),
    ],
    "src/utils/model_retrainer.py": [
        (
            'self.logger.info(f"Retraining model for {ticker}:\n    {performance}")',
            'self.logger.info(f"Retraining model for {ticker}: {performance}")',
        ),
    ],
}

total_fixes = 0
fixed_files = 0

for file_path, fixes in all_fixes.items():
    filepath = base_dir / file_path

    if not filepath.exists():
        print(f"X {file_path}: NOT FOUND")
        continue

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    changes = 0
    for old, new in fixes:
        if old in content:
            content = content.replace(old, new)
            changes += 1

    if changes > 0:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        fixed_files += 1
        total_fixes += changes
        print(f"OK {file_path}: {changes} fixes")

print(f"\n{'='*70}")
print(f"Fixed {fixed_files} files ({total_fixes} total fixes)")
print(f"{'='*70}")
