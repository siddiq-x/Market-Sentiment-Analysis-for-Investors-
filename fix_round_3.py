#!/usr/bin/env python3
"""Round 3 - Fix next batch of broken strings"""
from pathlib import Path

base_dir = Path(__file__).parent

all_fixes = {
    "src/fusion/feature_engineer.py": [
        (
            "def _resample_sentiment_data(self, sentiment_data: pd.DataFrame) ->\n    pd.DataFrame:",
            "def _resample_sentiment_data(self, sentiment_data: pd.DataFrame) -> pd.DataFrame:",
        ),
    ],
    "src/fusion/fusion_manager.py": [
        (
            "training_config: Optional[TrainingConfig] = None) ->\n    Dict[str, Any]:",
            "training_config: Optional[TrainingConfig] = None) -> Dict[str, Any]:",
        ),
    ],
    "src/fusion/lstm_model.py": [
        (
            "self.training_config = TrainingConfig(**checkpoint['training_config\n    '])",
            "self.training_config = TrainingConfig(**checkpoint['training_config'])",
        ),
    ],
    "src/pipeline/stream_processor.py": [
        (
            'self.logger.error(f"Error fetching news data:\n    {str(e)}")',
            'self.logger.error(f"Error fetching news data: {str(e)}")',
        ),
    ],
    "src/preprocessing/bot_detector.py": [
        (
            "content_score, content_reasons =\n    self._check_content_patterns(text)",
            "content_score, content_reasons = self._check_content_patterns(text)",
        ),
    ],
    "src/preprocessing/ner_extractor.py": [
        (
            "'Goldman Sachs', 'Morgan Stanley', 'JPMorgan', 'Bank of\n    America',",
            "'Goldman Sachs', 'Morgan Stanley', 'JPMorgan', 'Bank of America',",
        ),
    ],
    "src/sentiment/lexicon_analyzer.py": [
        (
            "def _determine_sentiment(self, word_scores: Dict[str, float]) ->\n    Tuple[str, float]:",
            "def _determine_sentiment(self, word_scores: Dict[str, float]) -> Tuple[str, float]:",
        ),
    ],
    "src/utils/model_retrainer.py": [
        (
            "precision = true_positives / (true_positives + false_positives) if\n    (true_positives + false_positives) > 0 else 0",
            "precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0",
        ),
    ],
}

total_fixes = 0
fixed_files = 0

for file_path, fixes in all_fixes.items():
    filepath = base_dir / file_path

    if not filepath.exists():
        print(f"✗ {file_path}: NOT FOUND")
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
        print(f"✓ {file_path}: {changes} fixes applied")

print(f"\n{'='*70}")
print(f"Fixed {fixed_files} files ({total_fixes} total fixes)")
print(f"{'='*70}")
