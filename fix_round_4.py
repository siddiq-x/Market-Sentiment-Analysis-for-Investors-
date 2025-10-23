#!/usr/bin/env python3
"""Round 4 - Continue fixing"""
from pathlib import Path

base_dir = Path(__file__).parent

all_fixes = {
    'src/fusion/feature_engineer.py': [
        ('sentiment_features[\'credibility_score\'] =\n    sentiment_aggregated[\'credibility_score\'].mean()',
         'sentiment_features[\'credibility_score\'] = sentiment_aggregated[\'credibility_score\'].mean()'),
    ],
    
    'src/preprocessing/bot_detector.py': [
        ('content_score, content_reasons =\n    self._analyze_content_patterns(content)',
         'content_score, content_reasons = self._analyze_content_patterns(content)'),
        ('author_score, author_reasons =\n    self._analyze_author_info(author_info)',
         'author_score, author_reasons = self._analyze_author_info(author_info)'),
    ],
    
    'src/pipeline/stream_processor.py': [
        ('if hasattr(ingestion_manager.connectors.get(\'market\'),\n    \'fetch_data\'):',
         'if hasattr(ingestion_manager.connectors.get(\'market\'), \'fetch_data\'):'),
        ('market_data = ingestion_manager.fetch_by_connector(\'mar\n    ket\')',
         'market_data = ingestion_manager.fetch_by_connector(\'market\')'),
    ],
    
    'src/sentiment/lexicon_analyzer.py': [
        ('positive_score = sum(score for score in word_scores.values() if score\n    > 0)',
         'positive_score = sum(score for score in word_scores.values() if score > 0)'),
    ],
    
    'src/utils/model_retrainer.py': [
        ('recall = true_positives / (true_positives + false_negatives) if\n    (true_positives + false_negatives) > 0 else 0',
         'recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0'),
    ],
}

total_fixes = 0
fixed_files = 0

for file_path, fixes in all_fixes.items():
    filepath = base_dir / file_path
    
    if not filepath.exists():
        print(f"X {file_path}: NOT FOUND")
        continue
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    changes = 0
    for old, new in fixes:
        if old in content:
            content = content.replace(old, new)
            changes += 1
    
    if changes > 0:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        fixed_files += 1
        total_fixes += changes
        print(f"OK {file_path}: {changes} fixes")

print(f"\n{'='*70}")
print(f"Fixed {fixed_files} files ({total_fixes} total fixes)")
print(f"{'='*70}")
