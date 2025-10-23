#!/usr/bin/env python3
"""Complete fix for ALL remaining broken strings"""
from pathlib import Path

base_dir = Path(__file__).parent

all_fixes = {
    'src/fusion/feature_engineer.py': [
        ('sentiment_data[\'timestamp\'] =\n    pd.to_datetime(sentiment_data.index)',
         'sentiment_data[\'timestamp\'] = pd.to_datetime(sentiment_data.index)'),
        ('ticker_market_data = market_data[market_data.get(\'ticker\', \'\') ==\n    target_ticker].copy()',
         'ticker_market_data = market_data[market_data.get(\'ticker\', \'\') == target_ticker].copy()'),
        ('self.logger.warning(f"No market data found for ticker\n    {target_ticker}")',
         'self.logger.warning(f"No market data found for ticker {target_ticker}")'),
    ],
    
    'src/fusion/fusion_manager.py': [
        ('self.logger.info(f"Prepared training data for {target_ticker}:\n    {feature_set.metadata}")',
         'self.logger.info(f"Prepared training data for {target_ticker}: {feature_set.metadata}")'),
    ],
    
    'src/fusion/lstm_model.py': [
        ('f"Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f},\n    Val Acc: {val_acc:.4f}"',
         'f"Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}"'),
        ('if patience_counter >= self.training_config.early_stopping_patience\n    :',
         'if patience_counter >= self.training_config.early_stopping_patience:'),
    ],
    
    'src/pipeline/stream_processor.py': [
        ('ingestion_thread = threading.Thread(target=self._simulate_data_ingestio\n    n)',
         'ingestion_thread = threading.Thread(target=self._simulate_data_ingestion)'),
    ],
    
    'src/preprocessing/bot_detector.py': [
        ('r\'guaranteed\\s+(?:profit|return|gains)\',  # Unrealistic\n    promises',
         'r\'guaranteed\\s+(?:profit|return|gains)\',  # Unrealistic promises'),
    ],
    
    'src/preprocessing/ner_extractor.py': [
        ('\'New York Stock Exchange\', \'London Stock Exchange\', \'Tokyo\n    Stock Exchange\'',
         '\'New York Stock Exchange\', \'London Stock Exchange\', \'Tokyo Stock Exchange\''),
        ('\'Technology\', \'Healthcare\', \'Financial\', \'Energy\', \'Consumer\',\n    \'Industrial\',',
         '\'Technology\', \'Healthcare\', \'Financial\', \'Energy\', \'Consumer\', \'Industrial\','),
        ('\'Materials\', \'Utilities\', \'Real Estate\', \'Communication\n    Services\',',
         '\'Materials\', \'Utilities\', \'Real Estate\', \'Communication Services\','),
    ],
    
    'src/sentiment/lexicon_analyzer.py': [
        ('adjusted_scores[word] = current_score *\n    self.intensifiers[prev_word]',
         'adjusted_scores[word] = current_score * self.intensifiers[prev_word]'),
        ('adjusted_scores[word] = current_score *\n    self.diminishers[prev_word]',
         'adjusted_scores[word] = current_score * self.diminishers[prev_word]'),
    ],
    
    'src/utils/model_retrainer.py': [
        ('def _calculate_performance_metrics(self, ticker: str) ->\n    Optional[ModelPerformanceMetrics]:',
         'def _calculate_performance_metrics(self, ticker: str) -> Optional[ModelPerformanceMetrics]:'),
    ],
}

total_fixes = 0
fixed_files = 0

for file_path, fixes in all_fixes.items():
    filepath = base_dir / file_path
    
    if not filepath.exists():
        print(f"✗ {file_path}: NOT FOUND")
        continue
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    changes = 0
    for old, new in fixes:
        if old in content:
            content = content.replace(old, new)
            changes += 1
        else:
            print(f"  ⚠ Pattern not found in {file_path}: {old[:50]}...")
    
    if changes > 0:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        fixed_files += 1
        total_fixes += changes
        print(f"✓ {file_path}: {changes} fixes applied")
    else:
        print(f"- {file_path}: No changes needed")

print(f"\n{'='*70}")
print(f"Fixed {fixed_files} files ({total_fixes} total fixes)")
print(f"{'='*70}")
