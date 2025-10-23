#!/usr/bin/env python3
"""Fix ALL remaining files with broken strings"""
from pathlib import Path

base_dir = Path(__file__).parent

all_fixes = {
    "src/fusion/feature_engineer.py": [
        (
            'self.logger.info(f"Initialized feature engineer - lookback:\n    {lookback_window}h, horizon: {prediction_horizon}h")',
            'self.logger.info(f"Initialized feature engineer - lookback: {lookback_window}h, horizon: {prediction_horizon}h")',
        ),
    ],
    "src/fusion/fusion_manager.py": [
        (
            "from .lstm_model import MultimodalFusionEngine, ModelConfig, TrainingConfig,\n    PredictionResult",
            "from .lstm_model import (MultimodalFusionEngine, ModelConfig, TrainingConfig,\n                         PredictionResult)",
        ),
        (
            'self.logger.info(f"Initialized fusion manager - lookback:\n    {lookback_window}h, horizon: {prediction_horizon}h")',
            'self.logger.info(f"Initialized fusion manager - lookback: {lookback_window}h, horizon: {prediction_horizon}h")',
        ),
        (
            'self.logger.info(f"Prepared training data for {target_ticker}:\n    {len(features.features)} samples")',
            'self.logger.info(f"Prepared training data for {target_ticker}: {len(features.features)} samples")',
        ),
    ],
    "src/fusion/lstm_model.py": [
        (
            "trainable_params = sum(p.numel() for p in self.model.parameters() if\n    p.requires_grad)",
            "trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)",
        ),
        (
            'self.logger.info(f"Model built - Total params: {total_params:,},\n    Trainable: {trainable_params:,}")',
            'self.logger.info(f"Model built - Total params: {total_params:,}, Trainable: {trainable_params:,}")',
        ),
    ],
    "src/pipeline/stream_processor.py": [
        (
            'self.logger.error(f"Error in subscriber callback:\n    {str(e)}")',
            'self.logger.error(f"Error in subscriber callback: {str(e)}")',
        ),
        (
            "def get_messages(self, topic: str, max_messages: int = 100) ->\n    List[StreamMessage]:",
            "def get_messages(self, topic: str, max_messages: int = 100) -> List[StreamMessage]:",
        ),
    ],
    "src/preprocessing/bot_detector.py": [
        (
            "r'(?:\\d+%|\\d+x)\\s+(?:profit|return|gains?)\\s+(?:guaranteed|cert\n    ain)',",
            "r'(?:\\d+%|\\d+x)\\s+(?:profit|return|gains?)\\s+(?:guaranteed|certain)',",
        ),
    ],
    "src/preprocessing/ner_extractor.py": [
        (
            '[{"TEXT": {"REGEX": r"^\\d+\\.?\\d*[BMK]?\\s*(dollars?|USD|billion|mill\n    ion|thousand)$"}}]',
            '[{"TEXT": {"REGEX": r"^\\d+\\.?\\d*[BMK]?\\s*(dollars?|USD|billion|million|thousand)$"}}]',
        ),
    ],
    "src/sentiment/lexicon_analyzer.py": [
        (
            'self.logger.warning(f"Could not load additional lexicons:\n    {str(e)}")',
            'self.logger.warning(f"Could not load additional lexicons: {str(e)}")',
        ),
    ],
    "src/utils/model_retrainer.py": [
        (
            'self.logger.error(f"Error evaluating model for {ticker}:\n    {str(e)}")',
            'self.logger.error(f"Error evaluating model for {ticker}: {str(e)}")',
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
    else:
        print(f"- {file_path}: No changes needed")

print(f"\n{'='*70}")
print(f"Fixed {fixed_files} files ({total_fixes} total fixes)")
print(f"{'='*70}")
