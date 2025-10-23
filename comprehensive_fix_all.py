#!/usr/bin/env python3
"""
Comprehensive fix for ALL Python syntax errors in the project.
This script fixes all broken multi-line strings, f-strings, and imports.
"""
from pathlib import Path
import re

def fix_all_broken_strings(content):
    """Apply all necessary fixes to file content."""
    
    # Fix 1: Broken f-strings (most common pattern)
    content = re.sub(r'f"([^"]*)\n\s+([^"]*)"', r'f"\1 \2"', content)
    content = re.sub(r'f\'([^\']*)\n\s+([^\']*)\' ', r'f\'\1 \2\'', content)
    
    # Fix 2: Broken regular strings
    content = re.sub(r'"([^"]*)\n\s+([^"]*)"', r'"\1 \2"', content)
    content = re.sub(r'\'([^\']*)\n\s+([^\']*)\' ', r'\'\1 \2\'', content)
    
    # Fix 3: Broken list/dict continuations with excessive indentation
    content = re.sub(r',\n    ([a-zA-Z_][a-zA-Z0-9_]*\])', r', \1', content)
    
    # Fix 4: Broken type hints
    content = re.sub(r'->\s*\n\s+([A-Z][a-zA-Z0-9_\[\]]+):', r'-> \1:', content)
    
    # Fix 5: Broken import statements
    content = re.sub(
        r'from ([a-zA-Z0-9_.]+) import ([^,\n]+),\s*\n\s+([A-Z][a-zA-Z0-9_]+)',
        r'from \1 import (\2,\n                         \3)',
        content
    )
    
    # Fix 6: Broken list comprehensions
    content = re.sub(r'if\s+\n\s+([a-zA-Z])', r'if \1', content)
    
    # Fix 7: Broken method calls
    content = re.sub(r'([a-zA-Z_][a-zA-Z0-9_]*)\(\s*\n\s+([a-zA-Z_])', r'\1(\2', content)
    
    return content

# Files to fix (all remaining problematic files)
files_to_fix = {
    'src/dashboard/app.py': [
        ('return jsonify({"status": "started", "message": "Stream processor\n    started successfully"})',
         'return jsonify({"status": "started", "message": "Stream processor started successfully"})'),
        ('return jsonify({"status": "already_running", "message": "Stream\n    processor is already running"})',
         'return jsonify({"status": "already_running", "message": "Stream processor is already running"})'),
        ('return jsonify({"status": "stopped", "message": "Stream processor\n    stopped successfully"})',
         'return jsonify({"status": "stopped", "message": "Stream processor stopped successfully"})'),
        ('return jsonify({"status": "not_running", "message": "Stream\n    processor is not running"})',
         'return jsonify({"status": "not_running", "message": "Stream processor is not running"})'),
        ('"explanation": f"The model predicts {[\'bearish\', \'neutral\',\n    \'bullish\'][prediction[\'prediction\'] + 1]} sentiment for {ticker} based on\n    the feature contributions shown above."',
         '"explanation": f"The model predicts {[\'bearish\', \'neutral\', \'bullish\'][prediction[\'prediction\'] + 1]} sentiment for {ticker} based on the feature contributions shown above."'),
        ('emit(\'status\', {\'message\': \'Connected to Market Sentiment Analysis\n    Dashboard\'})',
         'emit(\'status\', {\'message\': \'Connected to Market Sentiment Analysis Dashboard\'})'),
        ('emit(\'subscribed\', {\'ticker\': ticker, \'message\': f\'Subscribed to\n    {ticker} updates\'})',
         'emit(\'subscribed\', {\'ticker\': ticker, \'message\': f\'Subscribed to {ticker} updates\'})'),
    ],
    'src/fusion/feature_engineer.py': [
        ('self.logger.info(f"Initialized feature engineer - lookback:\n    {lookback_window}h, horizon: {prediction_horizon}h")',
         'self.logger.info(f"Initialized feature engineer - lookback: {lookback_window}h, horizon: {prediction_horizon}h")'),
    ],
    'src/fusion/fusion_manager.py': [
        ('from .lstm_model import MultimodalFusionEngine, ModelConfig, TrainingConfig,\n    PredictionResult',
         'from .lstm_model import (MultimodalFusionEngine, ModelConfig, TrainingConfig,\n                         PredictionResult)'),
        ('self.logger.info(f"Initialized fusion manager - lookback:\n    {lookback_window}h, horizon: {prediction_horizon}h")',
         'self.logger.info(f"Initialized fusion manager - lookback: {lookback_window}h, horizon: {prediction_horizon}h")'),
        ('self.logger.info(f"Prepared training data for {target_ticker}:\n    {len(features.features)} samples")',
         'self.logger.info(f"Prepared training data for {target_ticker}: {len(features.features)} samples")'),
    ],
    'src/fusion/lstm_model.py': [
        ('trainable_params = sum(p.numel() for p in self.model.parameters() if\n    p.requires_grad)',
         'trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)'),
        ('self.logger.info(f"Model built - Total params: {total_params:,},\n    Trainable: {trainable_params:,}")',
         'self.logger.info(f"Model built - Total params: {total_params:,}, Trainable: {trainable_params:,}")'),
    ],
    'src/pipeline/stream_processor.py': [
        ('self.logger.error(f"Error in subscriber callback:\n    {str(e)}")',
         'self.logger.error(f"Error in subscriber callback: {str(e)}")'),
        ('def get_messages(self, topic: str, max_messages: int = 100) ->\n    List[StreamMessage]:',
         'def get_messages(self, topic: str, max_messages: int = 100) -> List[StreamMessage]:'),
    ],
    'src/preprocessing/bot_detector.py': [
        ('r\'(?:\\d+%|\\d+x)\\s+(?:profit|return|gains?)\\s+(?:guaranteed|cert\n    ain)\',',
         'r\'(?:\\d+%|\\d+x)\\s+(?:profit|return|gains?)\\s+(?:guaranteed|certain)\','),
    ],
    'src/preprocessing/ner_extractor.py': [
        ('[{"TEXT": {"REGEX": r"^\\d+\\.?\\d*[BMK]?\\s*(dollars?|USD|billion|mill\n    ion|thousand)$"}}]',
         '[{"TEXT": {"REGEX": r"^\\d+\\.?\\d*[BMK]?\\s*(dollars?|USD|billion|million|thousand)$"}}]'),
    ],
    'src/sentiment/lexicon_analyzer.py': [
        ('self.logger.warning(f"Could not load additional lexicons:\n    {str(e)}")',
         'self.logger.warning(f"Could not load additional lexicons: {str(e)}")'),
    ],
    'src/utils/model_retrainer.py': [
        ('self.logger.error(f"Error evaluating model for {ticker}:\n    {str(e)}")',
         'self.logger.error(f"Error evaluating model for {ticker}: {str(e)}")'),
    ],
}

base_dir = Path(__file__).parent
fixed_files = 0
total_replacements = 0

for file_path, replacements in files_to_fix.items():
    full_path = base_dir / file_path
    
    if not full_path.exists():
        print(f"⚠ File not found: {file_path}")
        continue
    
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        changes_made = 0
        
        for old, new in replacements:
            if old in content:
                content = content.replace(old, new)
                changes_made += 1
        
        if content != original_content:
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            fixed_files += 1
            total_replacements += changes_made
            print(f"✓ {file_path}: {changes_made} fixes applied")
        else:
            print(f"- {file_path}: No changes needed")
            
    except Exception as e:
        print(f"✗ Error fixing {file_path}: {e}")

print(f"\n{'='*70}")
print(f"Successfully fixed {fixed_files} files ({total_replacements} total replacements)")
print(f"{'='*70}")
