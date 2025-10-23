#!/usr/bin/env python3
"""Test compilation of all files"""
import py_compile
from pathlib import Path

base_dir = Path(__file__).parent

all_files = [
    'main.py',
    'src/dashboard/app.py',
    'src/data_ingestion/market_connector.py',
    'src/data_ingestion/news_connector.py',
    'src/data_ingestion/social_connector.py',
    'src/preprocessing/text_cleaner.py',
    'src/sentiment/ensemble_analyzer.py',
    'src/sentiment/finbert_analyzer.py',
    'src/fusion/feature_engineer.py',
    'src/fusion/fusion_manager.py',
    'src/fusion/lstm_model.py',
    'src/pipeline/stream_processor.py',
    'src/preprocessing/bot_detector.py',
    'src/preprocessing/ner_extractor.py',
    'src/sentiment/lexicon_analyzer.py',
    'src/utils/model_retrainer.py',
]

ok_files = []
error_files = []

for file_rel in all_files:
    filepath = base_dir / file_rel
    if not filepath.exists():
        error_files.append((file_rel, "NOT FOUND"))
        continue
    
    try:
        py_compile.compile(str(filepath), doraise=True)
        ok_files.append(file_rel)
        print(f"OK   {file_rel}")
    except py_compile.PyCompileError as e:
        error_files.append((file_rel, str(e.exc_value)))
        print(f"ERR  {file_rel}: {e.exc_value}")

print(f"\n{'='*70}")
print(f"WORKING: {len(ok_files)}/{len(all_files)} files")
print(f"ERRORS:  {len(error_files)} files")
print(f"{'='*70}")

if error_files:
    print("\nFiles with errors:")
    for file, error in error_files:
        print(f"  - {file}")
