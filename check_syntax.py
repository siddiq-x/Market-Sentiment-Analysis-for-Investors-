#!/usr/bin/env python3
"""Check which files have syntax errors."""
import py_compile
from pathlib import Path

files_to_check = [
    "src/dashboard/app.py",
    "src/fusion/feature_engineer.py",
    "src/fusion/fusion_manager.py",
    "src/fusion/lstm_model.py",
    "src/pipeline/stream_processor.py",
    "src/preprocessing/bot_detector.py",
    "src/preprocessing/ner_extractor.py",
    "src/sentiment/lexicon_analyzer.py",
    "src/utils/model_retrainer.py",
]

base_dir = Path(__file__).parent
errors = []

for file in files_to_check:
    filepath = base_dir / file
    if filepath.exists():
        try:
            py_compile.compile(str(filepath), doraise=True)
            print(f"✓ {file}")
        except py_compile.PyCompileError as e:
            print(
                f"✗ {file}: {e.msg} at line {e.exc_value.lineno if hasattr(e.exc_value, 'lineno') else '?'}"
            )
            errors.append((file, str(e)))
    else:
        print(f"! File not found: {file}")

print(f"\n{'='*60}")
print(f"Files with errors: {len(errors)}")
if errors:
    print("\nErrors summary:")
    for file, error in errors:
        print(f"  - {file}")
print(f"{'='*60}")
