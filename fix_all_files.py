#!/usr/bin/env python3
"""
Fix all broken multi-line strings in Python files by joining them.
"""
import re
from pathlib import Path


def fix_file_completely(filepath):
    """Read file, find and join all broken lines."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    original = content

    # Pattern 1: Broken f-strings and regular strings across lines
    # Matches: "text\n    more" or f"text\n    more"
    content = re.sub(r'(["\'])([^"\']*)\n\s+([^"\']+)\1', r"\1\2 \3\1", content)

    # Pattern 2: Broken type hints:  -> \n    Type:
    content = re.sub(r"\) ->\n\s+(\w+(?:\[[\w, ]+\])?):", r") -> \1:", content)

    # Pattern 3: Broken list continuation with 'and\n'
    content = re.sub(r" and\n\s+(\w+)", r" and \1", content)

    # Pattern 4: Function params across lines: ,\n    param
    content = re.sub(r",\n\s{4,}([a-zA-Z_])", r", \1", content)

    # Pattern 5: Broken strings with comments in between
    content = re.sub(
        r'(["\'])([^"\']*)\n\s+#[^\n]+\n\s+([^"\']+)\1', r"\1\2 \3\1", content
    )

    if content != original:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    return False


# Files to fix
files = [
    "src/fusion/feature_engineer.py",
    "src/fusion/fusion_manager.py",
    "src/fusion/lstm_model.py",
    "src/pipeline/stream_processor.py",
    "src/preprocessing/bot_detector.py",
    "src/preprocessing/ner_extractor.py",
    "src/sentiment/lexicon_analyzer.py",
    "src/utils/model_retrainer.py",
]

base = Path(__file__).parent
fixed = 0

for file in files:
    filepath = base / file
    if filepath.exists():
        print(f"Fixing: {file}...")
        if fix_file_completely(filepath):
            fixed += 1
            print("  ✓ Fixed")
        else:
            print("  - No automatic fixes")

print(f"\n{'='*60}")
print(f"Auto-fixed {fixed} files")
print(f"{'='*60}")
