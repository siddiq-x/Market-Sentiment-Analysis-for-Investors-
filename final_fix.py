#!/usr/bin/env python3
"""
Final comprehensive fix for all broken multi-line strings.
Reads files line-by-line and joins broken strings carefully.
"""
from pathlib import Path


def fix_file_line_by_line(filepath):
    """Fix file by processing line by line."""
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    fixed_lines = []
    i = 0
    changes = 0

    while i < len(lines):
        line = lines[i]

        # Check if line ends with an unterminated string
        stripped = line.rstrip()

        # Pattern 1: Line ends with -> (broken type hint)
        if stripped.endswith(" ->") or stripped.endswith(")->"):
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                # Join: remove newline, add space, strip next line
                fixed_line = line.rstrip() + " " + next_line.lstrip()
                fixed_lines.append(fixed_line)
                i += 2
                changes += 1
                continue

        # Pattern 2: Line has unterminated f-string or string
        # Check for unmatched quotes (simple heuristic)
        # Count quotes (excluding escaped ones)
        temp = stripped.replace('\\"', "").replace("\\'", "")
        double_q = temp.count('"')
        single_q = temp.count("'")

        # If odd number of quotes, likely broken
        if (double_q % 2 == 1 or single_q % 2 == 1) and i + 1 < len(lines):
            next_line = lines[i + 1]
            # Check if next line is indented more (continuation)
            curr_indent = len(line) - len(line.lstrip())
            next_indent = len(next_line) - len(next_line.lstrip())

            if next_indent > curr_indent + 2:
                # Join them
                fixed_line = line.rstrip() + " " + next_line.lstrip()
                fixed_lines.append(fixed_line)
                i += 2
                changes += 1
                continue

        # Pattern 3: Line ends with 'and' (broken condition)
        if stripped.endswith(" and") and i + 1 < len(lines):
            next_line = lines[i + 1]
            curr_indent = len(line) - len(line.lstrip())
            next_indent = len(next_line) - len(next_line.lstrip())

            if next_indent > curr_indent:
                fixed_line = line.rstrip() + " " + next_line.lstrip()
                fixed_lines.append(fixed_line)
                i += 2
                changes += 1
                continue

        # Pattern 4: Line ends with comma and next line is heavily indented
        if stripped.endswith(",") and i + 1 < len(lines):
            next_line = lines[i + 1]
            curr_indent = len(line) - len(line.lstrip())
            next_indent = len(next_line) - len(next_line.lstrip())
            next_stripped = next_line.strip()

            # If next line starts with a continuation (not a new statement)
            if (
                next_indent > curr_indent + 8
                and next_stripped
                and next_stripped[0] not in "([{#"
            ):
                # Check it's not a keyword
                keywords = [
                    "def ",
                    "class ",
                    "if ",
                    "for ",
                    "while ",
                    "try:",
                    "except",
                    "finally:",
                    "with ",
                    "return ",
                    "import ",
                    "from ",
                ]
                if not any(next_stripped.startswith(kw) for kw in keywords):
                    fixed_line = line.rstrip() + " " + next_line.lstrip()
                    fixed_lines.append(fixed_line)
                    i += 2
                    changes += 1
                    continue

        # No changes needed
        fixed_lines.append(line)
        i += 1

    if changes > 0:
        with open(filepath, "w", encoding="utf-8") as f:
            f.writelines(fixed_lines)
        return True, changes
    return False, 0


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
total_fixed = 0
total_changes = 0

for file in files:
    filepath = base / file
    if filepath.exists():
        print(f"Processing: {file}...")
        fixed, changes = fix_file_line_by_line(filepath)
        if fixed:
            total_fixed += 1
            total_changes += changes
            print(f"  ✓ Fixed {changes} broken lines")
        else:
            print("  - No issues found")

print(f"\n{'='*60}")
print(f"Fixed {total_fixed} files ({total_changes} total line joins)")
print(f"{'='*60}")
