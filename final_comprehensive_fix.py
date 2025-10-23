#!/usr/bin/env python3
"""
Final comprehensive fix - Find and fix ALL broken strings in all files.
Uses a more aggressive pattern matching approach.
"""
from pathlib import Path
import re

def fix_file_completely(filepath):
    """Fix all broken strings in a file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    fixed_lines = []
    i = 0
    changes = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check if line ends with incomplete string
        if i < len(lines) - 1:
            next_line = lines[i + 1]
            
            # Pattern 1: f"text\n    text"
            if re.search(r'f"[^"]*$', line) and re.match(r'^\s+[^"]*"', next_line):
                combined = line.rstrip('\n') + ' ' + next_line.lstrip()
                fixed_lines.append(combined)
                i += 2
                changes += 1
                continue
            
            # Pattern 2: "text\n    text"
            if re.search(r'"[^"]*$', line) and not line.rstrip().endswith('\\') and re.match(r'^\s+[^"]*"', next_line):
                combined = line.rstrip('\n') + ' ' + next_line.lstrip()
                fixed_lines.append(combined)
                i += 2
                changes += 1
                continue
            
            # Pattern 3: Broken type hint: ) ->\n    Type:
            if re.search(r'\)\s*->\s*$', line) and re.match(r'^\s+[A-Z].*:', next_line):
                combined = line.rstrip('\n') + ' ' + next_line.lstrip()
                fixed_lines.append(combined)
                i += 2
                changes += 1
                continue
            
            # Pattern 4: Broken assignment: var =\n    value
            if re.search(r'=\s*$', line) and re.match(r'^\s+[a-zA-Z_]', next_line):
                combined = line.rstrip('\n') + ' ' + next_line.lstrip()
                fixed_lines.append(combined)
                i += 2
                changes += 1
                continue
            
            # Pattern 5: Broken list/call: something(\n    arg)
            if re.search(r'\(\s*$', line) and re.match(r'^\s+[a-zA-Z_].*\)', next_line):
                combined = line.rstrip('\n').rstrip() + next_line.lstrip()
                fixed_lines.append(combined)
                i += 2
                changes += 1
                continue
        
        fixed_lines.append(line)
        i += 1
    
    if changes > 0:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(fixed_lines)
        return changes
    return 0

# Process all problematic files
base_dir = Path(__file__).parent
files = [
    'src/fusion/feature_engineer.py',
    'src/fusion/fusion_manager.py',
    'src/fusion/lstm_model.py',
    'src/pipeline/stream_processor.py',
    'src/preprocessing/bot_detector.py',
    'src/preprocessing/ner_extractor.py',
    'src/sentiment/lexicon_analyzer.py',
    'src/utils/model_retrainer.py',
]

total_fixed = 0
total_changes = 0

for file_rel in files:
    filepath = base_dir / file_rel
    if filepath.exists():
        changes = fix_file_completely(filepath)
        if changes > 0:
            total_fixed += 1
            total_changes += changes
            print(f"✓ {file_rel}: {changes} fixes")
        else:
            print(f"- {file_rel}: No changes")
    else:
        print(f"✗ {file_rel}: Not found")

print(f"\n{'='*70}")
print(f"Fixed {total_fixed} files with {total_changes} total changes")
print(f"{'='*70}")
