#!/usr/bin/env python3
"""
Automatically fix broken multi-line strings and type annotations in Python files.
"""
import re
import os
from pathlib import Path

def fix_broken_strings(content):
    """Fix strings that are broken across lines."""
    # Pattern 1: f-strings or regular strings broken across lines
    # Match: "text\n    more text"
    pattern1 = r'(["\'])([^"\']*)\n\s+([^"\']*)\1'
    content = re.sub(pattern1, r'\1\2 \3\1', content)
    
    # Pattern 2: Type annotations broken: ) ->\n    Type:
    pattern2 = r'\) ->\n\s+(\w+(?:\[[\w, ]+\])?):' 
    content = re.sub(pattern2, r') -> \1:', content)
    
    # Pattern 3: List comprehensions broken: condition and\n    condition
    pattern3 = r' and\n\s+'
    content = re.sub(pattern3, ' and ', content)
    
    # Pattern 4: Function calls broken: function(\n    arg)
    pattern4 = r',\n\s{4,}([a-zA-Z_]\w+)'
    content = re.sub(pattern4, r', \1', content, flags=re.MULTILINE)
    
    return content

def fix_file(filepath):
    """Fix a single Python file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        content = fix_broken_strings(content)
        
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✓ Fixed: {filepath}")
            return True
        else:
            print(f"- No changes: {filepath}")
            return False
    except Exception as e:
        print(f"✗ Error fixing {filepath}: {e}")
        return False

def main():
    """Fix all Python files in the project."""
    files_to_fix = [
        'src/dashboard/app.py',
        'src/fusion/feature_engineer.py',
        'src/fusion/fusion_manager.py',
        'src/fusion/lstm_model.py',
        'src/pipeline/stream_processor.py',
        'src/preprocessing/bot_detector.py',
        'src/preprocessing/ner_extractor.py',
        'src/sentiment/lexicon_analyzer.py',
        'src/utils/model_retrainer.py',
    ]
    
    base_dir = Path(__file__).parent
    fixed_count = 0
    
    for file in files_to_fix:
        filepath = base_dir / file
        if filepath.exists():
            if fix_file(filepath):
                fixed_count += 1
        else:
            print(f"! File not found: {filepath}")
    
    print(f"\n{'='*50}")
    print(f"Fixed {fixed_count} files")
    print(f"{'='*50}")

if __name__ == '__main__':
    main()
