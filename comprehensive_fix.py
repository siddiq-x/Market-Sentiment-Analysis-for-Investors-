#!/usr/bin/env python3
"""
Comprehensive Python syntax fixer for broken multi-line strings.
"""
import re
from pathlib import Path
from typing import List

def fix_broken_file(filepath: Path) -> bool:
    """Fix a single Python file by joining broken lines."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        fixed_lines = []
        i = 0
        changes_made = False
        
        while i < len(lines):
            line = lines[i]
            
            # Check if line has unterminated string (ends with quote but not closed)
            # Pattern: line ends with text" or text' but has unmatched quotes
            stripped = line.rstrip()
            
            # Count quotes to see if they're balanced
            single_quotes = stripped.count("'") - stripped.count("\\'")
            double_quotes = stripped.count('"') - stripped.count('\\"')
            
            # Check for f-strings
            f_string_double = stripped.count('f"') + stripped.count('f\'')
            
            # If we have unbalanced quotes OR line ends mid-string
            needs_join = False
            
            # Pattern 1: Line clearly breaks in middle of string (no closing quote at end)
            if re.search(r'["\']([^"\']*?)$', stripped) and not stripped.endswith(('"""', "'''")):
                # Check if this quote is actually unclosed
                # Simple heuristic: if there's an odd number of quotes, it's probably broken
                if (single_quotes % 2 == 1) or (double_quotes % 2 == 1):
                    needs_join = True
            
            # Pattern 2: Type annotation broken across lines (-> \n Type:)
            if stripped.endswith(' ->') or stripped.endswith(')->'):
                needs_join = True
            
            # Pattern 3: List/dict broken with 'and\n' or ',\n    '
            if stripped.endswith(' and') or (stripped.endswith(',') and i + 1 < len(lines) and lines[i+1].strip() and not lines[i+1].strip().startswith(('#', ')', ']', '}'))):
                # Check if next line starts with just a word (continuation)
                if i + 1 < len(lines):
                    next_stripped = lines[i+1].strip()
                    if next_stripped and not next_stripped[0] in '([{#\'\"' and not any(next_stripped.startswith(kw) for kw in ['def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'except', 'finally:', 'with ', 'return ', 'yield ', 'import ', 'from ']):
                        # This might be a continuation
                        if len(lines[i+1]) - len(lines[i+1].lstrip()) > len(line) - len(line.lstrip()) + 2:
                            needs_join = True
            
            if needs_join and i + 1 < len(lines):
                # Join with next line
                next_line = lines[i + 1]
                # Remove trailing newline from current, strip leading spaces from next
                joined = line.rstrip() + ' ' + next_line.lstrip()
                fixed_lines.append(joined)
                i += 2  # Skip the next line since we merged it
                changes_made = True
            else:
                fixed_lines.append(line)
                i += 1
        
        if changes_made:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(fixed_lines)
            return True
        return False
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def main():
    """Fix all problematic Python files."""
    files_to_fix = [
        'main.py',
        'src/dashboard/app.py',
        'src/data_ingestion/ingestion_manager.py',
        'src/data_ingestion/market_connector.py',
        'src/data_ingestion/news_connector.py',
        'src/data_ingestion/social_connector.py',
        'src/fusion/feature_engineer.py',
        'src/fusion/fusion_manager.py',
        'src/fusion/lstm_model.py',
        'src/pipeline/stream_processor.py',
        'src/preprocessing/bot_detector.py',
        'src/preprocessing/ner_extractor.py',
        'src/preprocessing/text_cleaner.py',
        'src/sentiment/ensemble_analyzer.py',
        'src/sentiment/finbert_analyzer.py',
        'src/sentiment/lexicon_analyzer.py',
        'src/utils/model_retrainer.py',
        'tests/conftest.py',
    ]
    
    base_dir = Path(__file__).parent
    fixed_count = 0
    
    for file_path in files_to_fix:
        full_path = base_dir / file_path
        if full_path.exists():
            print(f"Processing: {file_path}...")
            if fix_broken_file(full_path):
                fixed_count += 1
                print(f"  ✓ Fixed")
            else:
                print(f"  - No changes needed")
        else:
            print(f"  ! Not found: {file_path}")
    
    print(f"\n{'='*60}")
    print(f"Fixed {fixed_count} files")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
