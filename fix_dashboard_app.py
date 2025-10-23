#!/usr/bin/env python3
"""Fix all broken strings in dashboard/app.py"""
from pathlib import Path

filepath = Path(__file__).parent / "src/dashboard/app.py"

fixes = [
    # Line 309-310
    (
        'return jsonify({"status": "started", "message": "Stream processor\n    started successfully"})',
        'return jsonify({"status": "started", "message": "Stream processor started successfully"})',
    ),
    # Line 312-313
    (
        'return jsonify({"status": "already_running", "message": "Stream\n    processor is already running"})',
        'return jsonify({"status": "already_running", "message": "Stream processor is already running"})',
    ),
    # Line 324-325
    (
        'return jsonify({"status": "stopped", "message": "Stream processor\n    stopped successfully"})',
        'return jsonify({"status": "stopped", "message": "Stream processor stopped successfully"})',
    ),
    # Line 327-328
    (
        'return jsonify({"status": "not_running", "message": "Stream\n    processor is not running"})',
        'return jsonify({"status": "not_running", "message": "Stream processor is not running"})',
    ),
    # Line 404-407
    (
        "\"explanation\": f\"The model predicts {['bearish', 'neutral',\n    'bullish'][prediction['prediction'] + 1]} sentiment for {ticker} based on\n    the feature contributions shown above.\"",
        "\"explanation\": f\"The model predicts {['bearish', 'neutral', 'bullish'][prediction['prediction'] + 1]} sentiment for {ticker} based on the feature contributions shown above.\"",
    ),
    # Line 413
    (
        "emit('status', {'message': 'Connected to Market Sentiment Analysis\n    Dashboard'})",
        "emit('status', {'message': 'Connected to Market Sentiment Analysis Dashboard'})",
    ),
    # Line 423
    (
        "emit('subscribed', {'ticker': ticker, 'message': f'Subscribed to\n    {ticker} updates'})",
        "emit('subscribed', {'ticker': ticker, 'message': f'Subscribed to {ticker} updates'})",
    ),
]

with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

for old, new in fixes:
    if old in content:
        content = content.replace(old, new)
        print(f"✓ Fixed: {old[:50]}...")
    else:
        print(f"✗ Not found: {old[:50]}...")

with open(filepath, "w", encoding="utf-8") as f:
    f.write(content)

print("\n✓ app.py fixed!")
