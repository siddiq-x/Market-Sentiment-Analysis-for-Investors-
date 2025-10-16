"""
Quick start example for the Market Sentiment Analysis System
This script demonstrates basic usage of the system components
"""
import sys
import os
from datetime import datetime, timedelta
import asyncio

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_ingestion.ingestion_manager import IngestionManager
from sentiment.ensemble_analyzer import EnsembleSentimentAnalyzer
from fusion.fusion_manager import FusionManager
from preprocessing.text_cleaner import TextCleaner
from preprocessing.ner_extractor import NERExtractor

def basic_sentiment_analysis():
    """Demonstrate basic sentiment analysis"""
    print("=== Basic Sentiment Analysis Demo ===")

    # Initialize sentiment analyzer
    analyzer = EnsembleSentimentAnalyzer()

    # Sample financial texts
    sample_texts = [
        "Apple reports record quarterly earnings, beating analyst
    expectations",
        "Tesla stock plummets amid production concerns and supply chain
    issues",
        "Microsoft maintains steady growth in cloud computing segment",
        "Market volatility increases due to geopolitical tensions",
        "Strong GDP growth signals economic recovery"
    ]

    print("\nAnalyzing sample financial texts:")
    print("-" * 50)

    for text in sample_texts:
        result = analyzer.analyze_sentiment(text)
        sentiment_label = {-1: "Bearish", 0: "Neutral", 1:
    "Bullish"}[result.sentiment]

        print(f"Text: {text[:60]}...")
        print(f"Sentiment: {sentiment_label} (Score:
    {result.ensemble_score:.3f}, Confidence: {result.confidence:.3f})")
        print()

def data_ingestion_demo():
    """Demonstrate data ingestion (simulated)"""
    print("=== Data Ingestion Demo ===")

    # Initialize ingestion manager
    manager = IngestionManager()

    # Note: This requires API keys to be set in .env file
    print("Attempting to fetch data for AAPL...")
    print("(This requires valid API keys in .env file)")

    try:
        # Fetch data for the last 6 hours
        data_points = manager.fetch_all_data(['AAPL'], hours_back=6)

        if data_points:
            print(f"Successfully fetched {len(data_points)} data points")

            # Show sample data points
            for i, dp in enumerate(data_points[:3]):
                print(f"\nData Point {i+1}:")
                print(f"Source: {dp.source}")
                print(f"Ticker: {dp.ticker}")
                print(f"Content: {dp.content[:100]}...")
                print(f"Credibility: {dp.credibility_score:.2f}")
        else:
            print("No data points fetched (check API keys and network
    connection)")

    except Exception as e:
        print(f"Error during data ingestion: {str(e)}")
        print("This is expected if API keys are not configured")

def text_preprocessing_demo():
    """Demonstrate text preprocessing"""
    print("=== Text Preprocessing Demo ===")

    # Initialize preprocessors
    cleaner = TextCleaner()
    ner_extractor = NERExtractor()

    # Sample messy financial text
    messy_text = """
    OMG!!! $AAPL is going TO THE MOON 🚀🚀🚀
    Check this out: https://example.com/news
    CEO Tim Cook says Q4 earnings will be AMAZING!!!
    #Apple #Bullish @everyone
    """

    print("Original text:")
    print(messy_text)
    print("\n" + "-" * 50)

    # Clean the text
    cleaned_text = cleaner.clean_text(messy_text)
    print("Cleaned text:")
    print(cleaned_text)

    # Extract entities
    entities = ner_extractor.extract_entities(cleaned_text)
    print(f"\nExtracted entities:")
    for entity_type, entity_list in entities.items():
        if entity_list:
            print(f"{entity_type}: {', '.join(entity_list)}")

def fusion_model_demo():
    """Demonstrate fusion model (training simulation)"""
    print("=== Fusion Model Demo ===")

    # Initialize fusion manager
    fusion_manager = FusionManager()

    print("Fusion model initialized")
    print("Note: Actual training requires historical market data and sentiment
    data")
    print("This demo shows the model architecture and capabilities")

    # Show model summary
    try:
        model_info = fusion_manager.get_model_summary()
        print(f"\nModel Summary:")
        print(f"Input features: {model_info.get('input_features', 'N/A')}")
        print(f"Model type: {model_info.get('model_type', 'LSTM with
    Attention')}")
        print(f"Parameters: {model_info.get('total_params', 'N/A')}")
    except Exception as e:
        print(f"Model summary not available: {str(e)}")

def end_to_end_demo():
    """Demonstrate end-to-end pipeline"""
    print("=== End-to-End Pipeline Demo ===")

    # Sample pipeline workflow
    sample_news = "Apple Inc. reported strong quarterly results with revenue
    growth of 15%"

    print(f"Processing: {sample_news}")
    print("\nPipeline steps:")

    # Step 1: Text cleaning
    cleaner = TextCleaner()
    cleaned = cleaner.clean_text(sample_news)
    print(f"1. Text Cleaning: {cleaned}")

    # Step 2: Entity extraction
    ner = NERExtractor()
    entities = ner.extract_entities(cleaned)
    print(f"2. Entity Extraction: {entities}")

    # Step 3: Sentiment analysis
    analyzer = EnsembleSentimentAnalyzer()
    sentiment = analyzer.analyze_sentiment(cleaned)
    sentiment_label = {-1: "Bearish", 0: "Neutral", 1:
    "Bullish"}[sentiment.sentiment]
    print(f"3. Sentiment Analysis: {sentiment_label} (Score:
    {sentiment.ensemble_score:.3f})")

    # Step 4: Feature preparation (simulated)
    print("4. Feature Engineering: Combining sentiment with market data...")

    # Step 5: Prediction (simulated)
    print("5. Price Movement Prediction: [Simulated - requires trained model]")

def main():
    """Main demo function"""
    print("Market Sentiment Analysis System - Quick Start Demo")
    print("=" * 60)

    demos = [
        ("Basic Sentiment Analysis", basic_sentiment_analysis),
        ("Text Preprocessing", text_preprocessing_demo),
        ("Data Ingestion", data_ingestion_demo),
        ("Fusion Model", fusion_model_demo),
        ("End-to-End Pipeline", end_to_end_demo)
    ]

    for i, (name, demo_func) in enumerate(demos, 1):
        print(f"\n{i}. {name}")
        print("=" * 60)

        try:
            demo_func()
        except Exception as e:
            print(f"Demo error: {str(e)}")
            print("This may be due to missing dependencies or API keys")

        print("\n" + "=" * 60)

        if i < len(demos):
            input("Press Enter to continue to next demo...")

if __name__ == "__main__":
    main()
