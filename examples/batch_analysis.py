# Add src to path before any imports
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

"""
Batch analysis example for processing historical data
"""
import json  # noqa: E402
from datetime import datetime  # noqa: E402

import pandas as pd  # noqa: E402

from data_ingestion.ingestion_manager import IngestionManager  # noqa: E402
from preprocessing.bot_detector import BotDetector  # noqa: E402
from preprocessing.ner_extractor import NERExtractor  # noqa: E402
from preprocessing.text_cleaner import TextCleaner  # noqa: E402
from sentiment.ensemble_analyzer import EnsembleSentimentAnalyzer  # noqa: E402


def batch_sentiment_analysis(tickers, days_back=7):
    """
    Perform batch sentiment analysis for multiple tickers

    Args:
        tickers: List of stock tickers to analyze
        days_back: Number of days of historical data to analyze
    """
    print(
        f"Starting batch analysis for {len(tickers)} tickers over " f"{days_back} days"
    )

    # Initialize components
    ingestion_manager = IngestionManager()
    sentiment_analyzer = EnsembleSentimentAnalyzer()
    text_cleaner = TextCleaner()
    ner_extractor = NERExtractor()
    bot_detector = BotDetector()

    results = {}

    for ticker in tickers:
        print(f"\nProcessing {ticker}...")

        try:
            # Fetch data
            data_points = ingestion_manager.fetch_all_data(
                [ticker], hours_back=days_back * 24
            )

            if not data_points:
                print(f"No data found for {ticker}")
                continue

            ticker_results = {
                "ticker": ticker,
                "total_data_points": len(data_points),
                "sentiment_distribution": {"bullish": 0, "neutral": 0, "bearish": 0},
                "source_breakdown": {},
                "credibility_stats": {"high": 0, "medium": 0, "low": 0},
                "bot_detection": {"human": 0, "suspicious": 0, "bot": 0},
                "entities": {},
                "daily_sentiment": {},
                "processed_texts": [],
            }

            for dp in data_points:
                # Clean text
                cleaned_text = text_cleaner.clean_text(dp.content)

                # Bot detection for social media content
                if dp.source in ["twitter", "reddit"]:
                    bot_score = bot_detector.detect_bot(
                        dp.content,
                        {
                            "author": dp.metadata.get("author", ""),
                            "timestamp": dp.timestamp,
                        },
                    )

                    if bot_score > 0.7:
                        ticker_results["bot_detection"]["bot"] += 1
                        continue  # Skip bot content
                    elif bot_score > 0.4:
                        ticker_results["bot_detection"]["suspicious"] += 1
                    else:
                        ticker_results["bot_detection"]["human"] += 1

                # Sentiment analysis
                sentiment_result = sentiment_analyzer.analyze_sentiment(
                    cleaned_text, context={"source": dp.source, "ticker": ticker}
                )

                # Update sentiment distribution
                if sentiment_result.sentiment == 1:
                    ticker_results["sentiment_distribution"]["bullish"] += 1
                elif sentiment_result.sentiment == -1:
                    ticker_results["sentiment_distribution"]["bearish"] += 1
                else:
                    ticker_results["sentiment_distribution"]["neutral"] += 1

                # Source breakdown
                source = dp.source
                if source not in ticker_results["source_breakdown"]:
                    ticker_results["source_breakdown"][source] = 0
                ticker_results["source_breakdown"][source] += 1

                # Credibility stats
                if dp.credibility_score >= 0.7:
                    ticker_results["credibility_stats"]["high"] += 1
                elif dp.credibility_score >= 0.4:
                    ticker_results["credibility_stats"]["medium"] += 1
                else:
                    ticker_results["credibility_stats"]["low"] += 1

                # Entity extraction
                entities = ner_extractor.extract_entities(cleaned_text)
                for entity_type, entity_list in entities.items():
                    if entity_type not in ticker_results["entities"]:
                        ticker_results["entities"][entity_type] = {}
                    for entity in entity_list:
                        entity_dict = ticker_results["entities"][entity_type]
                        if entity not in entity_dict:
                            entity_dict[entity] = 0
                        entity_dict[entity] += 1

                # Daily sentiment tracking
                day_key = dp.timestamp.strftime("%Y-%m-%d")
                if day_key not in ticker_results["daily_sentiment"]:
                    ticker_results["daily_sentiment"][day_key] = {
                        "scores": [],
                        "count": 0,
                        "avg_score": 0,
                    }
                ticker_results["daily_sentiment"][day_key]["scores"].append(
                    sentiment_result.ensemble_score
                )
                ticker_results["daily_sentiment"][day_key]["count"] += 1

                # Store processed text sample
                if len(ticker_results["processed_texts"]) < 10:
                    ticker_results["processed_texts"].append(
                        {
                            "original": dp.content[:200],
                            "cleaned": (
                                cleaned_text[:200] + "..."
                                if len(cleaned_text) > 200
                                else cleaned_text
                            ),
                            "sentiment": sentiment_result.overall_sentiment,
                            "score": sentiment_result.ensemble_score,
                            "source": dp.source,
                            "timestamp": dp.timestamp.isoformat(),
                        }
                    )

            # Calculate daily averages
            for day_data in ticker_results["daily_sentiment"].values():
                if day_data["scores"]:
                    scores = day_data["scores"]
                    day_data["avg_score"] = sum(scores) / len(scores)

            # Calculate overall sentiment score
            total_sentiment = ticker_results["sentiment_distribution"]
            total_count = sum(total_sentiment.values())

            if total_count > 0:
                ticker_results["overall_sentiment_score"] = (
                    total_sentiment["bullish"] - total_sentiment["bearish"]
                ) / total_count
                ticker_results["sentiment_confidence"] = (
                    max(total_sentiment.values()) / total_count
                )
            else:
                ticker_results["overall_sentiment_score"] = 0
                ticker_results["sentiment_confidence"] = 0

            results[ticker] = ticker_results

            print(
                f"Processed {ticker}: {total_count} data points, "
                f"Overall sentiment: {ticker_results['overall_sentiment_score']:.3f}"
            )

        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")
            results[ticker] = {"error": str(e)}

    return results


def generate_report(results, output_file=None):
    """Generate a comprehensive analysis report"""

    report = []
    report.append("MARKET SENTIMENT ANALYSIS BATCH REPORT")
    report.append("=" * 50)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Tickers analyzed: {len(results)}")
    report.append("")

    # Summary statistics
    report.append("SUMMARY STATISTICS")
    report.append("-" * 30)

    total_data_points = sum(
        r.get("total_data_points", 0) for r in results.values() if "error" not in r
    )

    valid_results = [r for r in results.values() if "error" not in r]
    avg_sentiment = (
        sum(r.get("overall_sentiment_score", 0) for r in valid_results)
        / len(valid_results)
        if valid_results
        else 0
    )

    report.append(f"Total data points processed: {total_data_points}")
    report.append(f"Average sentiment score: {avg_sentiment:.3f}")
    report.append("")

    # Individual ticker analysis
    report.append("INDIVIDUAL TICKER ANALYSIS")
    report.append("-" * 40)

    for ticker, data in results.items():
        if "error" in data:
            report.append(f"\n{ticker}: ERROR - {data['error']}")
            continue

        report.append(f"\n{ticker}:")
        report.append(f"  Data Points: {data['total_data_points']}")
        report.append(f"  Overall Sentiment: {data['overall_sentiment_score']:.3f}")
        report.append(f"  Confidence: {data['sentiment_confidence']:.3f}")

        # Sentiment distribution
        sentiment_dist = data["sentiment_distribution"]
        total = sum(sentiment_dist.values())
        if total > 0:
            report.append("  Sentiment Distribution:")
            report.append(
                f"    Bullish: {sentiment_dist['bullish']} "
                f"({sentiment_dist['bullish']/total*100:.1f}%)"
            )
            report.append(
                f"    Neutral: {sentiment_dist['neutral']} "
                f"({sentiment_dist['neutral']/total*100:.1f}%)"
            )
            report.append(
                f"    Bearish: {sentiment_dist['bearish']} "
                f"({sentiment_dist['bearish']/total*100:.1f}%)"
            )

        # Source breakdown
        if data["source_breakdown"]:
            report.append("  Top Sources:")
            sorted_sources = sorted(
                data["source_breakdown"].items(), key=lambda x: x[1], reverse=True
            )
            for source, count in sorted_sources[:3]:
                report.append(f"    {source}: {count}")

        # Top entities
        if data["entities"].get("ORGANIZATION"):
            top_orgs = sorted(
                data["entities"]["ORGANIZATION"].items(),
                key=lambda x: x[1],
                reverse=True,
            )[:3]
            report.append(
                f"  Top Organizations: {', '.join([org for org, _ in top_orgs])}"
            )

    # Market trends
    report.append("\nMARKET TRENDS")
    report.append("-" * 20)

    # Sort tickers by sentiment
    sorted_tickers = sorted(
        [(ticker, data) for ticker, data in results.items() if "error" not in data],
        key=lambda x: x[1]["overall_sentiment_score"],
        reverse=True,
    )

    if sorted_tickers:
        report.append("Most Bullish:")
        for ticker, data in sorted_tickers[:3]:
            report.append(f"  {ticker}: {data['overall_sentiment_score']:.3f}")

        report.append("Most Bearish:")
        for ticker, data in sorted_tickers[-3:]:
            report.append(f"  {ticker}: {data['overall_sentiment_score']:.3f}")

    report_text = "\n".join(report)

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"Report saved to {output_file}")

    return report_text


def export_results(results, format="json", filename=None):
    """Export results in various formats"""

    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sentiment_analysis_{timestamp}.{format}"

    if format == "json":
        # Convert datetime objects to strings for JSON serialization
        json_results = {}
        for ticker, data in results.items():
            if "error" in data:
                json_results[ticker] = data
                continue

            json_data = data.copy()
            # Convert daily sentiment timestamps
            if "daily_sentiment" in json_data:
                for day, day_data in json_data["daily_sentiment"].items():
                    day_data["scores"] = [float(score) for score in day_data["scores"]]

            json_results[ticker] = json_data

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(json_results, f, indent=2, default=str)

    elif format == "csv":
        # Create a flattened CSV
        csv_data = []
        for ticker, data in results.items():
            if "error" in data:
                continue

            row = {
                "ticker": ticker,
                "total_data_points": data["total_data_points"],
                "overall_sentiment_score": data["overall_sentiment_score"],
                "sentiment_confidence": data["sentiment_confidence"],
                "bullish_count": data["sentiment_distribution"]["bullish"],
                "neutral_count": data["sentiment_distribution"]["neutral"],
                "bearish_count": data["sentiment_distribution"]["bearish"],
                "high_credibility": data["credibility_stats"]["high"],
                "medium_credibility": data["credibility_stats"]["medium"],
                "low_credibility": data["credibility_stats"]["low"],
            }
            csv_data.append(row)

        df = pd.DataFrame(csv_data)
        df.to_csv(filename, index=False)

    print(f"Results exported to {filename}")
    return filename


def main():
    """Main batch analysis function"""
    # Example tickers to analyze
    tickers = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]

    print("Market Sentiment Analysis - Batch Processing")
    print("=" * 50)

    # Run batch analysis
    results = batch_sentiment_analysis(tickers, days_back=3)

    # Generate report
    report = generate_report(results)
    print("\n" + report)

    # Export results
    json_file = export_results(results, format="json")
    csv_file = export_results(results, format="csv")

    print("\nBatch analysis complete!")
    print(f"Results exported to: {json_file}, {csv_file}")


if __name__ == "__main__":
    main()
