"""
Flask-based dashboard for Market Sentiment Analysis System
"""
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
import pandas as pd
import plotly.graph_objs as go
import plotly.utils
import numpy as np

from ..pipeline.stream_processor import StreamProcessor
from ..data_ingestion.ingestion_manager import ingestion_manager
from ..sentiment.ensemble_analyzer import EnsembleSentimentAnalyzer
from ..fusion.fusion_manager import FusionManager
from config.config import config

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'market_sentiment_secret_key'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize components
stream_processor = StreamProcessor()
sentiment_analyzer = EnsembleSentimentAnalyzer()
fusion_manager = FusionManager()

# Global state
dashboard_state = {
    "stream_processor_running": False,
    "last_update": datetime.now(),
    "active_tickers": config.monitored_tickers[:10]  # Limit for dashboard
}

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html',
                         tickers=dashboard_state["active_tickers"],
                         config=config.__dict__)

@app.route('/api/status')
def get_status():
    """Get system status"""
    try:
        # Get connector status
        connector_status = ingestion_manager.get_connector_status()

        # Get stream processor stats
        stream_stats = stream_processor.get_processing_stats()

        # Get fusion manager status
        fusion_status = fusion_manager.get_model_status()

        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "connectors": connector_status,
            "stream_processor": stream_stats,
            "fusion_models": fusion_status,
            "active_tickers": dashboard_state["active_tickers"]
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/sentiment/<ticker>')
def get_sentiment_data(ticker):
    """Get sentiment data for a specific ticker"""
    try:
        hours_back = request.args.get('hours', 24, type=int)

        # Get recent sentiment data
        sentiment_data = stream_processor.get_recent_sentiment_data(ticker,
    hours_back)

        if not sentiment_data:
            return jsonify({"message": "No sentiment data available", "data":
    []})

        # Process data for visualization
        processed_data = []
        for item in sentiment_data:
            processed_data.append({
                "timestamp": item["timestamp"],
                "sentiment": item["sentiment"],
                "confidence": item["confidence"],
                "ensemble_score": item["ensemble_score"]
            })

        return jsonify({
            "ticker": ticker,
            "data": processed_data,
            "count": len(processed_data)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/predictions')
def get_predictions():
    """Get current predictions for all active tickers"""
    try:
        predictions = {}

        for ticker in dashboard_state["active_tickers"]:
            prediction = stream_processor.trigger_fusion_prediction(ticker)
            if prediction:
                predictions[ticker] = prediction

        return jsonify(predictions)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/market-data/<ticker>')
def get_market_data(ticker):
    """Get market data for a specific ticker"""
    try:
        # Fetch recent market data
        market_data = ingestion_manager.fetch_by_connector(
            'market',
            tickers=[ticker],
            period="1d",
            interval="1h"
        )

        if not market_data:
            return jsonify({"message": "No market data available", "data": []})

        # Process data for visualization
        processed_data = []
        for data_point in market_data[-24:]:  # Last 24 hours
            if data_point.metadata.get('data_type') == 'stock_price':
                processed_data.append({
                    "timestamp": data_point.timestamp.isoformat(),
                    "open": data_point.metadata.get('open'),
                    "high": data_point.metadata.get('high'),
                    "low": data_point.metadata.get('low'),
                    "close": data_point.metadata.get('close'),
                    "volume": data_point.metadata.get('volume')
                })

        return jsonify({
            "ticker": ticker,
            "data": processed_data,
            "count": len(processed_data)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/sentiment-chart/<ticker>')
def get_sentiment_chart(ticker):
    """Generate sentiment chart for a ticker"""
    try:
        hours_back = request.args.get('hours', 24, type=int)
        sentiment_data = stream_processor.get_recent_sentiment_data(ticker,
    hours_back)

        if not sentiment_data:
            return jsonify({"error": "No data available"})

        # Prepare data for Plotly
        timestamps = [item["timestamp"] for item in sentiment_data]
        ensemble_scores = [item["ensemble_score"] for item in sentiment_data]
        confidences = [item["confidence"] for item in sentiment_data]

        # Create Plotly figure
        fig = go.Figure()

        # Add sentiment score line
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=ensemble_scores,
            mode='lines+markers',
            name='Sentiment Score',
            line=dict(color='blue', width=2),
            hovertemplate='<b>%{fullData.name}</b><br>' +
                         'Time: %{x}<br>' +
                         'Score: %{y:.3f}<br>' +
                         '<extra></extra>'
        ))

        # Add confidence as secondary y-axis
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=confidences,
            mode='lines',
            name='Confidence',
            line=dict(color='orange', width=1, dash='dash'),
            yaxis='y2',
            hovertemplate='<b>%{fullData.name}</b><br>' +
                         'Time: %{x}<br>' +
                         'Confidence: %{y:.3f}<br>' +
                         '<extra></extra>'
        ))

        # Update layout
        fig.update_layout(
            title=f'Sentiment Analysis - {ticker}',
            xaxis_title='Time',
            yaxis=dict(
                title='Sentiment Score',
                side='left',
                range=[-1, 1]
            ),
            yaxis2=dict(
                title='Confidence',
                side='right',
                overlaying='y',
                range=[0, 1]
            ),
            hovermode='x unified',
            template='plotly_white',
            height=400
        )

        # Add horizontal lines for sentiment thresholds
        fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
        fig.add_hline(y=0.1, line_dash="dot", line_color="green", opacity=0.3)
        fig.add_hline(y=-0.1, line_dash="dot", line_color="red", opacity=0.3)

        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return jsonify({"chart": graphJSON})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/market-chart/<ticker>')
def get_market_chart(ticker):
    """Generate market chart for a ticker"""
    try:
        # Get market data
        market_data = ingestion_manager.fetch_by_connector(
            'market',
            tickers=[ticker],
            period="1d",
            interval="1h"
        )

        if not market_data:
            return jsonify({"error": "No market data available"})

        # Process data
        price_data = []
        for data_point in market_data[-24:]:  # Last 24 hours
            if data_point.metadata.get('data_type') == 'stock_price':
                price_data.append({
                    "timestamp": data_point.timestamp,
                    "open": data_point.metadata.get('open'),
                    "high": data_point.metadata.get('high'),
                    "low": data_point.metadata.get('low'),
                    "close": data_point.metadata.get('close'),
                    "volume": data_point.metadata.get('volume')
                })

        if not price_data:
            return jsonify({"error": "No valid price data"})

        # Create candlestick chart
        fig = go.Figure()

        fig.add_trace(go.Candlestick(
            x=[item["timestamp"] for item in price_data],
            open=[item["open"] for item in price_data],
            high=[item["high"] for item in price_data],
            low=[item["low"] for item in price_data],
            close=[item["close"] for item in price_data],
            name=ticker,
            increasing_line_color='green',
            decreasing_line_color='red'
        ))

        fig.update_layout(
            title=f'Price Chart - {ticker}',
            xaxis_title='Time',
            yaxis_title='Price ($)',
            template='plotly_white',
            height=400,
            xaxis_rangeslider_visible=False
        )

        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return jsonify({"chart": graphJSON})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/start-stream')
def start_stream():
    """Start the stream processor"""
    try:
        if not dashboard_state["stream_processor_running"]:
            stream_processor.start()
            dashboard_state["stream_processor_running"] = True
            return jsonify({"status": "started", "message": "Stream processor
    started successfully"})
        else:
            return jsonify({"status": "already_running", "message": "Stream
    processor is already running"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/stop-stream')
def stop_stream():
    """Stop the stream processor"""
    try:
        if dashboard_state["stream_processor_running"]:
            stream_processor.stop()
            dashboard_state["stream_processor_running"] = False
            return jsonify({"status": "stopped", "message": "Stream processor
    stopped successfully"})
        else:
            return jsonify({"status": "not_running", "message": "Stream
    processor is not running"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/processing-results')
def get_processing_results():
    """Get recent processing results"""
    try:
        results = stream_processor.get_processed_results(max_results=50)

        processed_results = []
        for result in results:
            processed_results.append({
                "message_id": result.message_id,
                "source": result.original_data.source,
                "ticker": result.original_data.ticker,
                "content_preview": result.original_data.content[:100] + "..."
    if len(result.original_data.content) > 100 else
    result.original_data.content,
                "sentiment": result.sentiment_result.get("sentiment") if
    result.sentiment_result else None,
                "confidence": result.sentiment_result.get("confidence") if
    result.sentiment_result else None,
                "processing_time_ms": result.processing_time_ms,
                "timestamp": result.timestamp.isoformat(),
                "entities_count": len(result.entities)
            })

        return jsonify({
            "results": processed_results,
            "count": len(processed_results)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/explainability/<ticker>')
def get_explainability_data(ticker):
    """Get explainability data for predictions"""
    try:
        # This would integrate with SHAP for real explainability
        # For now, return mock explainability data

        # Get recent prediction
        prediction = stream_processor.trigger_fusion_prediction(ticker)

        if not prediction:
            return jsonify({"error": "No prediction available"})

        # Mock SHAP-like feature importance
        features = {
            "sentiment_trend": np.random.uniform(-0.3, 0.3),
            "price_momentum": np.random.uniform(-0.2, 0.2),
            "volume_trend": np.random.uniform(-0.1, 0.1),
            "volatility": np.random.uniform(-0.15, 0.15),
            "news_sentiment": np.random.uniform(-0.25, 0.25),
            "social_sentiment": np.random.uniform(-0.2, 0.2),
            "market_hours": np.random.uniform(-0.05, 0.05),
            "sector_performance": np.random.uniform(-0.1, 0.1)
        }

        # Create SHAP-like waterfall data
        base_value = 0.0
        feature_contributions = []
        cumulative = base_value

        for feature, contribution in features.items():
            feature_contributions.append({
                "feature": feature.replace("_", " ").title(),
                "contribution": contribution,
                "cumulative": cumulative + contribution
            })
            cumulative += contribution

        return jsonify({
            "ticker": ticker,
            "prediction": prediction,
            "base_value": base_value,
            "final_value": cumulative,
            "feature_contributions": feature_contributions,
            "explanation": f"The model predicts {['bearish', 'neutral',
    'bullish'][prediction['prediction'] + 1]} sentiment for {ticker} based on
    the feature contributions shown above."
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# WebSocket events for real-time updates
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    emit('status', {'message': 'Connected to Market Sentiment Analysis
    Dashboard'})

@socketio.on('subscribe_ticker')
def handle_subscribe_ticker(data):
    """Handle ticker subscription for real-time updates"""
    ticker = data.get('ticker')
    if ticker in dashboard_state["active_tickers"]:
        # Join ticker-specific room
        from flask_socketio import join_room
        join_room(f"ticker_{ticker}")
        emit('subscribed', {'ticker': ticker, 'message': f'Subscribed to
    {ticker} updates'})

# Background task for real-time updates (would be implemented with proper task queue)
def background_updates():
    """Send real-time updates to connected clients"""
    while True:
        try:
            # Get latest processing stats
            stats = stream_processor.get_processing_stats()
            socketio.emit('processing_stats', stats)

            # Get latest predictions for active tickers
            for ticker in dashboard_state["active_tickers"]:
                prediction = stream_processor.trigger_fusion_prediction(ticker)
                if prediction:
                    socketio.emit('prediction_update', {
                        'ticker': ticker,
                        'prediction': prediction
                    }, room=f"ticker_{ticker}")

            socketio.sleep(5)  # Update every 5 seconds

        except Exception as e:
            app.logger.error(f"Error in background updates: {str(e)}")
            socketio.sleep(10)

if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Start background task
    socketio.start_background_task(background_updates)

    # Run the app
    socketio.run(app,
                host=config.dashboard.host,
                port=config.dashboard.port,
                debug=config.dashboard.debug)
