"""
Fusion manager orchestrating multimodal sentiment and market data fusion
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass

from .feature_engineer import MultimodalFeatureEngineer, FeatureSet
from .lstm_model import MultimodalFusionEngine, ModelConfig, TrainingConfig, PredictionResult
from ..data_ingestion.base_connector import DataPoint
from ..sentiment.ensemble_analyzer import EnsembleSentimentResult

@dataclass
class FusionPrediction:
    """Container for fusion prediction results"""
    ticker: str
    prediction: int  # -1: bearish, 0: neutral, 1: bullish
    confidence: float
    probability_distribution: Dict[str, float]
    contributing_factors: Dict[str, float]
    timestamp: datetime
    horizon_hours: int

class FusionManager:
    """Main manager for multimodal fusion pipeline"""
    
    def __init__(self, 
                 lookback_window: int = 24,
                 prediction_horizon: int = 4,
                 retrain_threshold: float = 0.1):
        """
        Initialize fusion manager
        
        Args:
            lookback_window: Hours of historical data to use
            prediction_horizon: Hours ahead to predict
            retrain_threshold: Performance threshold for retraining
        """
        self.logger = logging.getLogger("fusion_manager")
        
        self.lookback_window = lookback_window
        self.prediction_horizon = prediction_horizon
        self.retrain_threshold = retrain_threshold
        
        # Initialize components
        self.feature_engineer = MultimodalFeatureEngineer(
            lookback_window=lookback_window,
            prediction_horizon=prediction_horizon
        )
        
        self.fusion_engine = MultimodalFusionEngine()
        
        # Model performance tracking
        self.performance_history = []
        self.last_training_time = None
        
        # Cached models per ticker
        self.ticker_models = {}
        
        self.logger.info(f"Initialized fusion manager - lookback: {lookback_window}h, horizon: {prediction_horizon}h")
    
    def prepare_training_data(self, 
                            sentiment_results: List[EnsembleSentimentResult],
                            market_data_points: List[DataPoint],
                            target_ticker: str) -> FeatureSet:
        """
        Prepare training data from sentiment and market data
        
        Args:
            sentiment_results: List of sentiment analysis results
            market_data_points: List of market data points
            target_ticker: Ticker symbol to prepare data for
        """
        # Convert sentiment results to DataFrame
        sentiment_df = self._sentiment_to_dataframe(sentiment_results, target_ticker)
        
        # Convert market data to DataFrame
        market_df = self._market_data_to_dataframe(market_data_points)
        
        # Create features
        feature_set = self.feature_engineer.create_features(
            sentiment_df, market_df, target_ticker
        )
        
        self.logger.info(f"Prepared training data for {target_ticker}: {feature_set.metadata}")
        
        return feature_set
    
    def train_model(self, 
                   sentiment_results: List[EnsembleSentimentResult],
                   market_data_points: List[DataPoint],
                   target_ticker: str,
                   model_config: Optional[ModelConfig] = None,
                   training_config: Optional[TrainingConfig] = None) -> Dict[str, Any]:
        """
        Train fusion model for a specific ticker
        
        Args:
            sentiment_results: Historical sentiment data
            market_data_points: Historical market data
            target_ticker: Ticker to train model for
            model_config: Model configuration
            training_config: Training configuration
        """
        # Prepare training data
        feature_set = self.prepare_training_data(
            sentiment_results, market_data_points, target_ticker
        )
        
        if feature_set.features.size == 0:
            raise ValueError(f"No training data available for {target_ticker}")
        
        # Create fusion engine for this ticker
        fusion_engine = MultimodalFusionEngine(model_config, training_config)
        
        # Train model
        training_results = fusion_engine.train(feature_set)
        
        # Cache trained model
        self.ticker_models[target_ticker] = {
            'engine': fusion_engine,
            'feature_engineer': self.feature_engineer,
            'last_trained': datetime.now(),
            'training_results': training_results
        }
        
        self.last_training_time = datetime.now()
        
        self.logger.info(f"Model trained for {target_ticker}: {training_results}")
        
        return training_results
    
    def predict(self, 
               sentiment_results: List[EnsembleSentimentResult],
               market_data_points: List[DataPoint],
               target_ticker: str) -> FusionPrediction:
        """
        Make prediction for a specific ticker
        
        Args:
            sentiment_results: Recent sentiment data
            market_data_points: Recent market data
            target_ticker: Ticker to predict
        """
        # Check if model exists for ticker
        if target_ticker not in self.ticker_models:
            raise ValueError(f"No trained model available for {target_ticker}")
        
        model_info = self.ticker_models[target_ticker]
        fusion_engine = model_info['engine']
        
        # Prepare prediction data
        feature_set = self.prepare_training_data(
            sentiment_results, market_data_points, target_ticker
        )
        
        if feature_set.features.size == 0:
            return self._create_neutral_prediction(target_ticker)
        
        # Make prediction
        prediction_result = fusion_engine.predict(feature_set)
        
        if len(prediction_result.predictions) == 0:
            return self._create_neutral_prediction(target_ticker)
        
        # Get latest prediction
        latest_prediction = prediction_result.predictions[-1]
        latest_confidence = prediction_result.confidence[-1]
        latest_probabilities = prediction_result.probabilities[-1]
        
        # Create probability distribution
        prob_dist = {
            'bearish': float(latest_probabilities[0]),
            'neutral': float(latest_probabilities[1]),
            'bullish': float(latest_probabilities[2])
        }
        
        # Analyze contributing factors
        contributing_factors = self._analyze_contributing_factors(
            sentiment_results, market_data_points, feature_set
        )
        
        return FusionPrediction(
            ticker=target_ticker,
            prediction=int(latest_prediction),
            confidence=float(latest_confidence),
            probability_distribution=prob_dist,
            contributing_factors=contributing_factors,
            timestamp=datetime.now(),
            horizon_hours=self.prediction_horizon
        )
    
    def batch_predict(self, 
                     sentiment_results: List[EnsembleSentimentResult],
                     market_data_points: List[DataPoint],
                     target_tickers: List[str]) -> Dict[str, FusionPrediction]:
        """Make predictions for multiple tickers"""
        predictions = {}
        
        for ticker in target_tickers:
            try:
                prediction = self.predict(sentiment_results, market_data_points, ticker)
                predictions[ticker] = prediction
            except Exception as e:
                self.logger.error(f"Error predicting for {ticker}: {str(e)}")
                predictions[ticker] = self._create_neutral_prediction(ticker)
        
        return predictions
    
    def _sentiment_to_dataframe(self, 
                               sentiment_results: List[EnsembleSentimentResult],
                               target_ticker: str) -> pd.DataFrame:
        """Convert sentiment results to DataFrame"""
        data = []
        
        for result in sentiment_results:
            # Filter for target ticker or general market sentiment
            if result.finbert_result and (
                not result.finbert_result.metadata.get('ticker') or 
                result.finbert_result.metadata.get('ticker') == target_ticker
            ):
                
                # Convert sentiment to numerical score
                sentiment_score = 0.0
                if result.sentiment == 'positive':
                    sentiment_score = result.ensemble_score
                elif result.sentiment == 'negative':
                    sentiment_score = result.ensemble_score
                else:
                    sentiment_score = 0.0
                
                # Extract individual scores
                finbert_scores = result.finbert_result.scores if result.finbert_result else {}
                
                data.append({
                    'timestamp': result.timestamp,
                    'sentiment_score': sentiment_score,
                    'confidence': result.confidence,
                    'positive_score': finbert_scores.get('positive', 0),
                    'negative_score': finbert_scores.get('negative', 0),
                    'neutral_score': finbert_scores.get('neutral', 0),
                    'credibility_score': getattr(result.finbert_result, 'credibility_score', 1.0) if result.finbert_result else 1.0
                })
        
        if not data:
            # Return empty DataFrame with required columns
            return pd.DataFrame(columns=[
                'timestamp', 'sentiment_score', 'confidence', 
                'positive_score', 'negative_score', 'neutral_score', 'credibility_score'
            ])
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.sort_values('timestamp')
    
    def _market_data_to_dataframe(self, market_data_points: List[DataPoint]) -> pd.DataFrame:
        """Convert market data points to DataFrame"""
        data = []
        
        for point in market_data_points:
            if point.source in ['YahooFinance', 'AlphaVantage'] and point.metadata.get('data_type') == 'stock_price':
                data.append({
                    'timestamp': point.timestamp,
                    'ticker': point.ticker,
                    'open': point.metadata.get('open', 0),
                    'high': point.metadata.get('high', 0),
                    'low': point.metadata.get('low', 0),
                    'close': point.metadata.get('close', 0),
                    'volume': point.metadata.get('volume', 0)
                })
        
        if not data:
            return pd.DataFrame(columns=['timestamp', 'ticker', 'open', 'high', 'low', 'close', 'volume'])
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.sort_values('timestamp')
    
    def _analyze_contributing_factors(self, 
                                    sentiment_results: List[EnsembleSentimentResult],
                                    market_data_points: List[DataPoint],
                                    feature_set: FeatureSet) -> Dict[str, float]:
        """Analyze factors contributing to the prediction"""
        factors = {}
        
        # Sentiment contribution
        recent_sentiment = [r for r in sentiment_results[-10:]]  # Last 10 sentiment points
        if recent_sentiment:
            avg_sentiment = np.mean([r.ensemble_score for r in recent_sentiment])
            factors['sentiment_trend'] = float(avg_sentiment)
            factors['sentiment_strength'] = float(np.mean([r.confidence for r in recent_sentiment]))
        
        # Market momentum contribution
        recent_market = [p for p in market_data_points[-10:] if p.metadata.get('data_type') == 'stock_price']
        if len(recent_market) >= 2:
            prices = [p.metadata.get('close', 0) for p in recent_market]
            if prices:
                momentum = (prices[-1] - prices[0]) / prices[0] if prices[0] != 0 else 0
                factors['price_momentum'] = float(momentum)
                
                # Volatility
                returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices)) if prices[i-1] != 0]
                if returns:
                    factors['volatility'] = float(np.std(returns))
        
        # Volume trend
        volumes = [p.metadata.get('volume', 0) for p in recent_market]
        if len(volumes) >= 2:
            volume_trend = (volumes[-1] - np.mean(volumes[:-1])) / np.mean(volumes[:-1]) if np.mean(volumes[:-1]) != 0 else 0
            factors['volume_trend'] = float(volume_trend)
        
        return factors
    
    def _create_neutral_prediction(self, ticker: str) -> FusionPrediction:
        """Create neutral prediction for error cases"""
        return FusionPrediction(
            ticker=ticker,
            prediction=0,
            confidence=0.33,
            probability_distribution={'bearish': 0.33, 'neutral': 0.34, 'bullish': 0.33},
            contributing_factors={},
            timestamp=datetime.now(),
            horizon_hours=self.prediction_horizon
        )
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all trained models"""
        status = {
            'total_models': len(self.ticker_models),
            'last_training_time': self.last_training_time,
            'models': {}
        }
        
        for ticker, model_info in self.ticker_models.items():
            status['models'][ticker] = {
                'last_trained': model_info['last_trained'],
                'training_results': model_info['training_results'],
                'model_summary': model_info['engine'].get_model_summary()
            }
        
        return status
    
    def should_retrain(self, ticker: str, recent_performance: float) -> bool:
        """Determine if model should be retrained"""
        if ticker not in self.ticker_models:
            return True
        
        model_info = self.ticker_models[ticker]
        
        # Check time since last training
        time_since_training = datetime.now() - model_info['last_trained']
        if time_since_training > timedelta(days=7):  # Retrain weekly
            return True
        
        # Check performance degradation
        if recent_performance < self.retrain_threshold:
            return True
        
        return False
    
    def cleanup_old_models(self, max_age_days: int = 30):
        """Remove old models to free memory"""
        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        
        tickers_to_remove = []
        for ticker, model_info in self.ticker_models.items():
            if model_info['last_trained'] < cutoff_time:
                tickers_to_remove.append(ticker)
        
        for ticker in tickers_to_remove:
            del self.ticker_models[ticker]
            self.logger.info(f"Removed old model for {ticker}")
        
        return len(tickers_to_remove)
    
    def export_predictions(self, predictions: Dict[str, FusionPrediction]) -> pd.DataFrame:
        """Export predictions to DataFrame"""
        data = []
        
        for ticker, prediction in predictions.items():
            data.append({
                'ticker': prediction.ticker,
                'prediction': prediction.prediction,
                'prediction_label': ['bearish', 'neutral', 'bullish'][prediction.prediction + 1],
                'confidence': prediction.confidence,
                'prob_bearish': prediction.probability_distribution['bearish'],
                'prob_neutral': prediction.probability_distribution['neutral'],
                'prob_bullish': prediction.probability_distribution['bullish'],
                'timestamp': prediction.timestamp,
                'horizon_hours': prediction.horizon_hours,
                **prediction.contributing_factors
            })
        
        return pd.DataFrame(data)
