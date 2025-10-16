"""
Feature engineering for multimodal fusion of sentiment and market data
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

@dataclass
class FeatureSet:
    """Container for engineered features"""
    features: np.ndarray
    feature_names: List[str]
    target: Optional[np.ndarray]
    timestamps: List[datetime]
    metadata: Dict[str, Any]

class MultimodalFeatureEngineer:
    """Feature engineering for combining sentiment and market data"""

    def __init__(self,
                 lookback_window: int = 24,
                 prediction_horizon: int = 4,
                 normalize_features: bool = True):
        """
        Initialize feature engineer

        Args:
            lookback_window: Hours of historical data to use
            prediction_horizon: Hours ahead to predict
            normalize_features: Whether to normalize features
        """
        self.logger = logging.getLogger("feature_engineer")
        self.lookback_window = lookback_window
        self.prediction_horizon = prediction_horizon
        self.normalize_features = normalize_features

        # Scalers for different feature types
        self.sentiment_scaler = StandardScaler()
        self.market_scaler = StandardScaler()
        self.volume_scaler = MinMaxScaler()

        # Feature selection
        self.feature_selector = SelectKBest(f_regression, k=20)

        # Fitted flags
        self.scalers_fitted = False
        self.selector_fitted = False

        self.logger.info(f"Initialized feature engineer - lookback:
    {lookback_window}h, horizon: {prediction_horizon}h")

    def create_features(self,
                       sentiment_data: pd.DataFrame,
                       market_data: pd.DataFrame,
                       target_ticker: str) -> FeatureSet:
        """
        Create features from sentiment and market data

        Args:
            sentiment_data: DataFrame with sentiment scores over time
            market_data: DataFrame with market data (OHLCV)
            target_ticker: Ticker symbol to predict
        """
        # Align data by timestamp
        aligned_data = self._align_data(sentiment_data, market_data,
    target_ticker)

        if aligned_data.empty:
            return self._create_empty_features()

        # Create time-based features
        time_features = self._create_time_features(aligned_data)

        # Create sentiment features
        sentiment_features = self._create_sentiment_features(aligned_data)

        # Create market features
        market_features = self._create_market_features(aligned_data)

        # Create technical indicators
        technical_features = self._create_technical_features(aligned_data)

        # Create interaction features
        interaction_features = self._create_interaction_features(aligned_data)

        # Combine all features
        all_features = pd.concat([
            time_features,
            sentiment_features,
            market_features,
            technical_features,
            interaction_features
        ], axis=1)

        # Create sequences for LSTM
        feature_sequences, targets, timestamps = self._create_sequences(
            all_features, aligned_data, target_ticker
        )

        # Normalize features if requested
        if self.normalize_features:
            feature_sequences = self._normalize_features(feature_sequences)

        # Feature selection
        if self.selector_fitted:
            feature_sequences = self.feature_selector.transform(
                feature_sequences.reshape(feature_sequences.shape[0], -1)
            ).reshape(feature_sequences.shape[0], -1,
    feature_sequences.shape[2])

        metadata = {
            "lookback_window": self.lookback_window,
            "prediction_horizon": self.prediction_horizon,
            "target_ticker": target_ticker,
            "feature_count": feature_sequences.shape[-1] if
    len(feature_sequences.shape) > 1 else 0,
            "sequence_length": feature_sequences.shape[1] if
    len(feature_sequences.shape) > 2 else 0,
            "sample_count": len(feature_sequences)
        }

        return FeatureSet(
            features=feature_sequences,
            feature_names=all_features.columns.tolist(),
            target=targets,
            timestamps=timestamps,
            metadata=metadata
        )

    def _align_data(self,
                   sentiment_data: pd.DataFrame,
                   market_data: pd.DataFrame,
                   target_ticker: str) -> pd.DataFrame:
        """Align sentiment and market data by timestamp"""
        try:
            # Ensure timestamp columns
            if 'timestamp' not in sentiment_data.columns:
                sentiment_data['timestamp'] =
    pd.to_datetime(sentiment_data.index)
            if 'timestamp' not in market_data.columns:
                market_data['timestamp'] = pd.to_datetime(market_data.index)

            # Filter market data for target ticker
            ticker_market_data = market_data[market_data.get('ticker', '') ==
    target_ticker].copy()

            if ticker_market_data.empty:
                self.logger.warning(f"No market data found for ticker
    {target_ticker}")
                return pd.DataFrame()

            # Resample to hourly data
            sentiment_hourly = self._resample_sentiment_data(sentiment_data)
            market_hourly = self._resample_market_data(ticker_market_data)

            # Merge on timestamp
            aligned = pd.merge(sentiment_hourly, market_hourly,
    on='timestamp', how='inner')

            # Sort by timestamp
            aligned = aligned.sort_values('timestamp').reset_index(drop=True)

            self.logger.info(f"Aligned data shape: {aligned.shape}")
            return aligned

        except Exception as e:
            self.logger.error(f"Error aligning data: {str(e)}")
            return pd.DataFrame()

    def _resample_sentiment_data(self, sentiment_data: pd.DataFrame) ->
    pd.DataFrame:
        """Resample sentiment data to hourly frequency"""
        df = sentiment_data.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

        # Aggregate sentiment scores
        resampled = df.resample('1H').agg({
            'sentiment_score': 'mean',
            'confidence': 'mean',
            'positive_score': 'mean',
            'negative_score': 'mean',
            'neutral_score': 'mean',
            'credibility_score': 'mean'
        }).fillna(0)

        # Add sentiment volume (count of sentiment data points)
        resampled['sentiment_volume'] = df.resample('1H').size()

        resampled.reset_index(inplace=True)
        return resampled

    def _resample_market_data(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Resample market data to hourly frequency"""
        df = market_data.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

        # Aggregate OHLCV data
        resampled = df.resample('1H').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).fillna(method='ffill')

        resampled.reset_index(inplace=True)
        return resampled

    def _create_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        df = data.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        time_features = pd.DataFrame(index=df.index)

        # Hour of day (cyclical encoding)
        time_features['hour_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour
    / 24)
        time_features['hour_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.hour
    / 24)

        # Day of week (cyclical encoding)
        time_features['dow_sin'] = np.sin(2 * np.pi *
    df['timestamp'].dt.dayofweek / 7)
        time_features['dow_cos'] = np.cos(2 * np.pi *
    df['timestamp'].dt.dayofweek / 7)

        # Month (cyclical encoding)
        time_features['month_sin'] = np.sin(2 * np.pi *
    df['timestamp'].dt.month / 12)
        time_features['month_cos'] = np.cos(2 * np.pi *
    df['timestamp'].dt.month / 12)

        # Market session indicators
        time_features['is_market_hours'] = (
            (df['timestamp'].dt.hour >= 9) & (df['timestamp'].dt.hour < 16) &
            (df['timestamp'].dt.dayofweek < 5)
        ).astype(int)

        time_features['is_premarket'] = (
            (df['timestamp'].dt.hour >= 4) & (df['timestamp'].dt.hour < 9) &
            (df['timestamp'].dt.dayofweek < 5)
        ).astype(int)

        time_features['is_aftermarket'] = (
            (df['timestamp'].dt.hour >= 16) & (df['timestamp'].dt.hour < 20) &
            (df['timestamp'].dt.dayofweek < 5)
        ).astype(int)

        return time_features

    def _create_sentiment_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create sentiment-based features"""
        sentiment_features = pd.DataFrame(index=data.index)

        # Basic sentiment features
        sentiment_features['sentiment_score'] = data.get('sentiment_score', 0)
        sentiment_features['confidence'] = data.get('confidence', 0)
        sentiment_features['positive_score'] = data.get('positive_score', 0)
        sentiment_features['negative_score'] = data.get('negative_score', 0)
        sentiment_features['neutral_score'] = data.get('neutral_score', 0)
        sentiment_features['credibility_score'] =
    data.get('credibility_score', 0)
        sentiment_features['sentiment_volume'] = data.get('sentiment_volume',
    0)

        # Rolling sentiment features
        for window in [3, 6, 12, 24]:
            if len(data) > window:
                sentiment_features[f'sentiment_ma_{window}h'] =
    data['sentiment_score'].rolling(window).mean()
                sentiment_features[f'sentiment_std_{window}h'] =
    data['sentiment_score'].rolling(window).std()
                sentiment_features[f'sentiment_momentum_{window}h'] = (
                    data['sentiment_score'] -
    data['sentiment_score'].shift(window)
                )

        # Sentiment strength
        sentiment_features['sentiment_strength'] = (
            sentiment_features['positive_score'] +
    sentiment_features['negative_score']
        )

        # Sentiment consensus (how decisive the sentiment is)
        sentiment_features['sentiment_consensus'] =
    np.abs(sentiment_features['sentiment_score'])

        return sentiment_features.fillna(0)

    def _create_market_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create market-based features"""
        market_features = pd.DataFrame(index=data.index)

        # Basic OHLCV features
        market_features['open'] = data.get('open', 0)
        market_features['high'] = data.get('high', 0)
        market_features['low'] = data.get('low', 0)
        market_features['close'] = data.get('close', 0)
        market_features['volume'] = data.get('volume', 0)

        # Price-based features
        market_features['hl_ratio'] = (data['high'] - data['low']) /
    data['close']
        market_features['oc_ratio'] = (data['close'] - data['open']) /
    data['open']

        # Returns
        market_features['returns_1h'] = data['close'].pct_change(1)
        market_features['returns_4h'] = data['close'].pct_change(4)
        market_features['returns_24h'] = data['close'].pct_change(24)

        # Volatility
        for window in [6, 12, 24]:
            if len(data) > window:
                market_features[f'volatility_{window}h'] =
    market_features['returns_1h'].rolling(window).std()

        # Volume features
        market_features['volume_ma_24h'] = data['volume'].rolling(24).mean()
        market_features['volume_ratio'] = data['volume'] /
    market_features['volume_ma_24h']

        return market_features.fillna(0)

    def _create_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicator features"""
        technical_features = pd.DataFrame(index=data.index)

        close_prices = data['close']

        # Moving averages
        for window in [6, 12, 24, 48]:
            if len(data) > window:
                ma = close_prices.rolling(window).mean()
                technical_features[f'ma_{window}h'] = ma
                technical_features[f'price_ma_ratio_{window}h'] = close_prices
    / ma

        # RSI (Relative Strength Index)
        if len(data) > 14:
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            technical_features['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        if len(data) > 26:
            exp1 = close_prices.ewm(span=12).mean()
            exp2 = close_prices.ewm(span=26).mean()
            technical_features['macd'] = exp1 - exp2
            technical_features['macd_signal'] =
    technical_features['macd'].ewm(span=9).mean()

        # Bollinger Bands
        if len(data) > 20:
            ma_20 = close_prices.rolling(20).mean()
            std_20 = close_prices.rolling(20).std()
            technical_features['bb_upper'] = ma_20 + (std_20 * 2)
            technical_features['bb_lower'] = ma_20 - (std_20 * 2)
            technical_features['bb_position'] = (close_prices -
    technical_features['bb_lower']) / (
                technical_features['bb_upper'] - technical_features['bb_lower']
            )

        return technical_features.fillna(0)

    def _create_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between sentiment and market data"""
        interaction_features = pd.DataFrame(index=data.index)

        # Sentiment-Price interactions
        sentiment_score = data.get('sentiment_score', 0)
        returns_1h = data['close'].pct_change(1)
        volume = data.get('volume', 0)

        # Sentiment-momentum alignment
        interaction_features['sentiment_momentum_align'] = sentiment_score *
    returns_1h

        # Sentiment-volume interaction
        interaction_features['sentiment_volume_interaction'] = sentiment_score
    * np.log1p(volume)

        # Sentiment strength during high volatility
        volatility_6h = returns_1h.rolling(6).std()
        interaction_features['sentiment_volatility_interaction'] = (
            np.abs(sentiment_score) * volatility_6h
        )

        # Credibility-weighted sentiment
        credibility = data.get('credibility_score', 1)
        interaction_features['credibility_weighted_sentiment'] =
    sentiment_score * credibility

        return interaction_features.fillna(0)

    def _create_sequences(self,
                         features: pd.DataFrame,
                         data: pd.DataFrame,
                         target_ticker: str) -> Tuple[np.ndarray, np.ndarray,
    List[datetime]]:
        """Create sequences for LSTM training"""
        # Create target variable (future price movement)
        target = self._create_target(data)

        # Convert to numpy arrays
        feature_array = features.values
        target_array = target.values
        timestamps = pd.to_datetime(data['timestamp']).tolist()

        # Create sequences
        sequences = []
        targets = []
        seq_timestamps = []

        for i in range(self.lookback_window, len(feature_array) -
    self.prediction_horizon):
            # Feature sequence
            seq = feature_array[i-self.lookback_window:i]
            sequences.append(seq)

            # Target
            targets.append(target_array[i + self.prediction_horizon])

            # Timestamp
            seq_timestamps.append(timestamps[i])

        return np.array(sequences), np.array(targets), seq_timestamps

    def _create_target(self, data: pd.DataFrame) -> pd.Series:
        """Create target variable for prediction"""
        close_prices = data['close']

        # Future price change (classification: -1, 0, 1)
        future_returns = close_prices.shift(-self.prediction_horizon).pct_chang
    e(self.prediction_horizon)

        # Convert to classification target
        threshold = 0.02  # 2% threshold
        target = pd.Series(0, index=data.index)  # Neutral
        target[future_returns > threshold] = 1    # Positive
        target[future_returns < -threshold] = -1  # Negative

        return target

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize feature sequences"""
        if not self.scalers_fitted:
            # Fit scalers on flattened data
            flattened = features.reshape(-1, features.shape[-1])
            self.sentiment_scaler.fit(flattened[:, :10])  # First 10 features
    assumed to be sentiment
            self.market_scaler.fit(flattened[:, 10:])     # Rest assumed to be
    market
            self.scalers_fitted = True

        # Transform sequences
        normalized_sequences = []
        for seq in features:
            normalized_seq = seq.copy()
            normalized_seq[:, :10] = self.sentiment_scaler.transform(seq[:,
    :10])
            normalized_seq[:, 10:] = self.market_scaler.transform(seq[:, 10:])
            normalized_sequences.append(normalized_seq)

        return np.array(normalized_sequences)

    def _create_empty_features(self) -> FeatureSet:
        """Create empty feature set for error cases"""
        return FeatureSet(
            features=np.array([]),
            feature_names=[],
            target=np.array([]),
            timestamps=[],
            metadata={"error": "No data available"}
        )

    def fit_feature_selector(self, features: np.ndarray, targets: np.ndarray):
        """Fit feature selector on training data"""
        if len(features) > 0 and len(targets) > 0:
            flattened_features = features.reshape(features.shape[0], -1)
            self.feature_selector.fit(flattened_features, targets)
            self.selector_fitted = True
            self.logger.info("Feature selector fitted")

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        if self.selector_fitted:
            scores = self.feature_selector.scores_
            selected_features = self.feature_selector.get_support()

            importance_dict = {}
            for i, (selected, score) in enumerate(zip(selected_features,
    scores)):
                if selected:
                    importance_dict[f"feature_{i}"] = score

            return importance_dict
        else:
            return {}
