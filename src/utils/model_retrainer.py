"""
Continuous learning pipeline for model retraining
"""
import schedule
import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass
import pickle
import os
import threading

from src.fusion.fusion_manager import FusionManager
from src.sentiment.ensemble_analyzer import EnsembleSentimentAnalyzer
from src.data_ingestion.ingestion_manager import ingestion_manager
from config.config import config


@dataclass
class ModelPerformanceMetrics:
    """Container for model performance metrics"""
    ticker: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    prediction_count: int
    correct_predictions: int
    timestamp: datetime


@dataclass
class RetrainingJob:
    """Container for retraining job information"""
    ticker: str
    scheduled_time: datetime
    priority: str  # 'high', 'medium', 'low'
    reason: str
    data_points_required: int
    estimated_duration_minutes: int


class ModelRetrainer:
    """Handles continuous learning and model retraining"""

    def __init__(self,
                 performance_threshold: float = 0.7,
                 min_data_points: int = 100,
                 retraining_interval_days: int = 7):
        """
        Initialize model retrainer

        Args:
            performance_threshold: Minimum performance to trigger retraining
            min_data_points: Minimum data points needed for retraining
            retraining_interval_days: Regular retraining interval
        """
        self.logger = logging.getLogger("model_retrainer")

        self.performance_threshold = performance_threshold
        self.min_data_points = min_data_points
        self.retraining_interval_days = retraining_interval_days

        # Performance tracking
        self.performance_history = {}  # ticker -> list of metrics
        self.retraining_queue = []
        self.active_jobs = {}

        # Components
        self.fusion_manager = FusionManager()
        self.sentiment_analyzer = EnsembleSentimentAnalyzer()

        # Threading
        self.running = False
        self.scheduler_thread = None
        self.worker_thread = None

        # Setup scheduling
        self._setup_schedule()

        self.logger.info("Model retrainer initialized")

    def _setup_schedule(self):
        """Setup retraining schedule"""
        # Daily performance evaluation
        schedule.every().day.at("02:00").do(self._evaluate_all_models)

        # Weekly full retraining
        schedule.every().sunday.at("03:00").do(self._schedule_weekly_retraining
    )

        # Hourly drift detection
        schedule.every().hour.do(self._detect_model_drift)

        self.logger.info("Retraining schedule configured")

    def start(self):
        """Start the retraining service"""
        if self.running:
            return

        self.running = True

        # Start scheduler thread
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()

        # Start worker thread
        self.worker_thread = threading.Thread(target=self._worker_loop)
        self.worker_thread.daemon = True
        self.worker_thread.start()

        self.logger.info("Model retrainer started")

    def stop(self):
        """Stop the retraining service"""
        self.running = False

        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        if self.worker_thread:
            self.worker_thread.join(timeout=5)

        self.logger.info("Model retrainer stopped")

    def _scheduler_loop(self):
        """Main scheduler loop"""
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                self.logger.error(f"Error in scheduler loop: {str(e)}")
                time.sleep(300)  # Wait 5 minutes on error

    def _worker_loop(self):
        """Main worker loop for processing retraining jobs"""
        while self.running:
            try:
                if self.retraining_queue:
                    job = self.retraining_queue.pop(0)
                    self._execute_retraining_job(job)
                else:
                    time.sleep(30)  # Wait 30 seconds if no jobs
            except Exception as e:
                self.logger.error(f"Error in worker loop: {str(e)}")
                time.sleep(60)

    def record_prediction_performance(self,
                                    ticker: str,
                                    predicted: int,
                                    actual: int,
                                    confidence: float):
        """Record prediction performance for evaluation"""
        if ticker not in self.performance_history:
            self.performance_history[ticker] = []

        # Store prediction result
        self.performance_history[ticker].append({
            'timestamp': datetime.now(),
            'predicted': predicted,
            'actual': actual,
            'confidence': confidence,
            'correct': predicted == actual
        })

        # Keep only recent history (last 1000 predictions)
        if len(self.performance_history[ticker]) > 1000:
            self.performance_history[ticker] = self.performance_history[ticker][-1000:]

        # Check if immediate retraining is needed
        self._check_immediate_retraining(ticker)

    def _check_immediate_retraining(self, ticker: str):
        """Check if immediate retraining is needed based on recent performance"""
        if ticker not in self.performance_history:
            return

        # Get last 20 predictions
        recent_predictions = self.performance_history[ticker][-20:]

        if len(recent_predictions) >= 10:
            accuracy = sum(1 for p in recent_predictions if p['correct']) / len(recent_predictions)

            if accuracy < self.performance_threshold:
                self.logger.warning(
                    f"Poor performance detected for {ticker}: {accuracy*100:.1f}% accuracy. "
                    "Triggering retraining..."
                )
                self._schedule_retraining(
                    ticker=ticker,
                    priority='high',
                    reason=f"Performance dropped to {accuracy:.3f}"
                )

    def _evaluate_all_models(self):
        """Evaluate performance of all models"""
        self.logger.info("Starting daily model evaluation")

        for ticker in config.monitored_tickers:
            try:
                metrics = self._calculate_performance_metrics(ticker)
                if metrics:
                    self._evaluate_model_performance(metrics)
            except Exception as e:
                self.logger.error(f"Error evaluating model for {ticker}: {str(e)}")

        self.logger.info("Daily model evaluation completed")

    def _calculate_performance_metrics(self, ticker: str) -> Optional[ModelPerformanceMetrics]:
        """Calculate performance metrics for a ticker"""
        if ticker not in self.performance_history:
            return None

        # Get recent predictions (last 7 days)
        cutoff_time = datetime.now() - timedelta(days=7)
        recent_predictions = [
            p for p in self.performance_history[ticker]
            if p['timestamp'] >= cutoff_time
        ]

        if len(recent_predictions) < 10:  # Need minimum predictions
            return None

        # Calculate metrics
        correct_predictions = sum(1 for p in recent_predictions if
    p['correct'])
        total_predictions = len(recent_predictions)
        accuracy = correct_predictions / total_predictions

        # Calculate precision, recall, F1 for each class
        classes = [-1, 0, 1]  # bearish, neutral, bullish
        precision_scores = []
        recall_scores = []

        for cls in classes:
            true_positives = sum(1 for p in recent_predictions
                               if p['predicted'] == cls and p['actual'] == cls)
            false_positives = sum(
                1 for p in recent_predictions if p['predicted'] == cls and p['actual'] != cls
            )
            false_negatives = sum(
                1 for p in recent_predictions if p['predicted'] != cls and p['actual'] == cls
            )

            precision = (
                true_positives / (true_positives + false_positives)
                if (true_positives + false_positives) > 0
                else 0
            )
            recall = (
                true_positives / (true_positives + false_negatives)
                if (true_positives + false_negatives) > 0
                else 0
            )

            precision_scores.append(precision)
            recall_scores.append(recall)

        avg_precision = np.mean(precision_scores)
        avg_recall = np.mean(recall_scores)
        f1_score = (
            2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
            if (avg_precision + avg_recall) > 0
            else 0
        )

        return ModelPerformanceMetrics(
            ticker=ticker,
            accuracy=accuracy,
            precision=avg_precision,
            recall=avg_recall,
            f1_score=f1_score,
            prediction_count=total_predictions,
            correct_predictions=correct_predictions,
            timestamp=datetime.now()
        )

    def _evaluate_model_performance(self, metrics: ModelPerformanceMetrics):
        """Evaluate if model needs retraining based on metrics"""
        ticker = metrics.ticker

        # Check accuracy threshold
        if metrics.accuracy < self.performance_threshold:
            self._schedule_retraining(
                ticker=ticker,
                priority='high',
                reason=f"Accuracy below threshold: {metrics.accuracy:.3f}"
            )
            return

        # Check F1 score
        if metrics.f1_score < 0.6:
            self._schedule_retraining(
                ticker=ticker,
                priority='medium',
                reason=f"F1 score below threshold: {metrics.f1_score:.3f}"
            )
            return

        # Check prediction volume
        if metrics.prediction_count < 20:
            self._schedule_retraining(
                ticker=ticker,
                priority='low',
                reason="Insufficient recent predictions for reliable evaluation",
            )

        self.logger.info(
            f"Model performance for {ticker}: Accuracy={metrics.accuracy:.3f}, F1={metrics.f1_score:.3f}"
        )

    def _detect_model_drift(self):
        """Detect model drift by comparing recent vs historical performance"""
        self.logger.debug("Checking for model drift")

        for ticker in config.monitored_tickers:
            try:
                if ticker not in self.performance_history:
                    continue

                # Compare last 24 hours vs previous week
                now = datetime.now()
                recent_cutoff = now - timedelta(hours=24)
                historical_cutoff = now - timedelta(days=7)

                recent_predictions = [
                    p for p in self.performance_history[ticker]
                    if p['timestamp'] >= recent_cutoff
                ]

                historical_predictions = [
                    p for p in self.performance_history[ticker]
                    if historical_cutoff <= p['timestamp'] < recent_cutoff
                ]

                if len(recent_predictions) >= 5 and len(historical_predictions) >= 20:
                    recent_accuracy = sum(1 for p in recent_predictions if p['correct']) / len(recent_predictions)
                    historical_accuracy = (
                        sum(1 for p in historical_predictions if p['correct']) / len(historical_predictions)
                    )

                    drift = historical_accuracy - recent_accuracy

                    if drift > 0.15:  # 15% performance drop
                        self.logger.warning(f"Model drift detected for {ticker}: {drift:.3f}")
                        self._schedule_retraining(
                            ticker=ticker,
                            priority='high',
                            reason=f"Model drift detected: {drift:.3f}"
                        )

            except Exception as e:
                self.logger.error(f"Error detecting drift for {ticker}: {str(e)}")

    def _schedule_weekly_retraining(self):
        """Schedule weekly retraining for all models"""
        self.logger.info("Scheduling weekly retraining")

        for ticker in config.monitored_tickers:
            self._schedule_retraining(
                ticker=ticker,
                priority='low',
                reason="Weekly scheduled retraining"
            )

    def _schedule_retraining(self, ticker: str, priority: str, reason: str):
        """Schedule a retraining job"""
        # Check if already scheduled
        if any(job.ticker == ticker for job in self.retraining_queue):
            self.logger.debug(f"Retraining already scheduled for {ticker}")
            return

        # Check if currently being retrained
        if ticker in self.active_jobs:
            self.logger.debug(f"Retraining already in progress for {ticker}")
            return

        # Calculate estimated duration and data requirements
        data_points_required = max(self.min_data_points, 200)
        estimated_duration = 30 if priority == 'high' else 60  # minutes

        # Schedule based on priority
        if priority == 'high':
            scheduled_time = datetime.now() + timedelta(minutes=5)
        elif priority == 'medium':
            scheduled_time = datetime.now() + timedelta(hours=2)
        else:
            scheduled_time = datetime.now() + timedelta(hours=12)

        job = RetrainingJob(
            ticker=ticker,
            scheduled_time=scheduled_time,
            priority=priority,
            reason=reason,
            data_points_required=data_points_required,
            estimated_duration_minutes=estimated_duration
        )

        # Insert in priority order
        inserted = False
        for i, existing_job in enumerate(self.retraining_queue):
            if self._job_priority_value(job.priority) > self._job_priority_value(existing_job.priority):
                self.retraining_queue.insert(i, job)
                inserted = True
                break

        if not inserted:
            self.retraining_queue.append(job)

        self.logger.info(f"Scheduled retraining for {ticker}: {reason} (priority: {priority})")

    def _job_priority_value(self, priority: str) -> int:
        """Convert priority string to numeric value"""
        return {'high': 3, 'medium': 2, 'low': 1}.get(priority, 1)

    def _execute_retraining_job(self, job: RetrainingJob):
        """Execute a retraining job"""
        ticker = job.ticker

        try:
            self.logger.info(f"Starting retraining for {ticker}: {job.reason}")
            self.active_jobs[ticker] = job

            # Collect training data
            training_data = self._collect_training_data(ticker, job.data_points_required)

            if not training_data or len(training_data) < self.min_data_points:
                self.logger.warning(
                    f"Insufficient data for retraining {ticker}: {len(training_data) if training_data else 0} points"
                )
                return

            # Retrain model
            success = self._retrain_model(ticker, training_data)

            if success:
                self.logger.info(f"Successfully retrained model for {ticker}")
                # Reset performance history to start fresh evaluation
                if ticker in self.performance_history:
                    self.performance_history[ticker] = []
            else:
                self.logger.error(f"Failed to retrain model for {ticker}")

        except Exception as e:
            self.logger.error(f"Error in retraining job for {ticker}: {str(e)}")
        finally:
            if ticker in self.active_jobs:
                del self.active_jobs[ticker]

    def _collect_training_data(self, ticker: str, required_points: int) -> Optional[List]:
        """Collect training data for retraining"""
        try:
            # Fetch historical data (last 30 days)
            all_data = ingestion_manager.fetch_all_data(
                tickers=[ticker],
                hours_back=24 * 30  # 30 days
            )

            if not all_data:
                return None

            # Process and prepare data for training
            # This would involve the same preprocessing as initial training
            processed_data = []

            for data_point in all_data:
                if data_point.content and data_point.content.strip():
                    # Analyze sentiment
                    sentiment_result = self.sentiment_analyzer.analyze_sentiment(
                        data_point.content,
                        context={"source": data_point.source, "ticker":
    data_point.ticker}
                    )

                    processed_data.append({
                        'timestamp': data_point.timestamp,
                        'content': data_point.content,
                        'sentiment': sentiment_result.sentiment,
                        'confidence': sentiment_result.confidence,
                        'ensemble_score': sentiment_result.ensemble_score,
                        'ticker': data_point.ticker,
                        'source': data_point.source
                    })

            return processed_data[-required_points:] if len(processed_data) >= required_points else processed_data

        except Exception as e:
            self.logger.error(f"Error collecting training data for {ticker}: {str(e)}")
            return None

    def _retrain_model(self, ticker: str, training_data: List) -> bool:
        """Retrain the fusion model for a ticker"""
        try:
            # This would involve retraining the LSTM fusion model
            # For now, we'll simulate the retraining process

            self.logger.info(f"Retraining model for {ticker} with {len(training_data)} data points")

            # Simulate training time
            time.sleep(2)

            # In a real implementation, this would:
            # 1. Convert training_data to proper format
            # 2. Create feature sets using the feature engineer
            # 3. Retrain the LSTM model
            # 4. Validate the new model
            # 5. Replace the old model if validation passes

            return True

        except Exception as e:
            self.logger.error(f"Error retraining model for {ticker}: {str(e)}")
            return False

    def get_retraining_status(self) -> Dict[str, Any]:
        """Get current retraining status"""
        return {
            "running": self.running,
            "queued_jobs": len(self.retraining_queue),
            "active_jobs": len(self.active_jobs),
            "queue": [
                {
                    "ticker": job.ticker,
                    "priority": job.priority,
                    "reason": job.reason,
                    "scheduled_time": job.scheduled_time.isoformat(),
                    "estimated_duration_minutes":
    job.estimated_duration_minutes
                }
                for job in self.retraining_queue[:10]  # Show first 10
            ],
            "active": [
                {
                    "ticker": ticker,
                    "reason": job.reason,
                    "started_time": datetime.now().isoformat()  # Simplified
                }
                for ticker, job in self.active_jobs.items()
            ],
            "performance_summary": {
                ticker: {
                    "total_predictions": len(history),
                    "recent_accuracy": sum(1 for p in history[-50:] if
    p['correct']) / min(50, len(history)) if history else 0
                }
                for ticker, history in self.performance_history.items()
            }
        }

    def force_retrain(self, ticker: str, reason: str = "Manual trigger") -> bool:
        """Force immediate retraining for a ticker"""
        try:
            self._schedule_retraining(
                ticker=ticker,
                priority='high',
                reason=reason
            )
            return True
        except Exception as e:
            self.logger.error(f"Error forcing retrain for {ticker}: {str(e)}")
            return False
