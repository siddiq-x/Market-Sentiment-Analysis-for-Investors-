"""
LSTM-based multimodal fusion model for sentiment and market data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime
import os

from src.fusion.feature_engineer import FeatureSet


@dataclass
class ModelConfig:
    """Configuration for LSTM model"""

    input_size: int
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = True
    attention: bool = True
    output_size: int = 3  # 3 classes: negative, neutral, positive


@dataclass
class TrainingConfig:
    """Configuration for model training"""

    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    early_stopping_patience: int = 10
    validation_split: float = 0.2
    weight_decay: float = 1e-5


@dataclass
class PredictionResult:
    """Container for model predictions"""

    predictions: np.ndarray
    probabilities: np.ndarray
    confidence: np.ndarray
    timestamps: List[datetime]
    metadata: Dict[str, Any]


class FinancialDataset(Dataset):
    """PyTorch dataset for financial time series data"""

    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.targets = torch.LongTensor(targets + 1)  # Convert -1,0,1 to 0,1,2

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class AttentionLayer(nn.Module):
    """Attention mechanism for LSTM"""

    def __init__(self, hidden_size: int):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        # lstm_output shape: (batch_size, seq_len, hidden_size)
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        # attention_weights shape: (batch_size, seq_len, 1)

        # Apply attention weights
        attended_output = torch.sum(attention_weights * lstm_output, dim=1)
        # attended_output shape: (batch_size, hidden_size)

        return attended_output, attention_weights


class LSTMFusionModel(nn.Module):
    """LSTM-based multimodal fusion model"""

    def __init__(self, config: ModelConfig):
        super(LSTMFusionModel, self).__init__()
        self.config = config

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional,
            batch_first=True,
        )

        # Calculate LSTM output size
        lstm_output_size = config.hidden_size * (2 if config.bidirectional else 1)

        # Attention layer
        if config.attention:
            self.attention = AttentionLayer(lstm_output_size)
            final_size = lstm_output_size
        else:
            self.attention = None
            final_size = lstm_output_size

        # Classification layers
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(final_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, config.output_size),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if "weight" in name:
                if "lstm" in name:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.kaiming_normal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)

        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        # lstm_out shape: (batch_size, seq_len, hidden_size * num_directions)

        # Apply attention or use last output
        if self.attention:
            attended_out, attention_weights = self.attention(lstm_out)
            final_output = attended_out
        else:
            final_output = lstm_out[:, -1, :]  # Use last time step

        # Classification
        predictions = self.classifier(final_output)

        return predictions


class MultimodalFusionEngine:
    """Main engine for multimodal fusion and prediction"""

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        training_config: Optional[TrainingConfig] = None,
    ):
        self.logger = logging.getLogger("fusion_engine")

        self.model_config = model_config
        self.training_config = training_config or TrainingConfig()

        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_trained = False

        # Training history
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

        self.logger.info(f"Initialized fusion engine on device: {self.device}")

    def build_model(self, input_size: int) -> None:
        """Build the LSTM model"""
        if not self.model_config:
            self.model_config = ModelConfig(input_size=input_size)
        else:
            self.model_config.input_size = input_size

        self.model = LSTMFusionModel(self.model_config)
        self.model.to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        self.logger.info(
            f"Model built - Total params: {total_params:,}, Trainable: {trainable_params:,}"
        )

    def train(self, feature_set: FeatureSet) -> Dict[str, Any]:
        """Train the model on feature set"""
        if feature_set.features.size == 0 or feature_set.target is None:
            raise ValueError("Empty feature set provided for training")

        # Build model if not already built
        if self.model is None:
            input_size = feature_set.features.shape[-1]
            self.build_model(input_size)

        # Prepare data
        train_loader, val_loader = self._prepare_data_loaders(feature_set)

        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5
        )

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0

        self.logger.info("Starting training...")

        for epoch in range(self.training_config.num_epochs):
            # Training phase
            train_loss, train_acc = self._train_epoch(
                train_loader, criterion, optimizer
            )

            # Validation phase
            val_loss, val_acc = self._validate_epoch(val_loader, criterion)

            # Update scheduler
            scheduler.step(val_loss)

            # Record history
            self.training_history["train_loss"].append(train_loss)
            self.training_history["val_loss"].append(val_loss)
            self.training_history["train_acc"].append(train_acc)
            self.training_history["val_acc"].append(val_acc)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self._save_checkpoint(epoch, val_loss)
            else:
                patience_counter += 1

            if epoch % 10 == 0:
                self.logger.info(
                    f"Epoch {epoch}: Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}"
                )

            if patience_counter >= self.training_config.early_stopping_patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break

        self.is_trained = True

        return {
            "final_train_loss": train_loss,
            "final_val_loss": val_loss,
            "final_train_acc": train_acc,
            "final_val_acc": val_acc,
            "best_val_loss": best_val_loss,
            "epochs_trained": epoch + 1,
            "training_history": self.training_history,
        }

    def _prepare_data_loaders(
        self, feature_set: FeatureSet
    ) -> Tuple[DataLoader, DataLoader]:
        """Prepare training and validation data loaders"""
        # Create dataset
        dataset = FinancialDataset(feature_set.features, feature_set.target)

        # Split into train/validation
        dataset_size = len(dataset)
        val_size = int(dataset_size * self.training_config.validation_split)
        train_size = dataset_size - val_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=False,
            drop_last=False,
        )

        return train_loader, val_loader

    def _train_epoch(
        self, train_loader: DataLoader, criterion, optimizer
    ) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(self.device)
            batch_targets = batch_targets.to(self.device)

            optimizer.zero_grad()

            outputs = self.model(batch_features)
            loss = criterion(outputs, batch_targets)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += batch_targets.size(0)
            correct += (predicted == batch_targets).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def _validate_epoch(self, val_loader: DataLoader, criterion) -> Tuple[float, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)

                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_targets)

                total_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += batch_targets.size(0)
                correct += (predicted == batch_targets).sum().item()

        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def predict(self, feature_set: FeatureSet) -> PredictionResult:
        """Make predictions on new data"""
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")

        if feature_set.features.size == 0:
            return PredictionResult(
                predictions=np.array([]),
                probabilities=np.array([]),
                confidence=np.array([]),
                timestamps=[],
                metadata={"error": "Empty feature set"},
            )

        self.model.eval()
        predictions = []
        probabilities = []

        # Create data loader
        dataset = FinancialDataset(
            feature_set.features, np.zeros(len(feature_set.features))
        )
        data_loader = DataLoader(
            dataset, batch_size=self.training_config.batch_size, shuffle=False
        )

        with torch.no_grad():
            for batch_features, _ in data_loader:
                batch_features = batch_features.to(self.device)

                outputs = self.model(batch_features)
                probs = torch.softmax(outputs, dim=1)

                predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                probabilities.extend(probs.cpu().numpy())

        predictions = np.array(predictions) - 1  # Convert back to -1,0,1
        probabilities = np.array(probabilities)

        # Calculate confidence as max probability
        confidence = np.max(probabilities, axis=1)

        metadata = {
            "model_config": self.model_config.__dict__,
            "prediction_count": len(predictions),
            "average_confidence": np.mean(confidence),
        }

        return PredictionResult(
            predictions=predictions,
            probabilities=probabilities,
            confidence=confidence,
            timestamps=feature_set.timestamps,
            metadata=metadata,
        )

    def _save_checkpoint(self, epoch: int, val_loss: float):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "val_loss": val_loss,
            "model_config": self.model_config.__dict__,
            "training_config": self.training_config.__dict__,
            "training_history": self.training_history,
        }

        # Save to models directory
        os.makedirs("models", exist_ok=True)
        torch.save(checkpoint, "models/best_fusion_model.pth")

    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load model from checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Restore configs
            self.model_config = ModelConfig(**checkpoint["model_config"])
            self.training_config = TrainingConfig(**checkpoint["training_config"])

            # Build and load model
            self.build_model(self.model_config.input_size)
            self.model.load_state_dict(checkpoint["model_state_dict"])

            # Restore training history
            self.training_history = checkpoint["training_history"]
            self.is_trained = True

            self.logger.info(f"Model loaded from {checkpoint_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {str(e)}")
            return False

    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of model architecture and training"""
        if self.model is None:
            return {"error": "Model not built"}

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        summary = {
            "model_config": self.model_config.__dict__ if self.model_config else {},
            "training_config": self.training_config.__dict__,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "is_trained": self.is_trained,
            "device": str(self.device),
        }

        if self.training_history["train_loss"]:
            summary["training_summary"] = {
                "epochs_trained": len(self.training_history["train_loss"]),
                "final_train_loss": self.training_history["train_loss"][-1],
                "final_val_loss": self.training_history["val_loss"][-1],
                "best_val_loss": min(self.training_history["val_loss"]),
                "final_train_acc": self.training_history["train_acc"][-1],
                "final_val_acc": self.training_history["val_acc"][-1],
                "best_val_acc": max(self.training_history["val_acc"]),
            }

        return summary
