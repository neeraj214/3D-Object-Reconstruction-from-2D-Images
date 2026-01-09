import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
import logging
import os
import yaml
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import time
from datetime import datetime

# Import custom modules
from ..models.fusion_classifier import FusionClassifier
from ..utils.visualization import ResultsVisualizer


class PointCloudDataset(Dataset):
    """
    Dataset class for point cloud classification.
    """
    
    def __init__(self, 
                 features_2d: np.ndarray,
                 features_3d: np.ndarray,
                 labels: np.ndarray,
                 transform=None):
        """
        Initialize the dataset.
        
        Args:
            features_2d: 2D features
            features_3d: 3D features
            labels: Class labels
            transform: Optional transform function
        """
        self.features_2d = torch.FloatTensor(features_2d)
        self.features_3d = torch.FloatTensor(features_3d)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        features_2d = self.features_2d[idx]
        features_3d = self.features_3d[idx]
        label = self.labels[idx]
        
        if self.transform:
            # Apply transforms if any
            pass
        
        return features_2d, features_3d, label


class FusionTrainer:
    """
    Trainer for the fusion classifier.
    """
    
    def __init__(self, config: dict):
        """
        Initialize the trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        self.setup_logging()
        
        # Initialize model
        self.model = self.create_model()
        
        # Initialize optimizer and scheduler
        self.optimizer = self.create_optimizer()
        self.scheduler = self.create_scheduler()
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Best model tracking
        self.best_val_accuracy = 0.0
        self.best_model_state = None
        
        self.logger.info(f"Trainer initialized on device: {self.device}")
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO'))
        log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        logging.basicConfig(level=log_level, format=log_format)
        self.logger = logging.getLogger(__name__)
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # File handler
        log_file = log_config.get('file', 'logs/training.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        self.logger.addHandler(file_handler)
    
    def create_model(self) -> FusionClassifier:
        """Create the fusion model."""
        model_config = self.config['model']['fusion']
        
        model = FusionClassifier(
            input_dim_2d=model_config.get('input_dim_2d', 2048),
            input_dim_3d=model_config.get('input_dim_3d', 1024),
            hidden_dim=model_config.get('hidden_dim', 512),
            num_classes=model_config.get('num_classes', 10),
            dropout=model_config.get('dropout', 0.3),
            fusion_method=model_config.get('fusion_method', 'concat')
        )
        
        model.to(self.device)
        return model
    
    def create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        optimizer_config = self.config['training']['optimizer']
        optimizer_type = optimizer_config.get('type', 'Adam')
        learning_rate = self.config['training']['learning_rate']
        weight_decay = optimizer_config.get('weight_decay', 0.0001)
        
        if optimizer_type == 'Adam':
            return optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_type == 'SGD':
            return optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    def create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        scheduler_config = self.config['training'].get('scheduler', {})
        scheduler_type = scheduler_config.get('type', None)
        
        if scheduler_type is None:
            return None
        
        if scheduler_type == 'StepLR':
            step_size = scheduler_config.get('step_size', 30)
            gamma = scheduler_config.get('gamma', 0.1)
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_type == 'ReduceLROnPlateau':
            patience = scheduler_config.get('patience', 10)
            factor = scheduler_config.get('factor', 0.5)
            return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=patience, factor=factor)
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    
    def prepare_data(self, 
                    features_2d: np.ndarray,
                    features_3d: np.ndarray,
                    labels: np.ndarray) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare training and validation data loaders.
        
        Args:
            features_2d: 2D features
            features_3d: 3D features
            labels: Class labels
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Split data
        validation_split = self.config['training'].get('validation_split', 0.2)
        random_state = self.config.get('system', {}).get('seed', 42)
        
        # Create full dataset
        full_dataset = PointCloudDataset(features_2d, features_3d, labels)
        
        # Split dataset
        dataset_size = len(full_dataset)
        val_size = int(validation_split * dataset_size)
        train_size = dataset_size - val_size
        
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size], 
            generator=torch.Generator().manual_seed(random_state)
        )
        
        # Create data loaders
        batch_size = self.config['training']['batch_size']
        num_workers = self.config.get('system', {}).get('num_workers', 4)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True
        )
        
        self.logger.info(f"Data prepared: {len(train_dataset)} train, {len(val_dataset)} val samples")
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for features_2d, features_3d, labels in progress_bar:
            # Move to device
            features_2d = features_2d.to(self.device)
            features_3d = features_3d.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs, confidences = self.model(features_2d, features_3d)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validation")
            
            for features_2d, features_3d, labels in progress_bar:
                # Move to device
                features_2d = features_2d.to(self.device)
                features_3d = features_3d.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs, confidences = self.model(features_2d, features_3d)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * correct / total:.2f}%'
                })
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> dict:
        """
        Complete training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Training history dictionary
        """
        num_epochs = self.config['training']['num_epochs']
        early_stopping_patience = self.config['training'].get('early_stopping_patience', 15)
        
        patience_counter = 0
        start_time = time.time()
        
        self.logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Log metrics
            self.logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            self.logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                self.best_model_state = self.model.state_dict()
                patience_counter = 0
                self.logger.info(f"New best model saved with validation accuracy: {val_acc:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        training_time = time.time() - start_time
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        self.logger.info(f"Best validation accuracy: {self.best_val_accuracy:.4f}")
        
        # Create training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'best_val_accuracy': self.best_val_accuracy,
            'training_time': training_time,
            'epochs_trained': len(self.train_losses)
        }
        
        return history
    
    def evaluate(self, test_loader: DataLoader) -> dict:
        """
        Evaluate the model on test data.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Evaluation metrics dictionary
        """
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_confidences = []
        
        with torch.no_grad():
            for features_2d, features_3d, labels in tqdm(test_loader, desc="Evaluation"):
                features_2d = features_2d.to(self.device)
                features_3d = features_3d.to(self.device)
                labels = labels.to(self.device)
                
                outputs, confidences = self.model(features_2d, features_3d)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': all_predictions,
            'labels': all_labels,
            'confidences': all_confidences
        }
        
        self.logger.info(f"Test Accuracy: {accuracy:.4f}")
        self.logger.info(f"Test Precision: {precision:.4f}")
        self.logger.info(f"Test Recall: {recall:.4f}")
        self.logger.info(f"Test F1-Score: {f1:.4f}")
        
        return metrics
    
    def save_model(self, filepath: str):
        """
        Save the model.
        
        Args:
            filepath: Path to save the model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'best_model_state_dict': self.best_model_state,
            'config': self.config,
            'best_val_accuracy': self.best_val_accuracy,
            'training_history': {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'train_accuracies': self.train_accuracies,
                'val_accuracies': self.val_accuracies
            }
        }
        
        torch.save(save_dict, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a saved model.
        
        Args:
            filepath: Path to the saved model
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_model_state = checkpoint.get('best_model_state_dict', checkpoint['model_state_dict'])
        self.best_val_accuracy = checkpoint.get('best_val_accuracy', 0.0)
        
        # Load training history if available
        history = checkpoint.get('training_history', {})
        self.train_losses = history.get('train_losses', [])
        self.val_losses = history.get('val_losses', [])
        self.train_accuracies = history.get('train_accuracies', [])
        self.val_accuracies = history.get('val_accuracies', [])
        
        self.logger.info(f"Model loaded from {filepath}")
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """
        Plot training history.
        
        Args:
            save_path: Path to save the plot
        """
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(epochs, self.train_losses, 'b-', label='Train Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracies
        ax2.plot(epochs, self.train_accuracies, 'b-', label='Train Accuracy')
        ax2.plot(epochs, self.val_accuracies, 'r-', label='Val Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def create_trainer(config: dict) -> FusionTrainer:
    """
    Factory function to create a trainer from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        FusionTrainer instance
    """
    return FusionTrainer(config)


if __name__ == "__main__":
    # Test the trainer
    print("Testing Fusion Trainer...")
    
    # Load config
    config = {
        'model': {
            'fusion': {
                'input_dim_2d': 2048,
                'input_dim_3d': 1024,
                'hidden_dim': 512,
                'num_classes': 10,
                'dropout': 0.3
            }
        },
        'training': {
            'batch_size': 16,
            'learning_rate': 0.001,
            'num_epochs': 5,
            'validation_split': 0.2,
            'early_stopping_patience': 15,
            'optimizer': {'type': 'Adam', 'weight_decay': 0.0001},
            'scheduler': {'type': 'StepLR', 'step_size': 30, 'gamma': 0.1}
        },
        'system': {
            'device': 'cuda',
            'num_workers': 4,
            'seed': 42
        },
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': 'logs/test_training.log'
        }
    }
    
    # Create trainer
    trainer = create_trainer(config)
    
    # Create dummy data
    n_samples = 100
    features_2d = np.random.randn(n_samples, 2048)
    features_3d = np.random.randn(n_samples, 1024)
    labels = np.random.randint(0, 10, n_samples)
    
    # Prepare data
    train_loader, val_loader = trainer.prepare_data(features_2d, features_3d, labels)
    
    # Train
    history = trainer.train(train_loader, val_loader)
    
    # Evaluate
    metrics = trainer.evaluate(val_loader)
    
    print("Trainer test completed successfully!")
    print(f"Best validation accuracy: {history['best_val_accuracy']:.4f}")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")