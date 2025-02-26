"""
Age Prediction Model Architectures

This module contains various model architectures for the age prediction task,
along with training and evaluation utilities.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from typing import Dict, List, Any


# Define different model architectures for experimentation
class BaselineModel(nn.Module):
    """Simple baseline model with one hidden layer"""

    def __init__(self, n_features):
        super(BaselineModel, self).__init__()
        self.fc1 = nn.Linear(n_features, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DeepModel(nn.Module):
    """Deeper model with three hidden layers"""

    def __init__(self, n_features):
        super(DeepModel, self).__init__()
        self.fc1 = nn.Linear(n_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class DropoutModel(nn.Module):
    """Model with dropout for regularization"""

    def __init__(self, n_features):
        super(DropoutModel, self).__init__()
        self.fc1 = nn.Linear(n_features, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


class BatchNormModel(nn.Module):
    """Model with batch normalization for better training"""

    def __init__(self, n_features):
        super(BatchNormModel, self).__init__()
        self.fc1 = nn.Linear(n_features, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x


class CompleteModel(nn.Module):
    """Advanced model with both batch normalization and dropout"""

    def __init__(self, n_features):
        super(CompleteModel, self).__init__()
        self.fc1 = nn.Linear(n_features, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x


def train_and_evaluate_model(
        model_class: nn.Module,
        model_name: str,
        n_features: int,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        test_loader: DataLoader,
        device: str,
        result_folder: str,
        lr: float = 0.001,
        epochs: int = 50
) -> Dict[str, Any]:
    """Train and evaluate a model architecture

    Args:
        model_class: The model class to instantiate
        model_name: Name of the model for saving/display
        n_features: Number of input features
        train_loader: DataLoader for training data
        valid_loader: DataLoader for validation data
        test_loader: DataLoader for test data
        device: Device to run the model on ('cuda' or 'cpu')
        result_folder: Directory to save results
        lr: Learning rate
        epochs: Number of training epochs

    Returns:
        Dict containing training results and metrics
    """
    # Create model
    model = model_class(n_features).to(device)
    print(f"\nTraining {model_name}...")

    # Define optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()

    # Initialize history and best validation loss
    history = {
        "train_loss": [],
        "train_rmse": [],
        "val_loss": [],
        "val_rmse": [],
        "epochs": []
    }
    best_val_loss = float('inf')

    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_rmse = 0.0
        train_samples = 0

        for embeddings, ages, _ in train_loader:
            embeddings, ages = embeddings.to(device), ages.to(device)

            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs.squeeze(), ages)
            loss.backward()
            optimizer.step()

            batch_size = embeddings.size(0)
            train_loss += loss.item() * batch_size
            train_rmse += torch.sqrt(torch.mean((outputs.squeeze() - ages) ** 2)).item() * batch_size
            train_samples += batch_size

        # Calculate average metrics
        train_loss /= train_samples
        train_rmse /= train_samples
        history["train_loss"].append(train_loss)
        history["train_rmse"].append(train_rmse)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_rmse = 0.0
        val_samples = 0

        with torch.no_grad():
            for embeddings, ages, _ in valid_loader:
                embeddings, ages = embeddings.to(device), ages.to(device)

                outputs = model(embeddings)
                loss = criterion(outputs.squeeze(), ages)

                batch_size = embeddings.size(0)
                val_loss += loss.item() * batch_size
                val_rmse += torch.sqrt(torch.mean((outputs.squeeze() - ages) ** 2)).item() * batch_size
                val_samples += batch_size

        # Calculate average metrics
        val_loss /= val_samples
        val_rmse /= val_samples
        history["val_loss"].append(val_loss)
        history["val_rmse"].append(val_rmse)
        history["epochs"].append(epoch)

        # Print epoch statistics (every 10 epochs)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_save_path = os.path.join(result_folder, f"{model_name}_best.pth")
            torch.save(model.state_dict(), model_save_path)

    # Load best model for evaluation
    model.load_state_dict(torch.load(os.path.join(result_folder, f"{model_name}_best.pth")))
    model.eval()

    # Evaluate on test set
    test_predictions = []
    test_true_values = []

    with torch.no_grad():
        for embeddings, ages, _ in test_loader:
            embeddings, ages = embeddings.to(device), ages.to(device)
            outputs = model(embeddings)
            test_predictions.extend(outputs.squeeze().cpu().numpy())
            test_true_values.extend(ages.cpu().numpy())

    # Convert to numpy arrays
    test_predictions = np.array(test_predictions)
    test_true_values = np.array(test_true_values)

    # Calculate metrics
    test_rmse = np.sqrt(np.mean((test_predictions - test_true_values) ** 2))
    test_mae = np.mean(np.abs(test_predictions - test_true_values))

    # Plot results
    plt.figure(figsize=(15, 10))

    # Training curves
    plt.subplot(2, 2, 1)
    plt.plot(history["epochs"], history["train_loss"], 'b-', label='Training')
    plt.plot(history["epochs"], history["val_loss"], 'r-', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MAE)')
    plt.title(f'{model_name} - Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(history["epochs"], history["train_rmse"], 'b-', label='Training')
    plt.plot(history["epochs"], history["val_rmse"], 'r-', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE (years)')
    plt.title(f'{model_name} - Training and Validation RMSE')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Predicted vs Actual
    plt.subplot(2, 2, 3)
    plt.scatter(test_true_values, test_predictions, alpha=0.5, color='blue')
    plt.plot([0, 100], [0, 100], 'r--')  # Perfect prediction line
    plt.xlabel('Actual Age')
    plt.ylabel('Predicted Age')
    plt.title(f'{model_name} - Predicted vs Actual (RMSE={test_rmse:.2f})')
    plt.grid(True, alpha=0.3)

    # Error distribution
    plt.subplot(2, 2, 4)
    plt.hist(test_predictions - test_true_values, bins=30, alpha=0.7, color='green')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Prediction Error (years)')
    plt.ylabel('Frequency')
    plt.title(f'{model_name} - Error Distribution (MAE={test_mae:.2f})')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(result_folder, f"{model_name}_results.png")
    plt.savefig(plot_path)
    plt.show()

    print(f"\n{model_name} Results:")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test MAE: {test_mae:.4f}")

    return {
        'model_name': model_name,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'best_val_loss': best_val_loss,
        'history': history
    }


def run_experiments(
        n_features: int,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        test_loader: DataLoader,
        device: str,
        result_folder: str,
        epochs: int = 50,
        lr: float = 0.001
) -> List[Dict[str, Any]]:
    """Run experiments with multiple model architectures

    Args:
        n_features: Number of input features
        train_loader: DataLoader for training data
        valid_loader: DataLoader for validation data
        test_loader: DataLoader for test data
        device: Device to run models on ('cuda' or 'cpu')
        result_folder: Directory to save results
        epochs: Number of training epochs
        lr: Learning rate

    Returns:
        List of dictionaries containing results for each model
    """
    results = [train_and_evaluate_model( # Baseline model
        BaselineModel, "Baseline", n_features,
        train_loader, valid_loader, test_loader,
        device, result_folder, lr, epochs
    ), train_and_evaluate_model( # Deep model
        DeepModel, "Deep", n_features,
        train_loader, valid_loader, test_loader,
        device, result_folder, lr, epochs
    ), train_and_evaluate_model( # Dropout model
        DropoutModel, "Dropout", n_features,
        train_loader, valid_loader, test_loader,
        device, result_folder, lr, epochs
    ), train_and_evaluate_model( # BatchNorm model
        BatchNormModel, "BatchNorm", n_features,
        train_loader, valid_loader, test_loader,
        device, result_folder, lr, epochs
    ), train_and_evaluate_model( # Complete model with BatchNorm and Dropout
        CompleteModel, "Complete", n_features,
        train_loader, valid_loader, test_loader,
        device, result_folder, lr, epochs
    )]

    # Compare results
    print("\nModel Comparison:")
    print("-" * 60)
    print(f"{'Model':<15} {'Test RMSE':<15} {'Test MAE':<15} {'Best Val Loss':<15}")
    print("-" * 60)

    for result in results:
        print(
            f"{result['model_name']:<15} {result['test_rmse']:<15.4f} {result['test_mae']:<15.4f} {result['best_val_loss']:<15.4f}")

    # Create comparison plot
    plt.figure(figsize=(14, 8))

    # RMSE comparison
    plt.subplot(1, 2, 1)
    names = [r['model_name'] for r in results]
    rmse_values = [r['test_rmse'] for r in results]
    plt.bar(names, rmse_values, color='skyblue')
    plt.ylabel('Test RMSE (years)')
    plt.title('Model Comparison - Test RMSE')
    plt.xticks(rotation=45)

    # MAE comparison
    plt.subplot(1, 2, 2)
    mae_values = [r['test_mae'] for r in results]
    plt.bar(names, mae_values, color='lightgreen')
    plt.ylabel('Test MAE (years)')
    plt.title('Model Comparison - Test MAE')
    plt.xticks(rotation=45)

    plt.tight_layout()
    comparison_path = os.path.join(result_folder, "model_comparison.png")
    plt.savefig(comparison_path)
    plt.show()

    return results


# Function to train a single model architecture
def train_model(
        model_class: nn.Module,
        model_name: str,
        n_features: int,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        device: str,
        result_folder: str,
        lr: float = 0.001,
        epochs: int = 50
) -> nn.Module:
    """Train a model and save the best version

    Args:
        model_class: The model class to instantiate
        model_name: Name of the model for saving
        n_features: Number of input features
        train_loader: DataLoader for training data
        valid_loader: DataLoader for validation data
        device: Device to use ('cuda' or 'cpu')
        result_folder: Directory to save the model
        lr: Learning rate
        epochs: Number of training epochs

    Returns:
        Trained model
    """
    # Create model
    model = model_class(n_features).to(device)
    print(f"Training {model_name}...")

    # Define optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()

    # Initialize best validation loss
    best_val_loss = float('inf')

    # Lists to track progress
    train_losses = []
    valid_losses = []

    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_samples = 0

        for embeddings, ages, _ in train_loader:
            embeddings, ages = embeddings.to(device), ages.to(device)

            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs.squeeze(), ages)
            loss.backward()
            optimizer.step()

            batch_size = embeddings.size(0)
            train_loss += loss.item() * batch_size
            train_samples += batch_size

        # Calculate average training loss
        epoch_train_loss = train_loss / train_samples
        train_losses.append(epoch_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_samples = 0

        with torch.no_grad():
            for embeddings, ages, _ in valid_loader:
                embeddings, ages = embeddings.to(device), ages.to(device)

                outputs = model(embeddings)
                loss = criterion(outputs.squeeze(), ages)

                batch_size = embeddings.size(0)
                val_loss += loss.item() * batch_size
                val_samples += batch_size

        # Calculate average validation loss
        epoch_val_loss = val_loss / val_samples
        valid_losses.append(epoch_val_loss)

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')

        # Save the best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            model_save_path = os.path.join(result_folder, f"{model_name}_best.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"Epoch {epoch + 1}: Saved new best model with validation loss: {epoch_val_loss:.4f}")

    # Plot training curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(train_losses)), train_losses, label='Training Loss')
    plt.plot(range(len(valid_losses)), valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MAE)')
    plt.title(f'{model_name} - Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(result_folder, f"{model_name}_training_curve.png"))
    plt.show()

    # Load the best model
    model.load_state_dict(torch.load(os.path.join(result_folder, f"{model_name}_best.pth")))

    return model


# Example of how to use these functions in a notebook:
"""
from model_architectures import (
    BaselineModel, DeepModel, DropoutModel, BatchNormModel, CompleteModel,
    run_experiments, train_model
)

# Option 1: Train and evaluate a single model
model = train_model(
    model_class=CompleteModel,
    model_name="CompleteModel",
    n_features=train_embeddings.shape[1],
    train_loader=train_loader,
    valid_loader=valid_loader,
    device=device,
    result_folder=resultFolder,
    lr=0.001,
    epochs=50
)

# Option 2: Run experiments with multiple model architectures
results = run_experiments(
    n_features=train_embeddings.shape[1], 
    train_loader=train_loader,
    valid_loader=valid_loader,
    test_loader=test_loader,
    device=device,
    result_folder=resultFolder,
    epochs=30,
    lr=0.001
)
"""