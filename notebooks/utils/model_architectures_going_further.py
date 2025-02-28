"""
Implementation of Age Bin Classification Models

This module contains model architectures and functions for classifying age into bins
rather than performing direct regression.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Any, Tuple, Optional


class AgeBinDataset(Dataset):
    """Dataset for age bin classification"""

    def __init__(self, embeddings, ages, filenames, bin_width=5):
        self.embeddings = embeddings.clone().detach()
        self.actual_ages = ages.clone().detach()
        # Convert ages to bin classes
        self.bin_width = bin_width
        self.age_bins = (ages / bin_width).floor().long()
        self.filenames = filenames

        # Calculate number of bins
        self.num_bins = int(self.age_bins.max().item()) + 1

        assert len(self.embeddings) == len(self.age_bins)
        assert len(self.embeddings) == len(self.filenames)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.age_bins[idx], self.filenames[idx]

    def get_actual_age(self, idx):
        """Get the actual age (not binned) for an index"""
        return self.actual_ages[idx]


# Classification head models
class BaselineClassifier(nn.Module):
    """Simple classifier for age bins"""

    def __init__(self, n_features, num_bins):
        super(BaselineClassifier, self).__init__()
        self.fc1 = nn.Linear(n_features, 64)
        self.fc2 = nn.Linear(64, num_bins)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DeepClassifier(nn.Module):
    """Deeper classifier for age bins"""

    def __init__(self, n_features, num_bins):
        super(DeepClassifier, self).__init__()
        self.fc1 = nn.Linear(n_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_bins)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class RegularizedClassifier(nn.Module):
    """Classifier with batch normalization and dropout"""

    def __init__(self, n_features, num_bins):
        super(RegularizedClassifier, self).__init__()
        self.fc1 = nn.Linear(n_features, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.3)

        self.fc4 = nn.Linear(64, num_bins)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)

        x = self.fc4(x)
        return x


def train_classifier(
        model: nn.Module,
        model_name: str,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        device: str,
        result_folder: str,
        bin_width: int = 5,
        lr: float = 0.001,
        epochs: int = 50
) -> nn.Module:
    """Train an age bin classifier

    Args:
        model: The model to train
        model_name: Name of the model for saving
        train_loader: DataLoader for training data
        valid_loader: DataLoader for validation data
        device: Device to use ('cuda' or 'cpu')
        result_folder: Directory to save the model
        bin_width: Width of age bins in years
        lr: Learning rate
        epochs: Number of training epochs

    Returns:
        Trained model
    """
    model = model.to(device)
    print(f"Training {model_name}...")

    # Define optimizer and criterion (use cross entropy for classification)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Initialize tracking variables
    best_val_loss = float('inf')
    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []

    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_samples = 0

        for embeddings, age_bins, _ in train_loader:
            embeddings, age_bins = embeddings.to(device), age_bins.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(embeddings)
            loss = criterion(outputs, age_bins)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Calculate metrics
            _, predicted = torch.max(outputs, 1)
            batch_size = embeddings.size(0)
            train_loss += loss.item() * batch_size
            train_correct += (predicted == age_bins).sum().item()
            train_samples += batch_size

        # Calculate epoch metrics
        epoch_train_loss = train_loss / train_samples
        epoch_train_acc = train_correct / train_samples
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_samples = 0

        with torch.no_grad():
            for embeddings, age_bins, _ in valid_loader:
                embeddings, age_bins = embeddings.to(device), age_bins.to(device)

                # Forward pass
                outputs = model(embeddings)
                loss = criterion(outputs, age_bins)

                # Calculate metrics
                _, predicted = torch.max(outputs, 1)
                batch_size = embeddings.size(0)
                val_loss += loss.item() * batch_size
                val_correct += (predicted == age_bins).sum().item()
                val_samples += batch_size

        # Calculate epoch metrics
        epoch_val_loss = val_loss / val_samples
        epoch_val_acc = val_correct / val_samples
        valid_losses.append(epoch_val_loss)
        valid_accuracies.append(epoch_val_acc)

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, "
                  f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, "
                  f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            model_save_path = os.path.join(result_folder, f"{model_name}_best.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved new best model with validation loss: {epoch_val_loss:.4f}")

    # Plot training curves
    plt.figure(figsize=(15, 6))

    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(range(len(train_losses)), train_losses, 'b-', label='Training')
    plt.plot(range(len(valid_losses)), valid_losses, 'r-', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Cross Entropy)')
    plt.title(f'{model_name} - Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(range(len(train_accuracies)), train_accuracies, 'b-', label='Training')
    plt.plot(range(len(valid_accuracies)), valid_accuracies, 'r-', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{model_name} - Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(result_folder, f"{model_name}_training.png"))
    plt.show()

    # Load best model
    model.load_state_dict(torch.load(os.path.join(result_folder, f"{model_name}_best.pth")))

    return model


def evaluate_classifier(
        model: nn.Module,
        test_loader: DataLoader,
        device: str,
        result_folder: str,
        model_name: str,
        bin_width: int = 5
) -> Dict[str, Any]:
    """Evaluate an age bin classifier on test data

    Args:
        model: Trained classifier model
        test_loader: DataLoader for test data
        device: Device to use ('cuda' or 'cpu')
        result_folder: Directory to save results
        model_name: Name of the model for plots
        bin_width: Width of age bins in years

    Returns:
        Dictionary of evaluation metrics
    """
    model = model.to(device)
    model.eval()

    # Initialize tracking variables
    all_preds = []
    all_true = []
    all_actual_ages = []  # For MAE in years
    bin_midpoints = []  # For predicted actual ages

    # Get dataset from loader to access actual ages
    dataset = test_loader.dataset

    with torch.no_grad():
        for i, (embeddings, age_bins, _) in enumerate(test_loader):
            embeddings, age_bins = embeddings.to(device), age_bins.to(device)

            # Get batch indices for the original dataset
            batch_size = embeddings.size(0)
            batch_indices = list(range(i * test_loader.batch_size,
                                       min((i + 1) * test_loader.batch_size, len(dataset))))

            # Forward pass
            outputs = model(embeddings)
            _, predicted = torch.max(outputs, 1)

            # Store predictions and true values
            all_preds.extend(predicted.cpu().numpy())
            all_true.extend(age_bins.cpu().numpy())

            # Get actual ages for MAE calculation
            for idx in batch_indices:
                if idx < len(dataset):
                    all_actual_ages.append(dataset.get_actual_age(idx).item())

                    # For the predicted bin, calculate midpoint age
                    pred_bin = predicted[batch_indices.index(idx)].item()
                    bin_midpoints.append(pred_bin * bin_width + bin_width / 2)

    # Calculate metrics
    accuracy = np.mean(np.array(all_preds) == np.array(all_true))

    # For age prediction: MAE in years
    mae_years = np.mean(np.abs(np.array(bin_midpoints) - np.array(all_actual_ages)))

    # Number of bins off (0 means correct bin, 1 means adjacent bin, etc.)
    bins_off = np.abs(np.array(all_preds) - np.array(all_true))
    mean_bins_off = np.mean(bins_off)
    median_bins_off = np.median(bins_off)

    # Confusion matrix
    cm = confusion_matrix(all_true, all_preds)
    num_bins = len(set(all_true))

    # Visualize confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True if num_bins <= 20 else False,
                fmt="d", cmap="Blues", cbar=True)
    plt.xlabel('Predicted Age Bin')
    plt.ylabel('True Age Bin')
    plt.title(f'Confusion Matrix (Bin Width: {bin_width} years)')
    plt.savefig(os.path.join(result_folder, f"{model_name}_confusion.png"))
    plt.show()

    # Scatter plot of predicted vs. true ages
    plt.figure(figsize=(10, 8))
    plt.scatter(all_actual_ages, bin_midpoints, alpha=0.5)
    plt.plot([0, max(all_actual_ages)], [0, max(all_actual_ages)], 'r--')
    plt.xlabel('Actual Age (years)')
    plt.ylabel('Predicted Age (bin midpoint, years)')
    plt.title(f'Predicted vs. Actual Age (MAE: {mae_years:.2f} years)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(result_folder, f"{model_name}_predictions.png"))
    plt.show()

    # Print metrics
    print(f"\nEvaluation Metrics for {model_name} (Bin Width: {bin_width} years):")
    print(f"Classification Accuracy: {accuracy:.4f}")
    print(f"Mean Absolute Error (years): {mae_years:.2f}")
    print(f"Mean Bins Off: {mean_bins_off:.2f}")
    print(f"Median Bins Off: {median_bins_off:.2f}")

    # Return metrics
    return {
        'model_name': model_name,
        'bin_width': bin_width,
        'accuracy': accuracy,
        'mae_years': mae_years,
        'mean_bins_off': mean_bins_off,
        'median_bins_off': median_bins_off,
        'confusion_matrix': cm
    }


def compare_bin_widths(
        n_features: int,
        train_embeddings: torch.Tensor,
        train_ages: torch.Tensor,
        train_filenames: List[str],
        valid_embeddings: torch.Tensor,
        valid_ages: torch.Tensor,
        valid_filenames: List[str],
        test_embeddings: torch.Tensor,
        test_ages: torch.Tensor,
        test_filenames: List[str],
        device: str,
        result_folder: str,
        bin_widths: List[int] = [5, 10],
        epochs: int = 50,
        batch_size: int = 256
) -> Dict[int, Dict[str, Any]]:
    """Compare classifier performance with different bin widths

    Args:
        n_features: Number of features in embeddings
        train_embeddings, train_ages, train_filenames: Training data
        valid_embeddings, valid_ages, valid_filenames: Validation data
        test_embeddings, test_ages, test_filenames: Test data
        device: Device to use ('cuda' or 'cpu')
        result_folder: Directory to save results
        bin_widths: List of bin widths to try (in years)
        epochs: Number of training epochs
        batch_size: Batch size for training

    Returns:
        Dictionary of results for each bin width
    """
    results = {}

    for bin_width in bin_widths:
        print(f"\n{'=' * 50}")
        print(f"Training with bin width: {bin_width} years")
        print(f"{'=' * 50}")

        # Create datasets and dataloaders
        train_dataset = AgeBinDataset(train_embeddings, train_ages, train_filenames, bin_width)
        valid_dataset = AgeBinDataset(valid_embeddings, valid_ages, valid_filenames, bin_width)
        test_dataset = AgeBinDataset(test_embeddings, test_ages, test_filenames, bin_width)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Create and train model
        num_bins = train_dataset.num_bins
        model = RegularizedClassifier(n_features, num_bins)
        model_name = f"AgeBin_{bin_width}"

        # Train model
        model = train_classifier(
            model=model,
            model_name=model_name,
            train_loader=train_loader,
            valid_loader=valid_loader,
            device=device,
            result_folder=result_folder,
            bin_width=bin_width,
            epochs=epochs
        )

        # Evaluate model
        eval_results = evaluate_classifier(
            model=model,
            test_loader=test_loader,
            device=device,
            result_folder=result_folder,
            model_name=model_name,
            bin_width=bin_width
        )

        # Store results
        results[bin_width] = eval_results

    # Compare results
    plt.figure(figsize=(10, 6))

    widths = list(results.keys())
    mae_values = [results[w]['mae_years'] for w in widths]
    acc_values = [results[w]['accuracy'] for w in widths]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot MAE (left axis)
    ax1.set_xlabel('Bin Width (years)')
    ax1.set_ylabel('MAE (years)', color='tab:blue')
    ax1.plot(widths, mae_values, 'o-', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Plot Accuracy (right axis)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', color='tab:red')
    ax2.plot(widths, acc_values, 'o-', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title('Age Bin Classification: Effect of Bin Width')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(result_folder, "bin_width_comparison.png"))
    plt.show()

    # Print summary
    print("\nBin Width Comparison Summary:")
    print(f"{'Bin Width':<10} {'Accuracy':<10} {'MAE (years)':<15} {'Mean Bins Off':<15}")
    print("-" * 50)
    for width in widths:
        r = results[width]
        print(f"{width:<10} {r['accuracy']:.4f}     {r['mae_years']:.2f}           {r['mean_bins_off']:.2f}")

    return results