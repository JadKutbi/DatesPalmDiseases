"""
Author: Jad Kutbi

This script is designed to evaluate the performance of a ConvNeXt model trained on an image dataset
for disease classification. The model is tested on a held-out test set, and the predictions are visualized
along with the true labels. The code uses MixUp data augmentation during training and includes functionality
for early stopping.

Citation:
Kutbi, J. (2024). Disease Classification Using ConvNeXt and MixUp Data Augmentation. [GitHub repository or paper title]
"""


import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import torch.nn as nn
import torch.optim as optim
import numpy as np

# === MixUp Data Augmentation ===
def mixup_data(x, y, alpha=1.0):
    """Applies MixUp data augmentation to create mixed inputs and target pairs."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Loss function for MixUp augmented data."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# === Dataset Loading and Preprocessing ===
data_dir = 'dataset'  # Ensure your dataset is in the 'dataset' folder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ImageNet preprocessing and augmentations
transform = transforms.Compose([
    ConvNeXt_Tiny_Weights.IMAGENET1K_V1.transforms(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomErasing(p=0.5)
])

# Load the full dataset
full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Split the dataset into train, validation, and test sets
train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

# Data loaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# === Model Configuration ===
model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)

# Modify the final classifier layer to match the number of classes
num_classes = len(full_dataset.classes)
model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
model = model.to(device)

# Loss function, optimizer, and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

# === Validation Function ===
def validate(model, val_loader):
    """Validates the model on the validation dataset."""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = val_loss / len(val_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# === Early Stopping Parameters ===
best_val_loss = float('inf')
patience = 6
epochs_no_improve = 0
early_stop = False

# === Training Loop ===
num_epochs = 50
best_model_path = "best_convnext_model_mix.pth"

for epoch in range(num_epochs):
    if early_stop:
        print("Early stopping activated.")
        break

    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Apply MixUp
        inputs, targets_a, targets_b, lam = mixup_data(inputs, labels)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Validate the model
    val_loss, val_acc = validate(model, val_loader)
    scheduler.step(val_loss)
    print(f"Epoch [{epoch + 1}/{num_epochs}], "
          f"Train Loss: {running_loss / len(train_loader):.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")

    # Check for early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved at epoch {epoch + 1}")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            early_stop = True
            print("Early stopping triggered.")

print("Training complete.")
