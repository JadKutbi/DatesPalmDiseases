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
from torch.utils.data import DataLoader
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import torch.nn as nn

# === Dataset Loading and Preprocessing ===
data_dir = 'dataset'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ImageNet preprocessing for ConvNeXt and transformations
transform = transforms.Compose([
    ConvNeXt_Tiny_Weights.IMAGENET1K_V1.transforms(),
])

# Load the full dataset and split into test set
full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
_, _, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])

# Data loader for the test set
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# === Model Configuration ===
model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)

# Modify the classifier for your dataset
num_classes = len(full_dataset.classes)
model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
model = model.to(device)

# Load the best model weights
best_model_path = "best_convnext_model_mix.pth"
model.load_state_dict(torch.load(best_model_path))
model.eval()  # Set model to evaluation mode

# === Testing the Model ===
correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Store predictions for further analysis if needed
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_accuracy = 100 * correct / total
print(f'Test Accuracy: {test_accuracy:.2f}%')

# (Optional) Print a few sample predictions
for i in range(10):
    print(f'Predicted: {test_dataset.dataset.classes[all_preds[i]]}, '
          f'Actual: {test_dataset.dataset.classes[all_labels[i]]}')
