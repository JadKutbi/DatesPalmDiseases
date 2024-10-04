# ConvNeXt Model with MixUp for Date Palm Leaves Disease Classification

This repository contains the code for training and evaluating a ConvNeXt model with MixUp data augmentation, applied to classify diseases in date palm leaves. The dataset is sourced from Mendeley Data, and the model leverages transfer learning using pretrained weights on ImageNet.

## Dataset

**Dataset Reference**:  
Namoun, Abdallah; Alkhodre, Ahmad B.; Abi Sen, Adnan Ahmad; Alsaawy, Yazed; Almoamari, Hani (2024), "Diseases of date palm leaves dataset", Mendeley Data, V2, doi: [10.17632/g684ghfxvg.2](https://doi.org/10.17632/g684ghfxvg.2)


## Model Overview

The model uses **ConvNeXt Tiny** architecture with MixUp data augmentation. It is trained using a Cross-Entropy loss function with MixUp, and the optimizer is Adam. Early stopping is applied with patience of 6 epochs.

## Key Features

- **MixUp Data Augmentation**: Helps improve generalization by mixing inputs and target labels.
- **ConvNeXt Tiny**: A powerful model pretrained on ImageNet.
- **Early Stopping**: Stops training when validation loss no longer improves.
- **Test Accuracy**: The best model achieves an accuracy of 97.42% on the test set.

## Requirements

- Python 3.8+
- PyTorch 1.10+
- Torchvision
- NumPy

## Setup and Usage

1. Clone the repository
git clone https://github.com/yourusername/date-palm-leaf-disease-classification.git
cd date-palm-leaf-disease-classification
2. Install the required dependencies
pip install -r requirements.txt
3. Dataset Setup
Download the dataset from Mendeley Data here and place it in the dataset folder, ensuring the structure is organized by class folders.

References
PyTorch: https://pytorch.org/
Torchvision Models: https://pytorch.org/vision/stable/models.html
MixUp: Zhang et al., "mixup: Beyond Empirical Risk Minimization" (https://arxiv.org/abs/1710.09412)
**Dataset Reference**:  
Namoun, Abdallah; Alkhodre, Ahmad B.; Abi Sen, Adnan Ahmad; Alsaawy, Yazed; Almoamari, Hani (2024), "Diseases of date palm leaves dataset", Mendeley Data, V2, doi: [10.17632/g684ghfxvg.2](https://doi.org/10.17632/g684ghfxvg.2)
