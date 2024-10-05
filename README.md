"""
Author: Jad Kutbi

This script is designed to evaluate the performance of a ConvNeXt model trained on an image dataset
for disease classification. The model is tested on a held-out test set, and the predictions are visualized
along with the true labels. The code uses MixUp data augmentation during training and includes functionality
for early stopping.

Citation:
Kutbi, J. (2024). Disease Classification Using ConvNeXt and MixUp Data Augmentation. [GitHub repository or paper title]
"""

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
git clone https://github.com/JadKutbi/DatesPalmDiseases.git

cd DatesPalmDiseases
2. Install the required dependencies

pip install -r requirements.txt
3. Dataset Setup

Download the dataset from Mendeley Data here and place it in the dataset folder, ensuring the structure is organized by class folders.

4. Training the Model
To train the model, run:

python train.py

5. Evaluation
Once training is complete, you can evaluate the model on the test set:

python evaluate.py

## References

PyTorch: https://pytorch.org/

Torchvision Models: https://pytorch.org/vision/stable/models.html

MixUp: Zhang et al., "mixup: Beyond Empirical Risk Minimization" (https://arxiv.org/abs/1710.09412)

**Dataset Reference**:  

Namoun, Abdallah; Alkhodre, Ahmad B.; Abi Sen, Adnan Ahmad; Alsaawy, Yazed; Almoamari, Hani (2024), "Diseases of date palm leaves dataset", Mendeley Data, V2, doi: [10.17632/g684ghfxvg.2](https://doi.org/10.17632/g684ghfxvg.2)


"""
Author: Jad Kutbi

This script is designed to evaluate the performance of a ConvNeXt model trained on an image dataset
for disease classification. The model is tested on a held-out test set, and the predictions are visualized
along with the true labels. The code uses MixUp data augmentation during training and includes functionality
for early stopping.

Citation:
Kutbi, J. (2024). Disease Classification Using ConvNeXt and MixUp Data Augmentation. [GitHub repository or paper title]
"""
