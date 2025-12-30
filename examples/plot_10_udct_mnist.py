"""
UDCT MNIST Classification
=========================

This example demonstrates how to use ``UDCTModule`` as a feature extractor
for image classification on the MNIST dataset.

Instead of using convolutional layers, this network:

1. Applies the Uniform Discrete Curvelet Transform (UDCT) to each image
2. Uses all curvelet coefficients as features (every pixel in every wedge)
3. Uses a linear layer to classify into 10 digit classes

The key insight is that curvelet coefficients capture directional information
at multiple scales, providing a meaningful representation for classification.

Architecture Overview
---------------------

- **Input**: MNIST images (28x28 grayscale)
- **Feature extraction**: UDCTModule transforms each image into curvelet coefficients
- **Features**: All coefficient values (every pixel in every wedge) form the feature vector
- **Classification**: Linear layer maps features to 10 classes
- **Batch processing**: ``torch.vmap`` enables efficient batched inference

Credits
-------

This example is adapted from the PyTorch MNIST example:
https://github.com/pytorch/examples/blob/main/mnist/main.py
"""

from __future__ import annotations

# %%
# sphinx_gallery_thumbnail_number = 2

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.manifold import TSNE
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

from curvelets.torch import UDCTModule

# %%
# UDCTNet: Curvelet-Based Classifier
# ###################################
#
# This network replaces convolutional layers with the curvelet transform.
# The key components are:
#
# 1. ``UDCTModule``: Computes curvelet coefficients for each image
# 2. Feature extraction: Uses all coefficient values as features
# 3. Linear classifier: Maps features to class probabilities
#
# We use ``torch.vmap`` to efficiently process batches of images.


class UDCTNet(nn.Module):
    """Neural network using UDCT for feature extraction."""

    def __init__(
        self,
        shape: tuple[int, int] = (28, 28),
        num_scales: int = 2,
        wedges_per_direction: int = 3,
    ) -> None:
        super().__init__()
        self.udct = UDCTModule(
            shape=shape,
            num_scales=num_scales,
            wedges_per_direction=wedges_per_direction,
        )
        # Precompute number of features via dummy forward pass during init
        # UDCTModule returns flattened coefficients (all pixels in all wedges)
        with torch.inference_mode():
            dummy = torch.zeros(shape)
            n_features = self.udct(dummy).numel()
        self.linear = nn.Linear(n_features, 10)

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract all curvelet coefficients from a single 2D image."""
        # UDCTModule returns flattened coefficients - take abs to get real features
        return self.udct(x).abs()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with batched curvelet feature extraction."""
        # x: (batch, 1, 28, 28) -> squeeze to (batch, 28, 28)
        x = x.squeeze(1)
        # Use vmap for batch processing
        features = torch.vmap(self._extract_features)(x)
        return F.log_softmax(self.linear(features), dim=1)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification (for visualization)."""
        x = x.squeeze(1)
        return torch.vmap(self._extract_features)(x)


# %%
# Training and Testing Functions
# ##############################
#
# Standard PyTorch training loop with negative log-likelihood loss.


def train(
    model: nn.Module,
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    epoch: int,
    log_interval: int = 100,
) -> float:
    """Train the model for one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % log_interval == 0:
            print(
                f"Train Epoch: {epoch} "
                f"[{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100.0 * batch_idx / len(train_loader):.0f}%)]\t"
                f"Loss: {loss.item():.6f}"
            )
    return total_loss / len(train_loader)


def test(
    model: nn.Module,
    device: torch.device,
    test_loader: torch.utils.data.DataLoader,
) -> float:
    """Evaluate the model on the test set and return average loss."""
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.inference_mode():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)

    print(
        f"\nTest set: Average loss: {test_loss:.4f}, "
        f"Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)\n"
    )
    return test_loss


# %%
# Main Training Script
# ####################
#
# We use a simplified configuration suitable for a gallery example:
#
# - 2 epochs (for quick demonstration)
# - Batch size of 64
# - Adadelta optimizer with learning rate scheduling

# Device selection
if torch.accelerator.is_available():
    device = torch.accelerator.current_accelerator()
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Data loading
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST("./data", train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)

# %%
# Model Initialization
# ####################
#
# Create the UDCTNet model with 2 scales and 3 wedges per direction.
# With all curvelet coefficients as features, we get a high-dimensional
# feature vector that preserves all transform information.

model = UDCTNet(shape=(28, 28), num_scales=2, wedges_per_direction=3).to(device)
print(f"Model: {model}")
print(f"Linear layer input features: {model.linear.in_features}")

# %%
# Training Loop
# #############
#
# Train for 2 epochs with Adadelta optimizer and step learning rate decay.

optimizer = optim.Adadelta(model.parameters(), lr=1.0)
scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

num_epochs = 3
train_losses: list[float] = []
test_losses: list[float] = []

for epoch in range(1, num_epochs + 1):
    train_loss = train(model, device, train_loader, optimizer, epoch)
    test_loss = test(model, device, test_loader)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    scheduler.step()

# %%
# Loss Plot
# #########
#
# Visualize the training and test loss over epochs.

fig, ax = plt.subplots(figsize=(8, 5))
epochs = range(1, num_epochs + 1)
ax.plot(epochs, train_losses, "o-", label="Train Loss", linewidth=2, markersize=8)
ax.plot(epochs, test_losses, "s-", label="Test Loss", linewidth=2, markersize=8)
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Loss", fontsize=12)
ax.set_title("UDCT MNIST Classification: Training Progress", fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xticks(epochs)
plt.tight_layout()
plt.show()

# %%
# t-SNE Feature Visualization
# ###########################
#
# Visualize the curvelet features in 2D using t-SNE. Each digit class is shown
# in a different color, revealing how well the features separate the classes.
#
# t-SNE (t-distributed Stochastic Neighbor Embedding) is a nonlinear
# dimensionality reduction technique that preserves local structure.

# Extract features from a subset of training data for visualization
n_samples = 2000
subset_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=n_samples, shuffle=True
)
data_batch, labels_batch = next(iter(subset_loader))

# Extract features using the trained model
model.eval()
with torch.inference_mode():
    features_batch = model.get_features(data_batch.to(device)).cpu().numpy()
labels_np = labels_batch.numpy()

# Apply t-SNE to reduce to 2 dimensions
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
features_tsne = tsne.fit_transform(features_batch)

fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(
    features_tsne[:, 0],
    features_tsne[:, 1],
    c=labels_np,
    cmap="tab10",
    vmin=-0.5,
    vmax=9.5,
    alpha=0.6,
    s=10,
)
cbar = plt.colorbar(scatter, ax=ax, ticks=range(10))
cbar.set_label("Digit Class", fontsize=12)
ax.set_xlabel("t-SNE 1", fontsize=12)
ax.set_ylabel("t-SNE 2", fontsize=12)
ax.set_title("UDCT Features: 2D t-SNE Projection", fontsize=14)
plt.tight_layout()
plt.show()

# %%
# Results
# #######
#
# The UDCT-based classifier provides a simple baseline for MNIST classification.
# By using all curvelet coefficients as features, we preserve all transform
# information and let the linear layer learn the optimal weighting.
#
# Key takeaways:
#
# - ``UDCTModule`` integrates seamlessly with PyTorch's autograd system
# - ``torch.vmap`` enables efficient batched processing
# - Using all coefficients as features provides a rich representation
