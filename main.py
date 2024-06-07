import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet18
from torchvision.datasets import ImageFolder
import numpy as np
import matplotlib.pyplot as plt

# Select device: GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mean and standard deviation for normalization
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

# Data preprocessing and augmentation
data_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Resize all images to 224x224 pixels
        transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally for data augmentation
        transforms.RandomRotation(10),  # Randomly rotate images for data augmentation
        transforms.ToTensor(),  # Convert images to PyTorch tensors and scale pixel values to [0, 1]
        transforms.Normalize(
            mean=mean, std=std
        ),  # Normalize images with specified mean and std
    ]
)

# Load training dataset with defined transformations and labels according to the folder stucture
train_dataset = ImageFolder(
    "./brain-tumor-mri-dataset/Training", transform=data_transforms
)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Load validation dataset with defined transformations
val_dataset = ImageFolder(
    "./brain-tumor-mri-dataset/Testing", transform=data_transforms
)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


# Define a convolutional neural network
class TumorClassifier(nn.Module):
    def __init__(self, num_classes):
        super(TumorClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)  # Extract features using convolutional layers
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.classifier(x)  # Classify the input
        return x


# Initialize the model, loss function, and optimizer
model = TumorClassifier(num_classes=4)
model.to(device)

criterion = nn.CrossEntropyLoss()  # Loss function for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

# Lists to store training history
train_losses, val_losses, train_accuracies, val_accuracies = ([] for i in range(4))

# Training loop
num_epochs = 20
best_val_accuracy = 0.0

for epoch in range(num_epochs):
    """
        Each epoch constits of a training phase and a validation phase.
        The validation phase in each epoch enusures that the quality
        of the model improves or stays the same with each epoch.
    """
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  # Zero the gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss between predited and true
        loss.backward()  # Backward pass
        """loss.backwards() it computes the gradients of the loss function with respect to all
        the learnable parameters in the model. These gradients indicate
        the direction and magnitude of change that needs to be made to each
        parameter to decrease the loss. It essentially calculates how much
        each parameter contributed to the error in the prediction."""
        optimizer.step()  # Update parameters

        train_loss += loss.item()  # Accumulate training loss
        _, predicted = torch.max(outputs, 1)  # Get predictions
        total += labels.size(0)  # Total number of samples
        correct += (predicted == labels).sum().item()  # Number of correct predictions

    train_accuracy = correct / total  # Compute training accuracy
    train_losses.append(train_loss)  # Store training loss
    train_accuracies.append(train_accuracy)  # Store training accuracy

    # Validation phase
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss

            val_loss += loss.item()  # Accumulate validation loss
            _, predicted = torch.max(outputs, 1)  # Get predictions
            total += labels.size(0)  # Total number of samples
            correct += (
                (predicted == labels).sum().item()
            )  # Number of correct predictions

    val_loss /= len(val_loader)  # Compute average validation loss
    val_accuracy = correct / total  # Compute validation accuracy
    val_losses.append(val_loss)  # Store validation loss
    val_accuracies.append(val_accuracy)  # Store validation accuracy

    # Print epoch statistics
    print(
        f"Epoch [{epoch+1}/{num_epochs}], "
        f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2%}, "
        f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2%}"
    )

    # Save the best model based on validation accuracy
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), "best_model.pth")

# Final validation accuracy
accuracy = correct / total
print(f"Validation Accuracy: {accuracy:.2%}")

# Visualize training history
plt.figure(figsize=(12, 6))

# Plot training and validation loss
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss History")

# Plot training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Training Accuracy")
plt.plot(val_accuracies, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy History")

plt.tight_layout()
plt.show()
