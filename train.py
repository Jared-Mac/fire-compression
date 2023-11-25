import torch
import torchvision.models as models
import torch.nn as nn
from torch.nn.functional import sigmoid
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

import os 


# Assuming the dataset directory structure and transformations are defined as before
dataset_path = 'the_wildfire_dataset'
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

# Load datasets
train_dataset = datasets.ImageFolder(os.path.join(dataset_path, 'train'), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(dataset_path, 'val'), transform=transform)
test_dataset = datasets.ImageFolder(os.path.join(dataset_path, 'test'), transform=transform)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
test_dataset_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

# Define the same model structure as was used for training
model = models.resnet101(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 1)  # Assuming it's the modified ResNet-101 for binary classification

# Load the model weights
model_path = 'trained_resnet101.pth'  # Replace with your .pth file path
model.load_state_dict(torch.load(model_path))
# Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Move the model to GPU if available
device = torch.device("cuda")
model = model.to(device)

# TensorBoard SummaryWriter
writer = SummaryWriter('./logs/')

# Training loop
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device).float()

        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Log training loss
    avg_train_loss = running_loss / len(train_loader)
    writer.add_scalar('Loss/Train', avg_train_loss, epoch)

    # Validation
    model.eval()
    running_corrects = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = sigmoid(model(inputs).squeeze())  # Apply sigmoid
            predicted = outputs > 0.5  # Threshold to get binary class
            total += labels.size(0)
            running_corrects += (predicted == labels).sum().item()

    # Log validation accuracy
    val_accuracy = 100 * running_corrects / total
    writer.add_scalar('Accuracy/Val', val_accuracy, epoch)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss:.4f}, '
          f'Validation Accuracy: {val_accuracy:.2f}%')

# Close the TensorBoard writer
writer.close()

# Save the trained model
torch.save(model.state_dict(), 'fire_resnet101.pth')
