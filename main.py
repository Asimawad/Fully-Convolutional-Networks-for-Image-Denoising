import os
import torch
from torch.utils.data import DataLoader
from src.data_preprocessing import load_data, LFWDataset
from src.model import AutoEncoder
from src.train import train_model, validate_model
from src.evaluate import test_model, visualize_results
import matplotlib.pyplot as plt

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths and parameters
data_path = "./data/lfwcrop_color/faces"  # Path to the dataset
batch_size = 32
num_epochs = 50
learning_rate = 0.001
std_noise = 0.1

# Step 1: Load and preprocess the data
print("Loading and processing data...")
processed = load_data(data_path, std=std_noise)

# Step 2: Create datasets and data loaders
train_dataset = LFWDataset(processed, split="train")
valid_dataset = LFWDataset(processed, split="validation")
test_dataset = LFWDataset(processed, split="test")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Step 3: Initialize the model
model = AutoEncoder().to(device)

# Step 4: Set up the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
criterion = torch.nn.MSELoss()

# Step 5: Train and validate the model
train_losses, validation_losses = [], []
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train_loss = train_model(model, train_loader, optimizer, criterion, device)
    validation_loss = validate_model(model, valid_loader, criterion, device)
    train_losses.append(train_loss)
    validation_losses.append(validation_loss)

# Save the trained model
model_path = "./models/AutoEncoder.pth"
os.makedirs(os.path.dirname(model_path), exist_ok=True)
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Step 6: Plot training and validation losses
plt.figure(figsize=(10, 6))
plt.plot(range(num_epochs), train_losses, label="Training Loss")
plt.plot(range(num_epochs), validation_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()

# Step 7: Test the model
print("Evaluating the model on the test set...")
test_loss = test_model(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f}")

# Step 8: Visualize results
print("Visualizing results...")
visualize_results(model, test_loader, device)
