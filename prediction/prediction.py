import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import pandas as pd

# device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# neural network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(27, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

# dataset class
class FireDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x.float(), y.float()  # Ensure the data is in float format


# Load data from CSV
df = pd.read_csv('/Users/paulafrindte/Library/CloudStorage/OneDrive-TUM/Dokumente/8_Nerden/cassini/github_links/training_data.csv')
df.drop(['Unnamed: 0', 'time', 'valid_time', 'number', 'step'], axis=1, inplace=True)
print(df.shape)
training_predictor = torch.tensor(df['fire'].values)
training_data = torch.tensor(df.drop('fire', axis=1).values)


# Create dataset and dataloader
dataset = FireDataset(training_data, training_predictor)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize the model, loss function, and optimizer
model = NeuralNetwork().to(device)
loss_fn = nn.MSELoss()  # Binary classification loss
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
def train(model, dataloader, loss_fn, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        running_loss = 0.0
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Forward pass
            outputs = model(X).squeeze(1)  # Squeeze to match the shape
            loss = loss_fn(outputs, y)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch % 100 == 0:
                print(f"Batch {batch}, Loss: {loss.item():.4f}")

        print(f"Epoch {epoch+1} Loss: {running_loss/len(dataloader):.4f}")

    print("Training complete!")


# Run the training process
train(model, train_loader, loss_fn, optimizer, epochs=10)

# Save the model
torch.save(model.state_dict(), "regression_model.pth")
print("Model saved to regression_model.pth")
