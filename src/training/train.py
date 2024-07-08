import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.models.defect_detector import DefectDetector
from src.data.data_generation import generate_dataset
from typing import Tuple

def prepare_data() -> Tuple[DataLoader, DataLoader]:
    images, labels = generate_dataset(1000)
    train_images = torch.tensor(images[:800]).permute(0, 3, 1, 2).float() / 255.0
    train_labels = torch.tensor(labels[:800])
    test_images = torch.tensor(images[800:]).permute(0, 3, 1, 2).float() / 255.0
    test_labels = torch.tensor(labels[800:])

    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader

def train(model: nn.Module, train_loader: DataLoader, num_epochs: int = 10):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

def main():
    train_loader, test_loader = prepare_data()
    model = DefectDetector()
    train(model, train_loader)
    torch.save(model.state_dict(), "defect_detector.pth")
    print(1111111111111111111111111111111111111111111111111111111)

if __name__ == "__main__":
    main()