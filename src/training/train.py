import torch
import torch.nn as nn
import torch.optim as optim
from src.models.defect_detector import DefectDetector
from src.data.data_processing import get_data_loaders

def train(model, train_loader, test_loader, num_epochs=10, device='cpu', log_file='training_logs.txt'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    with open(log_file, 'w') as f:
        f.write('Epoch, Loss, Accuracy\n')  # Write the header for the log file

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # Validation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            epoch_loss = running_loss / len(train_loader)
            epoch_accuracy = 100 * correct / total

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
            
            # Write to log file
            f.write(f"{epoch+1}, {epoch_loss:.4f}, {epoch_accuracy:.2f}%\n")

    return model

def main():
    train_loader, test_loader = get_data_loaders('data/NEU-DET/train/images')
    model = DefectDetector()
    trained_model = train(model, train_loader, test_loader)
    torch.save(trained_model.state_dict(), "defect_detector.pth")

if __name__ == "__main__":
    main()
