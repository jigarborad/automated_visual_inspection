import torch
from src.models.defect_detector import DefectDetector

def test_model_output():
    model = DefectDetector()
    input_tensor = torch.randn(1, 3, 224, 224)
    output = model(input_tensor)
    assert output.shape == (1, 2), "Output shape is incorrect"

def test_model_training():
    model = DefectDetector()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    input_tensor = torch.randn(32, 3, 224, 224)
    target = torch.randn(32, 2)
    
    for _ in range(10):
        optimizer.zero_grad()
        output = model(input_tensor)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
    assert loss.item() < 1.0, "Model is not learning"

if __name__ == "__main__":
    test_model_output()
    test_model_training()
    print("All tests passed!")