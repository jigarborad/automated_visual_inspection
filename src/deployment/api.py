from fastapi import FastAPI, File, UploadFile
import torch
import numpy as np
from PIL import Image
from src.models.defect_detector import DefectDetector

app = FastAPI()

model = DefectDetector()

model.load_state_dict(torch.load("defect_detector.pth"))
model.eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file)
    image = image.resize((224, 224))
    image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0)
    
    with torch.no_grad():
        prediction = model(image_tensor)
    
    defect_location = prediction[0].tolist()
    return {"defect_location": defect_location}