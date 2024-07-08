from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import torch
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO
from src.models.defect_detector import DefectDetector
import os

app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

model_path = "defect_detector.pth"
model = None

if os.path.exists(model_path):
    model = DefectDetector()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class_map = {0: 'crazing', 1: 'inclusion', 2: 'patches', 3: 'pitted_surface', 4: 'rolled-in-scale', 5: 'scratches'}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")

    image = Image.open(BytesIO(await file.read())).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        prediction = model(image_tensor)
    _, predicted_class = torch.max(prediction, 1)
    defect_type = class_map[predicted_class.item()]
    
    return {"defect_type": defect_type}

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("src/deployment/frontend.html") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)
