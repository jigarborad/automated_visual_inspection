import argparse
import os
from src.training.train import main as train_main
from src.deployment.api import app
import uvicorn

def main():
    parser = argparse.ArgumentParser(description="Train model or run API.")
    parser.add_argument('--mode', type=str, choices=['train', 'api'], required=True, help="Mode to run the script in: 'train' or 'api'")
    args = parser.parse_args()

    if args.mode == 'train':
        train_main()
    elif args.mode == 'api':
        model_path = "defect_detector.pth"
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}. Please train the model first using `--mode train`.")
            return
        
        uvicorn.run("src.deployment.api:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()
