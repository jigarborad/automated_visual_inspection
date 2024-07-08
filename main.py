from src.training.train import main as train_main
from src.deployment.api import app
import uvicorn

if __name__ == "__main__":
    # Train the model
    #train_main()
    
    # Run the API
    uvicorn.run(app, host="0.0.0.0", port=8000)