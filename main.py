from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn

# Define your model class
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define your model layers here
        self.linear = nn.Linear(3, 1)  # Example: A simple linear layer

    def forward(self, x):
        return self.linear(x)

# Create an instance of the model
model = MyModel()

# Load the state dictionary with error handling
try:
    model.load_state_dict(torch.load("best3.pt"), strict=False)
except Exception as e:
    print("Error loading state_dict:", e)

# Ensure that the model is in evaluation mode
model.eval()

# Create FastAPI instance
app = FastAPI()

# Define input data structure
class Item(BaseModel):
    data: list

# Define prediction endpoint
@app.post("/predict/")
async def predict(item: Item):
    try:
        # Convert input data to tensor
        input_tensor = torch.tensor(item.data, dtype=torch.float32).unsqueeze(0)  # Add batch dimension if needed

        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
        
        # Convert output to list and return
        return {"prediction": output.squeeze().tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Define a root endpoint
@app.get("/")
async def read_root():
    return {"message": "Welcome to the FastAPI and PyTorch integration example"}
