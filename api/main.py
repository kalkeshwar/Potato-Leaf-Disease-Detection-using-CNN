from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os

app = FastAPI()

# Configure CORS
origins = [
    "http://localhost",
    "http://localhost:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
path_directory =os.getcwd()+"\potato-disease-classification\potato.h5"
# Define file path for the model
print(path_directory)
model_path = "C:/Users/91703/Desktop/Major Project Potato Leaf disease Detection using cnn/major project/potato leaf disease detection/potatoes.h5"  # Adjust the path as needed

# Check if the model file or directory exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file or directory not found at {model_path}")

# Load the model
try:
    MODEL = tf.keras.models.load_model(model_path)
except Exception as e:
    raise RuntimeError(f"Error loading the model: {e}")

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)).convert("RGB").resize((256,256)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    
    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    print(predicted_class)
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
