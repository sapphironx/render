from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

#CONFIGURATION
MODEL_PATH = "face_antispoofing_model.keras"
IMG_SIZE = 224
THRESHOLD = 0.5

# Load the model once at startup 
print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded.")

# Initialize FastAPI app
app = FastAPI()

@app.post("/")
async def predict_spoof(file: UploadFile = File(...)):
    try:
        # Read uploaded image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Preprocess image
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        # Predict
        pred = model.predict(img)[0][0]
        label = "Live" if pred < THRESHOLD else "Spoof"
        confidence = float(pred) if label == "Spoof" else 1 - float(pred)

        return {
            "result": label,
            "confidence": round(confidence, 4)
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

