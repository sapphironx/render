
import os
import json
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from deepface import DeepFace
import cv2
import tempfile
from typing import List

app = FastAPI()

# Configuration
DATA_FILE = "facedata.json"
SAVE_PATH = os.path.join(os.getcwd(), DATA_FILE)
EMBEDDING_MODEL = "Facenet"

# Helper: Load and Save Database
def load_face_data():
    if os.path.exists(SAVE_PATH):
        with open(SAVE_PATH, "r") as f:
            return json.load(f)
    else:
        return {}

def save_face_data(data):
    with open(SAVE_PATH, "w") as f:
        json.dump(data, f)

# POST API: Register User Face (expects 3 images)
@app.post("/")
async def register_face(username: str = Form(...), images: List[UploadFile] = File(...), overwrite: bool = Form(False)):
    if len(images) != 3:
        return JSONResponse(content={"success": False, "message": "Exactly 3 images are required for registration."}, status_code=400)

    try:
        face_db = load_face_data()

        if not overwrite and username in face_db:
                return JSONResponse(
                    content={"success": False, "message": "Username taken, please try another username."},
                    status_code=409
                )

        embeddings = []
        for image in images:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                contents = await image.read()
                temp_file.write(contents)
                temp_image_path = temp_file.name

            img = cv2.imread(temp_image_path)
            if img is None:
                return JSONResponse(content={"success": False, "message": "One of the images is invalid."}, status_code=400)

            try:
                embedding_obj = DeepFace.represent(img, model_name=EMBEDDING_MODEL, enforce_detection=True)
                if isinstance(embedding_obj, list):
                    embedding = embedding_obj[0]['embedding']
                    embeddings.append(embedding)
            except Exception as e:
                return JSONResponse(content={"success": False, "message": f"Face detection failed: {str(e)}"}, status_code=400)
            finally:
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)

        # Save embeddings and compute baseline
        face_db[username] = embeddings
        user_embeddings = np.array(embeddings)
        mean = np.mean(user_embeddings, axis=0)
        computed_std = np.std(user_embeddings, axis=0)

        # Apply minimum spread to avoid over-sensitive anomaly detection
        min_allowed_std = 0.25
        adjusted_std = np.clip(computed_std, min_allowed_std, None)

        face_db[f"{username}_mean"] = mean.tolist()
        face_db[f"{username}_std"] = adjusted_std.tolist()

        print(f"[REGISTER DEBUG] mean(std): {np.mean(computed_std):.6f}, adjusted(std): {np.mean(adjusted_std):.6f}")


        save_face_data(face_db)
        return JSONResponse(content={"success": True, "message": f"User {username} registered successfully."})

    except Exception as e:
        return JSONResponse(content={"success": False, "message": f"Server error: {str(e)}"}, status_code=500)
