from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import json
import time
from scipy.spatial.distance import cosine
from deepface import DeepFace

app = FastAPI()

# Constants
FACE_DB = "facedata.json"
THRESHOLD = 0.7
Z_SCORE_THRESHOLD = 2.5  # You can tune this

@app.post("/")
async def authorize(image: UploadFile = File(...)):
    start_time = time.perf_counter()
    TARGET_RESPONSE_TIME = 15  # seconds

    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            elapsed = time.perf_counter() - start_time
            delay = TARGET_RESPONSE_TIME - elapsed
            if delay > 0:
                time.sleep(delay)
            return JSONResponse(content={"matched_user": "Unknown User"}, status_code=400)

        try:
            live_embedding = DeepFace.represent(img, model_name="Facenet", enforce_detection=True)[0]["embedding"]
        except Exception:
            elapsed = time.perf_counter() - start_time
            delay = TARGET_RESPONSE_TIME - elapsed
            if delay > 0:
                time.sleep(delay)
            return JSONResponse(content={"matched_user": "Unknown User"}, status_code=400)

        try:
            with open(FACE_DB, "r") as f:
                database = json.load(f)
        except Exception:
            elapsed = time.perf_counter() - start_time
            delay = TARGET_RESPONSE_TIME - elapsed
            if delay > 0:
                time.sleep(delay)
            return JSONResponse(content={"matched_user": "Unknown User"}, status_code=500)

        matched_user = "Unknown User"
        min_distance = float("inf")

        for username, embeddings in database.items():
            if "_mean" in username or "_std" in username:
                continue
            for stored_embedding in embeddings[:2]:  # Only compare first 2 embeddings to save time
                distance = cosine(live_embedding, stored_embedding)
                if distance < THRESHOLD and distance < min_distance:
                    matched_user = username
                    min_distance = distance

        anomalous = False
        if matched_user != "Unknown User":
            mean_key = f"{matched_user}_mean"
            std_key = f"{matched_user}_std"

            if mean_key in database and std_key in database:
                mean = np.array(database[mean_key])
                std = np.array(database[std_key])
                adjusted_std = np.clip(std, 0.15, None)
                z_scores = np.abs((np.array(live_embedding) - mean) / adjusted_std)

                outlier_count = np.sum(z_scores > Z_SCORE_THRESHOLD)
                outlier_ratio = outlier_count / len(z_scores)

                if outlier_ratio > 0.25:
                    anomalous = True

        elapsed = time.perf_counter() - start_time
        delay = TARGET_RESPONSE_TIME - elapsed
        if delay > 0:
            time.sleep(delay)
        else:
            print(f"[⏱️ WARNING] Execution took too long: {elapsed:.4f}s")

        return {
            "matched_user": matched_user,
            "result": "Live",
            "confidence": round(1 - min_distance, 4) if matched_user != "Unknown User" else 0.0,
            "anomalous": anomalous
        }

    except Exception as e:
        elapsed = time.perf_counter() - start_time
        delay = TARGET_RESPONSE_TIME - elapsed
        if delay > 0:
            time.sleep(delay)
        return JSONResponse(content={"matched_user": "Unknown User"}, status_code=500)
