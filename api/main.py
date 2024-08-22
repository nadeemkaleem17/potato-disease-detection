from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
import uvicorn
import numpy as np
import tensorflow as tf
import os
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("../saved_models/model_1.keras")
def get_latest_model_path(model_dir="../saved_models"):
    models = [f for f in os.listdir(model_dir) if f.endswith('.keras') or f.endswith('.h5')]
    if not models:
        raise FileNotFoundError("No models found in the directory.")
    models.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    latest_model = models[-1]
    return os.path.join(model_dir, latest_model)
# Load the latest model
latest_model_path = get_latest_model_path()
MODEL = tf.keras.models.load_model(latest_model_path)

CLASS_NAMES = ['Early Blight', 'Late_Blight', 'Healthy']

@app.get('/ping')
async def ping():
    return "hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image_bytes = BytesIO(data)
    image = Image.open(image_bytes)
    np_image = np.array(image)
    return np_image


@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image, axis = 0)
    predictions = MODEL.predict(image_batch)
    index = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[index]
    confidence = np.max(predictions[0])
    print(predicted_class, confidence)
    return {"class": predicted_class, "confidence": float(confidence)}
    pass


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)