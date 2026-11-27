
from pathlib import Path
from functools import lru_cache
import json

import cv2
import numpy as np
import tensorflow as tf

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Form,
    HTTPException
)

router = APIRouter(
    prefix="/api",
    tags=["Image Analysis"]
)

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    BASE_DIR / "models" / "agrivision_mobilenetv2_final.keras"
)

CLASS_PATH = (
    BASE_DIR / "models" / "class_indices.json"
)

IMG_SIZE = (224, 224)

MAX_FILE_SIZE = 10 * 1024 * 1024

ALLOWED_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp"
}

ALLOWED_CROPS = {
    "tomato",
    "potato",
    "pepper",
    "other"
}


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

@lru_cache(maxsize=1)
def get_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


# --------------------------------------------------
# LOAD CLASS MAPPING
# --------------------------------------------------

@lru_cache(maxsize=1)
def get_class_names():

    if not CLASS_PATH.exists():
        raise FileNotFoundError(
            f"Class mapping not found: {CLASS_PATH}"
        )

    with open(
        CLASS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        class_indices = json.load(file)

    return {
        int(index): name
        for name, index in class_indices.items()
    }


# --------------------------------------------------
# DETECT CROP NAME
# --------------------------------------------------

def get_crop_name(class_name):

    if class_name.startswith("Tomato"):
        return "tomato"

    if class_name.startswith("Potato"):
        return "potato"

    if class_name.startswith("Pepper"):
        return "pepper"

    return "unknown"


# --------------------------------------------------
# FORMAT DISEASE NAME
# --------------------------------------------------

def format_disease_name(class_name):

    name = class_name

    name = name.replace("___", " ")
    name = name.replace("__", " ")
    name = name.replace("_", " ")

    return " ".join(name.split())


# --------------------------------------------------
# MODEL STATUS API
# --------------------------------------------------

@router.get("/model-status")
async def model_status():

    return {
        "model_file_exists": MODEL_PATH.exists(),
        "class_mapping_exists": CLASS_PATH.exists(),
        "model_loaded": get_model.cache_info().currsize > 0,
        "model_name": "MobileNetV2",
        "classes": 15
    }


# --------------------------------------------------
# IMAGE UPLOAD AND DISEASE PREDICTION API
# --------------------------------------------------

@router.post("/upload")
async def upload_leaf(
    file: UploadFile = File(...),
    crop: str = Form(...)
):

    # Normalize crop selection
    crop = crop.lower().strip()

    # Validate selected crop
    if crop not in ALLOWED_CROPS:

        raise HTTPException(
            status_code=400,
            detail="Unsupported crop selection."
        )

    # Validate image format
    if file.content_type not in ALLOWED_TYPES:

        raise HTTPException(
            status_code=400,
            detail="Upload JPG, PNG or WEBP images only."
        )

    # Read uploaded image
    contents = await file.read()

    if not contents:

        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty."
        )

    # Validate image size
    if len(contents) > MAX_FILE_SIZE:

        raise HTTPException(
            status_code=400,
            detail="Maximum image size is 10 MB."
        )

    # Decode image using OpenCV
    image_array = np.frombuffer(
        contents,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:

        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )

    # Original image dimensions
    height, width = image.shape[:2]

    # Convert BGR to RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Resize image to 224x224
    image = cv2.resize(
        image,
        IMG_SIZE,
        interpolation=cv2.INTER_AREA
    )

    # Convert image to float32
    image = image.astype(np.float32)

    # MobileNetV2 preprocessing
    image = tf.keras.applications.mobilenet_v2.preprocess_input(
        image
    )

    # Add batch dimension
    image = np.expand_dims(
        image,
        axis=0
    )

    # --------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------

    try:

        model = get_model()

        class_names = get_class_names()

        predictions = model.predict(
            image,
            verbose=0
        )[0]

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Model prediction failed: {str(error)}"
        )

    # --------------------------------------------------
    # GET TOP PREDICTION
    # --------------------------------------------------

    predicted_index = int(
        np.argmax(predictions)
    )

    predicted_class = class_names[predicted_index]

    confidence = (
        float(predictions[predicted_index]) * 100
    )

    predicted_crop = get_crop_name(
        predicted_class
    )

    disease_name = format_disease_name(
        predicted_class
    )

    # --------------------------------------------------
    # VALIDATE SELECTED CROP
    # --------------------------------------------------

    if crop != "other" and crop != predicted_crop:

        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    "Selected crop does not match "
                    "the predicted crop."
                ),
                "selected_crop": crop,
                "detected_crop": predicted_crop,
                "predicted_disease": disease_name,
                "confidence": round(confidence, 2),
                "suggestion": (
                    f"Model predicted {predicted_crop}. "
                    "Please verify the image and crop selection."
                )
            }
        )

    # --------------------------------------------------
    # GET TOP 3 PREDICTIONS
    # --------------------------------------------------

    top_indices = np.argsort(
        predictions
    )[::-1][:3]

    top_predictions = []

    for index in top_indices:

        index = int(index)

        class_name = class_names[index]

        top_predictions.append({

            "class_name": class_name,

            "disease_name": format_disease_name(
                class_name
            ),

            "crop": get_crop_name(
                class_name
            ),

            "confidence": round(
                float(predictions[index]) * 100,
                2
            )

        })

    # --------------------------------------------------
    # FINAL API RESPONSE
    # --------------------------------------------------

    return {

        "success": True,

        "message": (
            "Disease prediction completed successfully."
        ),

        "selected_crop": crop,

        "detected_crop": predicted_crop,

        "predicted_class": predicted_class,

        "disease_name": disease_name,

        "confidence": round(
            confidence,
            2
        ),

        "original_width": width,

        "original_height": height,

        "processed_size": [
            224,
            224,
            3
        ],

        "model_name": "MobileNetV2",

        "top_predictions": top_predictions

    }
