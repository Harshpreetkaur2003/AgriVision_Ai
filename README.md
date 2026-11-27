
# AgriVision AI

AI-powered crop leaf disease detection and health analysis web application.

## Technology Stack

- Python 3.12
- FastAPI
- HTML, CSS, JavaScript
- OpenCV
- TensorFlow/Keras
- Google Colab GPU
- Kaggle PlantVillage
- SQLite (planned)

## Application Pages

1. Home Page
2. Leaf Disease Analysis Page

## Project Development Phases

1. Project setup and website foundation
2. Dataset collection and exploration
3. OpenCV image processing
4. Multi-crop disease classification
5. Disease segmentation and severity
6. Recommendation engine
7. AI and website integration
8. Testing and evaluation

## Dataset

Initially, we will explore PlantVillage.
Multiple crop classes will be selected based on
dataset availability and quality.

## Run Application

Activate the virtual environment:

.venv\Scripts\Activate.ps1

Install requirements:

pip install -r requirements.txt

Start the application:

uvicorn backend.main:app --reload

Open:

http://127.0.0.1:8000

## Current Status

Phase 1: Website foundation and image upload.

Disease prediction will be integrated after
training and evaluating the AI model.
