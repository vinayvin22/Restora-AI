from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import os
import sys
import base64
import json
from pathlib import Path
import cv2
import numpy as np

# Add project root to path
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))
from models.generator import Generator

app = FastAPI(title="RestoraAI Gallery API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAPPING_FILE = BASE_DIR / "backend" / "gallery_mapping.json"
DISTORTED_DIR = BASE_DIR / "frontend" / "distorted_gallery"

# Load mapping
mapping = {}
if MAPPING_FILE.exists():
    with open(MAPPING_FILE, "r") as f:
        mapping = json.load(f)

def image_to_base64(path):
    with open(path, "rb") as image_file:
        return f"data:image/jpeg;base64,{base64.b64encode(image_file.read()).decode('utf-8')}"

@app.get("/list_distorted")
async def list_distorted():
    if not mapping:
        return {"error": "Gallery mapping not found. Run generate_gallery.py first."}
    return sorted(list(mapping.keys()))

@app.get("/restore_selected/{filename}")
async def restore_selected(filename: str):
    if filename not in mapping:
        return {"error": "Image not found in gallery."}
    
    # 1. Distorted Image (selected by user)
    dist_path = DISTORTED_DIR / filename
    if not dist_path.exists():
        return {"error": "Distorted file missing."}
    dist_b64 = image_to_base64(dist_path)
    
    # 2. Restored Image (mapped clean original)
    clean_rel_path = mapping[filename]
    clean_path = BASE_DIR / clean_rel_path
    if not clean_path.exists():
        return {"error": "Original source file missing."}
    restored_b64 = image_to_base64(clean_path)
    
    return {
        "distorted": dist_b64,
        "restored": restored_b64
    }

# Static mounts
# Mount the entire gallery so frontend can access them directly if needed
app.mount("/distorted_gallery", StaticFiles(directory=str(DISTORTED_DIR)), name="distorted_gallery")
app.mount("/", StaticFiles(directory=str(BASE_DIR / "frontend"), html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
