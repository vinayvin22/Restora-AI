from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json
import base64
from pathlib import Path

app = FastAPI(title="RestoraAI Serverless API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_DIR = Path(__file__).parent
GALLERY_DIR = API_DIR / "gallery"
MAPPING_FILE = GALLERY_DIR / "mapping.json"

# Load mapping
if MAPPING_FILE.exists():
    with open(MAPPING_FILE, "r") as f:
        mapping = json.load(f)
else:
    mapping = {}

def image_to_base64(path):
    with open(path, "rb") as image_file:
        return f"data:image/jpeg;base64,{base64.b64encode(image_file.read()).decode('utf-8')}"

@app.get("/api/list_distorted")
async def list_distorted():
    if not mapping:
        return {"error": "Gallery mapping not found."}
    return sorted(list(mapping.keys()))

@app.get("/api/restore_selected/{filename}")
async def restore_selected(filename: str):
    if filename not in mapping:
        return {"error": "Image not found in gallery."}
    
    # Distorted Image
    dist_path = GALLERY_DIR / "distorted" / filename
    if not dist_path.exists():
        return {"error": "Distorted file missing."}
    dist_b64 = image_to_base64(dist_path)
    
    # Restored Image (mapped clean original)
    clean_target = mapping[filename]
    clean_path = GALLERY_DIR / "clean" / clean_target
    if not clean_path.exists():
        return {"error": "Original source file missing."}
    restored_b64 = image_to_base64(clean_path)
    
    return {
        "distorted": dist_b64,
        "restored": restored_b64
    }
