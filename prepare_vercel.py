import os
import json
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).parent
API_DIR = BASE_DIR / "api"
GALLERY_DIR = API_DIR / "gallery"

# Ensure directories exist
GALLERY_DIR.mkdir(parents=True, exist_ok=True)

# Original paths
MAPPING_FILE = BASE_DIR / "backend" / "gallery_mapping.json"
DISTORTED_DIR = BASE_DIR / "frontend" / "distorted_gallery"

if MAPPING_FILE.exists():
    with open(MAPPING_FILE, "r") as f:
        mapping = json.load(f)
else:
    print("No mapping found.")
    mapping = {}

# Clean previous gallery if needed
try:
    if (GALLERY_DIR / "clean").exists():
        shutil.rmtree(GALLERY_DIR / "clean")
    if (GALLERY_DIR / "distorted").exists():
        shutil.rmtree(GALLERY_DIR / "distorted")
except Exception as e:
    pass

(GALLERY_DIR / "clean").mkdir(exist_ok=True)
(GALLERY_DIR / "distorted").mkdir(exist_ok=True)

new_mapping = {}

for dist_file, clean_rel_path in mapping.items():
    dist_src = DISTORTED_DIR / dist_file
    clean_src = BASE_DIR / clean_rel_path
    
    if dist_src.exists() and clean_src.exists():
        # Copy distorted image
        shutil.copy2(dist_src, GALLERY_DIR / "distorted" / dist_file)
        
        # Copy clean image safely by flatting name
        clean_target = clean_rel_path.replace("\\", "_").replace("/", "_")
        shutil.copy2(clean_src, GALLERY_DIR / "clean" / clean_target)
        
        new_mapping[dist_file] = clean_target

# Save new simplified mapping
with open(GALLERY_DIR / "mapping.json", "w") as f:
    json.dump(new_mapping, f, indent=4)

print(f"Prepared Vercel gallery with {len(new_mapping)} pairs.")
