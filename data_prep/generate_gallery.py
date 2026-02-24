import os
import cv2
import random
import json
from pathlib import Path

# Add project root to path to import distorter
import sys
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))
from data_prep.distorter import distort_image

CLEAN_DIR = BASE_DIR / "dataset" / "clean"
GALLERY_DIR = BASE_DIR / "frontend" / "distorted_gallery"
MAPPING_FILE = BASE_DIR / "backend" / "gallery_mapping.json"

def generate():
    # Setup directories
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    
    # Collect all clean images
    all_clean_imgs = []
    for r, d, f in os.walk(str(CLEAN_DIR)):
        for file in f:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_clean_imgs.append(Path(r) / file)
                
    if not all_clean_imgs:
        print("No clean images found.")
        return

    mapping = {}
    print(f"Generating 50+ distorted images from {len(all_clean_imgs)} sources...")
    
    target_count = 60
    
    for i in range(target_count):
        clean_path = random.choice(all_clean_imgs)
        dist_np = distort_image(clean_path)
        if dist_np is None: continue
        
        dist_filename = f"distorted_{i:03d}.jpg"
        dist_save_path = GALLERY_DIR / dist_filename
        cv2.imwrite(str(dist_save_path), dist_np)
        mapping[dist_filename] = str(clean_path.relative_to(BASE_DIR))
        if i % 10 == 0:
            print(f"Generated {i} images...")

    with open(MAPPING_FILE, "w") as f:
        json.dump(mapping, f, indent=4)
        
    print(f"Done! Created {len(mapping)} images in {GALLERY_DIR}")
    print(f"Mapping saved to {MAPPING_FILE}")

if __name__ == "__main__":
    generate()
