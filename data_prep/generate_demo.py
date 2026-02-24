import os
import random
import cv2
import shutil
from pathlib import Path
from distorter import distort_image

# Configuration
SOURCE_DIR = Path("../dataset/clean")
TARGET_DIR = Path("../static/demo_images")
NUM_DEMO_IMAGES = 50

def get_all_images(source_dir):
    """Recursively find all image files in the dataset."""
    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    image_paths = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(valid_extensions):
                image_paths.append(Path(root) / file)
    return image_paths

def main():
    # Ensure target directory exists
    if TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR)
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get all available images
    all_images = get_all_images(SOURCE_DIR)
    
    if len(all_images) < NUM_DEMO_IMAGES:
        print(f"Warning: Only {len(all_images)} images found. Using all for demo.")
        selected_images = all_images
    else:
        selected_images = random.sample(all_images, NUM_DEMO_IMAGES)
    
    print(f"Generating {len(selected_images)} distorted demo images...")
    
    for i, img_path in enumerate(selected_images):
        distorted = distort_image(img_path)
        if distorted is not None:
            # Scale to a standard size for demo (e.g., 512x512)
            distorted = cv2.resize(distorted, (512, 512))
            save_name = f"demo_{i+1}.jpg"
            cv2.imwrite(str(TARGET_DIR / save_name), distorted)
            # Also keep a copy of the original for ground truth comparison if needed
            # but requirement says gallery of distorted images for quick selection.
            print(f"Saved: {save_name}")

if __name__ == "__main__":
    main()
