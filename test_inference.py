import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import sys
from pathlib import Path
import numpy as np
import cv2

# Add project root to path
BASE_DIR = Path(__file__).parent
sys.path.append(str(BASE_DIR))

from models.generator import Generator
from data_prep.distorter import distort_image

def test_inference():
    device = "cpu"
    model = Generator().to(device)
    model_path = BASE_DIR / "saved_models" / "restora_gen.pth"
    
    if not model_path.exists():
        print("Model not found.")
        return
        
    model.load_state_dict(torch.load(str(model_path), map_location=device))
    model.eval()
    
    # Take an image from dataset/clean
    clean_dir = BASE_DIR / "dataset" / "clean"
    if not clean_dir.exists():
        print("Dataset clean not found.")
        return
        
    img_files = []
    for r, d, f in os.walk(str(clean_dir)):
        for file in f:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_files.append(Path(r) / file)
                
    if not img_files:
        print("No images in clean dataset.")
        return
        
    img_path = img_files[0]
    print(f"Testing on {img_path}")
    
    # Clean image
    clean_img = Image.open(str(img_path)).convert("RGB")
    
    # Distort it
    dist_np = distort_image(img_path)
    if dist_np is None:
        dist_img = clean_img
    else:
        dist_img = Image.fromarray(cv2.cvtColor(dist_np, cv2.COLOR_BGR2RGB))
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    input_tensor = transform(dist_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        
    # Post-process
    output = output.squeeze(0).cpu()
    output = (output * 0.5 + 0.5).clamp(0, 1)
    output_pil = transforms.ToPILImage()(output)
    
    # Save results
    output_pil.save("test_restored.png")
    dist_img.resize((256, 256)).save("test_distorted.png")
    print("Saved test_restored.png and test_distorted.png")

if __name__ == "__main__":
    test_inference()
