import torch
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import cv2

# Add project root to path
BASE_DIR = Path(__file__).parent
sys.path.append(str(BASE_DIR))

from models.generator import Generator

def diag():
    device = "cpu"
    model = Generator().to(device)
    
    # Simulate a batch that might have passed the check
    x = torch.randn(4, 3, 256, 256).to(device)
    y = torch.randn(4, 3, 256, 256).to(device)
    
    print(f"Testing model(x) with shape {x.shape}...")
    try:
        p = model(x)
        print(f"Output p shape: {p.shape}")
    except Exception as e:
        print(f"Model failed: {e}")
        return

    from train_supervised import SSIM, PerceptualLoss
    ssim = SSIM().to(device)
    vgg = PerceptualLoss().to(device)
    
    print("Testing SSIM...")
    try:
        s_loss = ssim(p, y)
        print(f"SSIM loss: {s_loss.item()}")
    except Exception as e:
        print(f"SSIM failed: {e}")

    print("Testing VGG...")
    try:
        v_loss = vgg(p, y)
        print(f"VGG loss: {v_loss.item()}")
    except Exception as e:
        print(f"VGG failed: {e}")

if __name__ == "__main__":
    diag()
