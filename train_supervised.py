import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import os
import cv2
import sys
import math
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import numpy as np

# --- FAST CPU OPTIMIZED CONFIG ---
DEVICE = "cpu" # Force CPU for predictable monitoring
IMAGE_SIZE = 256
BATCH_SIZE = 2 # Small batch for CPU
EPOCHS = 100 
L1_WEIGHT = 200
SSIM_WEIGHT = 100
LR = 5e-4 # Higher LR for fast convergence on small set

BASE_DIR = Path(__file__).parent
sys.path.append(str(BASE_DIR))
from models.generator import Generator
from data_prep.distorter import distort_image

class SSIM(nn.Module):
    def __init__(self):
        super().__init__()
        size = 11
        _1D = torch.Tensor([math.exp(-(x - size//2)**2/float(2*1.5**2)) for x in range(size)])
        _1D = (_1D / _1D.sum()).unsqueeze(1)
        win = _1D.mm(_1D.t()).float().unsqueeze(0).unsqueeze(0).expand(3, 1, size, size).contiguous()
        self.register_buffer('win', win)
    def forward(self, img1, img2):
        mu1, mu2 = F.conv2d(img1, self.win, padding=5, groups=3), F.conv2d(img2, self.win, padding=5, groups=3)
        mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1*mu2
        sig1_sq = F.conv2d(img1*img1, self.win, padding=5, groups=3) - mu1_sq
        sig2_sq = F.conv2d(img2*img2, self.win, padding=5, groups=3) - mu2_sq
        sig12 = F.conv2d(img1*img2, self.win, padding=5, groups=3) - mu1_mu2
        c1, c2 = 0.01**2, 0.03**2
        v = ((2*mu1_mu2 + c1)*(2*sig12 + c2))/((mu1_sq + mu2_sq + c1)*(sig1_sq + sig2_sq + c2))
        return 1 - v.mean()

class FastDemoDataset(Dataset):
    def __init__(self, root):
        self.ps = []
        # Take exactly ONE image from each monument subfolder for a 10-image fast-train set
        for subdir in sorted(os.listdir(root)):
            sub_path = Path(root) / subdir
            if sub_path.is_dir():
                imgs = sorted([sub_path/f for f in os.listdir(sub_path) if f.lower().endswith(('.jpg','.jpeg','.png'))])
                if imgs: self.ps.append(imgs[0]) 
        
        self.t = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])

    def __len__(self): return len(self.ps)
    def __getitem__(self, idx):
        p = self.ps[idx]
        cl_pil = Image.open(p).convert("RGB")
        dist_np = distort_image(p)
        dist_pil = Image.fromarray(cv2.cvtColor(dist_np, cv2.COLOR_BGR2RGB)) if dist_np is not None else cl_pil
        return self.t(dist_pil), self.t(cl_pil)

def run():
    model = Generator().to(DEVICE)
    l1, ssim = nn.L1Loss(), SSIM().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR)
    
    mp = BASE_DIR / "saved_models" / "restora_gen.pth"
    if mp.exists(): os.remove(str(mp))
    
    loader = DataLoader(FastDemoDataset(str(BASE_DIR / "dataset" / "clean")), batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"ULTRA-FAST DEMO TRAINING (Total: {len(loader.dataset)} images)...")
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            p = model(x)
            loss = l1(p, y)*L1_WEIGHT + ssim(p, y)*SSIM_WEIGHT
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item()
            
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{EPOCHS} -> Loss: {epoch_loss/len(loader):.4f}")
            torch.save(model.state_dict(), str(mp))
    print("Training Complete. Ready for Demo.")

if __name__ == "__main__":
    run()
