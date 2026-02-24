import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
import os
import cv2
import sys
import math
import torch.nn.functional as F
from pathlib import Path
from PIL import Image

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))
from models.generator import Generator
from models.discriminator import Discriminator

def distort_image_safe(path):
    """Fallback distorter implementation to ensure 3-channels."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None: return None
    # Simple distortion if the main one fails or for robustness
    noise = np.random.randint(0, 50, img.shape, dtype='uint8')
    return cv2.add(img, noise)

import numpy as np

# --- Hyperparameters ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 2e-4
BATCH_SIZE = 4 
IMAGE_SIZE = 256
L1_LAMBDA, SSIM_LAMBDA, VGG_LAMBDA, ADV_LAMBDA = 150, 50, 10, 0.05
EPOCHS = 30

class SupervisedDataset(Dataset):
    def __init__(self, root):
        self.paths = [Path(r)/f for r,d,fs in os.walk(root) for f in fs if f.lower().endswith(('.jpg','.jpeg','.png'))]
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        try:
            p = self.paths[idx]
            # Force RGB at the source
            clean = Image.open(p).convert("RGB")
            
            # Use original distort image but catch errors
            from data_prep.distorter import distort_image
            dist_np = distort_image(p)
            if dist_np is None:
                 dist = clean.copy() # Fallback to clean
            else:
                 dist = Image.fromarray(cv2.cvtColor(dist_np, cv2.COLOR_BGR2RGB)).convert("RGB")
            
            x = self.transform(dist)
            y = self.transform(clean)
            
            # NUCLEAR CHANNEL CHECK
            if x.shape[0] != 3: x = x.repeat(3, 1, 1) if x.shape[0] == 1 else x[:3,:,:]
            if y.shape[0] != 3: y = y.repeat(3, 1, 1) if y.shape[0] == 1 else y[:3,:,:]
            
            return x, y
        except Exception as e:
            return torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE)), torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE))

class VGGPerceptual(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.layers = nn.ModuleList([vgg[:4], vgg[4:9], vgg[9:16]]).to(DEVICE).eval()
        for p in self.parameters(): p.requires_grad = False
        self.l1 = nn.L1Loss()

    def forward(self, x, y):
        # ImageNet normalization
        m = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
        s = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)
        x = (x * 0.5 + 0.5 - m) / s
        y = (y * 0.5 + 0.5 - m) / s
        loss = 0
        for lay in self.layers:
            loss += self.l1(lay(x), lay(y))
        return loss

class SSIM(nn.Module):
    def __init__(self):
        super().__init__()
        size = 11
        _1D = torch.Tensor([math.exp(-(x - size//2)**2/float(2*1.5**2)) for x in range(size)])
        _1D = (_1D / _1D.sum()).unsqueeze(1)
        _2D = _1D.mm(_1D.t()).float().unsqueeze(0).unsqueeze(0)
        self.register_buffer('win', _2D.expand(3, 1, size, size).contiguous())

    def forward(self, img1, img2):
        mu1 = F.conv2d(img1, self.win, padding=5, groups=3)
        mu2 = F.conv2d(img2, self.win, padding=5, groups=3)
        mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1*mu2
        sig1_sq = F.conv2d(img1*img1, self.win, padding=5, groups=3) - mu1_sq
        sig2_sq = F.conv2d(img2*img2, self.win, padding=5, groups=3) - mu2_sq
        sig12 = F.conv2d(img1*img2, self.win, padding=5, groups=3) - mu1_mu2
        c1, c2 = 0.01**2, 0.03**2
        v = ((2*mu1_mu2 + c1)*(2*sig12 + c2))/((mu1_sq + mu2_sq + c1)*(sig1_sq + sig2_sq + c2))
        return 1 - v.mean()

def train():
    gen = Generator(in_channels=3).to(DEVICE)
    disc = Discriminator(in_channels=3).to(DEVICE)
    vgg, ssim, l1, bce = VGGPerceptual(), SSIM(), nn.L1Loss(), nn.BCEWithLogitsLoss()
    
    # Reset
    mp = ROOT_DIR / "saved_models" / "restora_gen.pth"
    if mp.exists(): os.remove(str(mp))
    
    opt_g = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    opt_d = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    
    dataset = SupervisedDataset(str(ROOT_DIR / "dataset" / "clean"))
    print(f"DEBUG: Dataset has {len(dataset)} items.")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print("STARTING SUPERVISED HIGH-FIDELITY TRAINING...")
    for epoch in range(1, EPOCHS + 1):
        for idx, (x, y) in enumerate(loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            # Safety shape check
            if x.shape[1] != 3:
                print(f"Skipping batch {idx} due to unexpected shape: {x.shape}")
                continue

            # Update D
            y_fake = gen(x)
            d_real_loss = bce(disc(x, y), torch.ones_like(disc(x, y)))
            d_fake_loss = bce(disc(x, y_fake.detach()), torch.zeros_like(disc(x, y_fake.detach())))
            d_loss = (d_real_loss + d_fake_loss) / 2
            opt_d.zero_grad(); d_loss.backward(); opt_d.step()
            
            # Update G
            g_adv = bce(disc(x, y_fake), torch.ones_like(disc(x, y_fake))) * ADV_LAMBDA
            g_rec = l1(y_fake, y) * L1_LAMBDA + ssim(y_fake, y) * SSIM_LAMBDA + vgg(y_fake, y) * VGG_LAMBDA
            g_loss = g_rec + g_adv
            opt_g.zero_grad(); g_loss.backward(); opt_g.step()
            
            if idx % 10 == 0:
                print(f"E{epoch} [{idx}/{len(loader)}] - G: {g_loss.item():.2f} (L1: {l1(y_fake, y).item()*L1_LAMBDA:.1f})")
                torch.save(gen.state_dict(), str(ROOT_DIR / "saved_models" / "restora_gen.pth"))

if __name__ == "__main__":
    train()
