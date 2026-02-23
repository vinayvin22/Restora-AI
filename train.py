import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import cv2
import numpy as np

print("Script Started...")


# Transform
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

dataset = datasets.ImageFolder("dataset", transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

# Damage Function
def add_damage(img):
    img = img.permute(1,2,0).numpy()
    img = ((img * 0.5) + 0.5) * 255   # denormalize
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = np.ascontiguousarray(img)   # ⭐ FIX

    h, w, _ = img.shape

    # Add random cracks
    for _ in range(10):
        x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
        x2, y2 = np.random.randint(0, w), np.random.randint(0, h)
        cv2.line(img, (x1,y1), (x2,y2), (0,0,0), 2)

    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5   # normalize back
    img = torch.tensor(img).permute(2,0,1)

    return img


# Generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,64,4,2,1),
            nn.ReLU(),
            nn.Conv2d(64,128,4,2,1),
            nn.ReLU(),
            nn.ConvTranspose2d(128,64,4,2,1),
            nn.ReLU(),
            nn.ConvTranspose2d(64,3,4,2,1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,64,4,2,1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64,128,4,2,1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128*32*32,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

G = Generator().to(device)
D = Discriminator().to(device)

loss_fn = nn.BCELoss()
opt_G = torch.optim.Adam(G.parameters(), lr=0.0002)
opt_D = torch.optim.Adam(D.parameters(), lr=0.0002)

# Training
for epoch in range(30):
    for real_imgs,_ in loader:
        real_imgs = real_imgs.to(device)
        damaged_imgs = torch.stack([add_damage(img.cpu()).to(device) for img in real_imgs])

        real_label = torch.ones(real_imgs.size(0),1).to(device)
        fake_label = torch.zeros(real_imgs.size(0),1).to(device)

        fake_imgs = G(damaged_imgs)
        d_loss = loss_fn(D(real_imgs), real_label) + loss_fn(D(fake_imgs.detach()), fake_label)

        opt_D.zero_grad(); d_loss.backward(); opt_D.step()

        l1_loss = nn.L1Loss()
        g_loss = loss_fn(D(fake_imgs), real_label) + 100 * l1_loss(fake_imgs, real_imgs)
   
        opt_G.zero_grad(); g_loss.backward(); opt_G.step()

    print("Epoch:",epoch,"D Loss:",d_loss.item(),"G Loss:",g_loss.item())

print("Training Complete!")

print("Training Finished. Generating Sample Output...")

G.eval()

# Get one real image
real_img = next(iter(loader))[0][0].to(device)

# Create damaged version
damaged_img = add_damage(real_img.cpu()).unsqueeze(0).to(device)

# Restore using GAN
with torch.no_grad():
    restored_img = G(damaged_img)

# Convert for display
def show(img):
    img = img.cpu().squeeze().permute(1,2,0).numpy()
    img = (img * 0.5) + 0.5
    return img

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("Original")
plt.imshow(show(real_img))

plt.subplot(1,3,2)
plt.title("Damaged")
plt.imshow(show(damaged_img[0]))

plt.subplot(1,3,3)
plt.title("Restored")
plt.imshow(show(restored_img[0]))

plt.show()
