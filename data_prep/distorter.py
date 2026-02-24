import cv2
import numpy as np
import random
from pathlib import Path

def apply_noise(image):
    """Adds a visible but controlled amount of Gaussian noise."""
    row, col, ch = image.shape
    mean = 0
    # Controlled noise variance for visibility
    var = 0.02 
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image + gauss * 255
    return np.clip(noisy, 0, 255).astype(np.uint8)

def apply_scratches(image):
    """Draws thin scratch lines that are visible but don't obscure the scene."""
    img = image.copy()
    num_scratches = random.randint(8, 15)
    for _ in range(num_scratches):
        x1, y1 = random.randint(0, img.shape[1]), random.randint(0, img.shape[0])
        # Short scratch length
        length = random.randint(20, 80)
        angle = random.uniform(0, 2 * np.pi)
        x2 = int(x1 + length * np.cos(angle))
        y2 = int(y1 + length * np.sin(angle))
        
        color = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255)) # Light colored scratches
        thickness = random.randint(1, 2)
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    return img

def apply_faded_patches(image):
    """Adds faded or 'missing' patches that allow context to be reconstructed."""
    img = image.copy()
    num_patches = random.randint(2, 4)
    for _ in range(num_patches):
        w = random.randint(30, 60)
        h = random.randint(30, 60)
        x = random.randint(0, img.shape[1] - w)
        y = random.randint(0, img.shape[0] - h)
        
        if random.random() > 0.5:
            # Semi-transparent dark patch (better for supervised learning than pure black)
            sub = img[y:y+h, x:x+w].astype(np.float32)
            alpha = 0.7 # 70% dark
            img[y:y+h, x:x+w] = (sub * (1 - alpha)).astype(np.uint8)
        else:
            # Significant blur patch
            roi = img[y:y+h, x:x+w]
            img[y:y+h, x:x+w] = cv2.GaussianBlur(roi, (31, 31), 0)
            
    return img

def distort_image(image_path):
    """Apply a balanced distortion suite for the RestoraAI demo."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None: return None
    
    try:
        # Sequence matters for visual balance
        image = apply_noise(image)
        image = apply_scratches(image)
        image = apply_faded_patches(image)
        return image
    except Exception as e:
        print(f"Distortion fail: {e}")
        return None
