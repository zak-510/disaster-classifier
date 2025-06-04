import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Set your training image and mask directories
IMG_DIR = os.path.join('Data', 'train', 'images')
MASK_DIR = os.path.join('Data', 'train', 'labels')
OUT_DIR = 'training_data_visualizations'
os.makedirs(OUT_DIR, exist_ok=True)

# List a few image files
img_files = [f for f in os.listdir(IMG_DIR) if f.endswith('.png')][:5]

for img_file in img_files:
    img_path = os.path.join(IMG_DIR, img_file)
    mask_path = os.path.join(MASK_DIR, img_file.replace('.png', '.png').replace('images', 'labels'))
    # Try both .png and .json for mask
    if not os.path.exists(mask_path):
        mask_path = mask_path.replace('.png', '.json')
    img = Image.open(img_path).convert('RGB')
    # Try to load mask as image, fallback to blank if not found
    try:
        mask = Image.open(mask_path).convert('L')
    except Exception:
        mask = Image.new('L', img.size)
    # Plot side by side
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(img)
    axs[0].set_title('Image')
    axs[0].axis('off')
    axs[1].imshow(mask, cmap='gray')
    axs[1].set_title('Mask')
    axs[1].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'{img_file.replace(".png", "_viz.png")}'))
    plt.close()
print(f"Saved visualizations to {OUT_DIR}/") 