import os
import random
import matplotlib.pyplot as plt
import cv2

IMG_DIR = os.path.join('Data', 'train', 'images')
MASK_DIR = os.path.join('Data', 'train', 'masks')

# Get list of image files
image_files = [f for f in os.listdir(IMG_DIR) if f.endswith('.png')]

# Pick 5 random images
random_images = random.sample(image_files, 5)

for img_file in random_images:
    base = img_file.replace('.png', '')
    mask_file = os.path.join(MASK_DIR, base + '_mask.png')
    if not os.path.exists(mask_file):
        print(f"Mask not found for {img_file}, skipping.")
        continue
    img = cv2.imread(os.path.join(IMG_DIR, img_file))
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Image')
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')
    plt.show() 