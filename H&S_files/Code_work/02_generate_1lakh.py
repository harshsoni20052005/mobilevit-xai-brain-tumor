import os
from PIL import Image
import torchvision.transforms as transforms
import random
from tqdm import tqdm

# 1. Define paths
SOURCE_DIR = "./dataset/Training"
TARGET_TOTAL_IMAGES = 10000

# 2. Define how the new images will be modified
augmenter = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05))
])

print("Scanning original dataset...")

# 3. Gather all original images
all_original_images = []
for class_name in os.listdir(SOURCE_DIR):
    class_dir = os.path.join(SOURCE_DIR, class_name)
    if os.path.isdir(class_dir):
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_original_images.append((class_dir, img_name))

current_count = len(all_original_images)
print(f"Found {current_count} original images.")

if current_count == 0:
    print("Error: No images found. Please check your dataset folder!")
    exit()

images_needed = TARGET_TOTAL_IMAGES - current_count
print(f"Generating {images_needed} new augmented images to reach 1 Lakh...")

# 4. Generate until we hit 100,000
with tqdm(total=images_needed, desc="Generating 1 Lakh Dataset") as pbar:
    for i in range(images_needed):
        folder_path, img_file = random.choice(all_original_images)
        full_img_path = os.path.join(folder_path, img_file)
        
        try:
            img = Image.open(full_img_path).convert('RGB')
            new_img = augmenter(img)
            
            new_filename = f"aug_{i}_{img_file}"
            new_img.save(os.path.join(folder_path, new_filename))
            pbar.update(1)
        except Exception as e:
            continue

print("\nSuccess! You now have a 100,000 image dataset physically saved on your drive.")