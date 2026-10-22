import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import timm
import os
import glob
import random # Added so we can grab a random image every time!

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

print("=== Starting Advanced XAI Random Tester ===")

# ==========================================
# 1. SETUP & LOAD MODEL
# ==========================================
MODEL_PATH = "./saved_models/mobilevit_epoch_20.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading AI Brain...")
model = timm.create_model('mobilevitv2_100', pretrained=False, num_classes=4)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# ==========================================
# 2. SELECT A RANDOM TESTING IMAGE
# ==========================================
# 👇 CHANGE THIS WORD TO TEST DIFFERENT TUMORS!
# Options: 'glioma', 'meningioma', 'notumor', 'pituitary'
TARGET_CLASS = "notumor" 

sample_folder = f"./dataset/Testing/{TARGET_CLASS}"
image_files = glob.glob(os.path.join(sample_folder, "*.jpg"))

if len(image_files) == 0:
    print(f"Error: Could not find images in {sample_folder}")
    exit()

# Grab a random image from the folder every single time you run the script
IMAGE_PATH = random.choice(image_files)
print(f"Testing random MRI: {IMAGE_PATH}")

# Prepare images
rgb_img = Image.open(IMAGE_PATH).convert('RGB')
rgb_img = rgb_img.resize((256, 256))
rgb_img_float = np.float32(rgb_img) / 255 

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
input_tensor = transform(rgb_img).unsqueeze(0).to(device)

# Let the AI make a prediction first, so it knows what to draw the heatmap for!
with torch.no_grad():
    outputs = model(input_tensor)
    _, predicted = torch.max(outputs, 1)
    
targets = [ClassifierOutputTarget(predicted.item())]

# ==========================================
# 3. EXTRACT LAYER 1: CNN ONLY (Local Features)
# ==========================================
print("Extracting CNN Features...")
layer_cnn = [model.stages[1]] 
cam_1 = GradCAM(model=model, target_layers=layer_cnn)
heatmap_cnn = cam_1(input_tensor=input_tensor, targets=targets)[0, :]
viz_cnn = show_cam_on_image(rgb_img_float, heatmap_cnn, use_rgb=True)

# ==========================================
# 4. EXTRACT LAYER 2: TRANSFORMER ONLY (Global Context)
# ==========================================
print("Extracting Transformer Features...")
layer_vit = [model.stages[3]] 
cam_2 = GradCAM(model=model, target_layers=layer_vit)
heatmap_vit = cam_2(input_tensor=input_tensor, targets=targets)[0, :]
viz_vit = show_cam_on_image(rgb_img_float, heatmap_vit, use_rgb=True)

# ==========================================
# 5. EXTRACT LAYER 3: FUSED OUTPUT (The Final Decision)
# ==========================================
print("Extracting Fused Features...")
try:
    layer_fused = [model.stages[-1]]
except AttributeError:
    layer_fused = [model.conv_head]

cam_3 = GradCAM(model=model, target_layers=layer_fused)
heatmap_fused = cam_3(input_tensor=input_tensor, targets=targets)[0, :]
viz_fused = show_cam_on_image(rgb_img_float, heatmap_fused, use_rgb=True)

# ==========================================
# 6. PLOT THE COMPARISON
# ==========================================
plt.figure(figsize=(20, 5)) 

# 1. Original
plt.subplot(1, 4, 1)
plt.imshow(rgb_img)
plt.title('1. Original MRI Scan\n(Raw Input)')
plt.axis('off')

# 2. CNN Stage
plt.subplot(1, 4, 2)
plt.imshow(viz_cnn)
plt.title('2. Early CNN Block\n(Detects local edges)')
plt.axis('off')

# 3. Transformer Stage
plt.subplot(1, 4, 3)
plt.imshow(viz_vit)
plt.title('3. Deep Transformer Block\n(Detects global context)')
plt.axis('off')

# 4. Fused Stage
plt.subplot(1, 4, 4)
plt.imshow(viz_fused)
plt.title('4. Final Fused Output\n(Precise localization)')
plt.axis('off')

plt.tight_layout()
plt.savefig('xai_comparison_report.png', dpi=300)
print("\nSUCCESS! Saved as 'xai_comparison_report.png'.")