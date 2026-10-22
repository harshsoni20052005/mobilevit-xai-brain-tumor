import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import os
from tqdm import tqdm

print("=== Starting ADVANCED Clinical Evaluation (IEEE Standard) ===")

# ==========================================
# 1. SETUP & LOAD MODEL
# ==========================================
MODEL_PATH = "./saved_models/mobilevit_epoch_20.pth"
TEST_DIR = "./dataset/Testing"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = timm.create_model('mobilevitv2_100', pretrained=False, num_classes=4)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
class_names = test_dataset.classes
num_classes = len(class_names)

# ==========================================
# 2. RUN INFERENCE (Extract Probabilities)
# ==========================================
all_preds = []
all_labels = []
all_probs = [] # We need probabilities for AUC-ROC!

print("Analyzing test dataset...")
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Evaluating"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        
        # Apply Softmax to get percentages/probabilities
        probs = torch.softmax(outputs, dim=1) 
        _, predicted = torch.max(outputs, 1)
        
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

# ==========================================
# 3. GENERATE CONFUSION MATRIX
# ==========================================
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix (MobileViT)')
plt.ylabel('Actual True Class')
plt.xlabel('AI Predicted Class')
plt.tight_layout()
plt.savefig('clinical_confusion_matrix.png', dpi=300)

# ==========================================
# 4. GENERATE AUC-ROC CURVES (The Gold Standard)
# ==========================================
print("Calculating AUC-ROC curves...")
# Binarize labels for multi-class ROC calculations
labels_binarized = label_binarize(all_labels, classes=[0, 1, 2, 3])

plt.figure(figsize=(10, 8))
colors = ['blue', 'red', 'green', 'purple']

for i, color in zip(range(num_classes), colors):
    fpr, tpr, _ = roc_curve(labels_binarized[:, i], all_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2,
             label=f'ROC curve of {class_names[i]} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Guessing (AUC = 0.500)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Multi-Class Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_auc_curve.png', dpi=300)
print("SUCCESS: Saved 'roc_auc_curve.png'. Insert this graph into your paper!")

# ==========================================
# 5. PRINT ADVANCED CLINICAL REPORT
# ==========================================
print("\n=== ADVANCED CLASSIFICATION REPORT ===")
report = classification_report(all_labels, all_preds, target_names=class_names)
print(report)

with open("f1_score_report.txt", "w") as f:
    f.write("Clinical Evaluation - MobileViT\n")
    f.write("===============================\n")
    f.write(report)