import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# ==========================================
# 1. DATA LOADING & AUGMENTATION (3,500 SPEED FIX)
# ==========================================
print("Loading dataset with AI Augmentation...")
DATA_DIR = "./dataset/Training"

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)

# Randomly select exactly 3,500 images to speed up local training
subset_indices = torch.randperm(len(full_dataset))[:3500]
small_dataset = torch.utils.data.Subset(full_dataset, subset_indices)

train_size = int(0.8 * len(small_dataset))
val_size = len(small_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(small_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"Using a fast subset of {len(small_dataset)} Images! | Classes: {full_dataset.classes}")

# ==========================================
# 2. MODEL SETUP (MobileViT)
# ==========================================
print("\nDownloading and Configuring MobileViT Model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = timm.create_model('mobilevitv2_100', pretrained=True, num_classes=4)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

os.makedirs("./saved_models", exist_ok=True)

# ==========================================
# 3. THE TRAINING & VALIDATION LOOP
# ==========================================
NUM_EPOCHS = 20  # Fast local run

history_train_loss = []
history_val_loss = []
history_train_acc = []
history_val_acc = []
epoch_checkpoints = []

print(f"\nStarting Training for {NUM_EPOCHS} epochs...")

for epoch in range(NUM_EPOCHS):
    # --- TRAINING PHASE ---
    model.train()
    running_loss = 0.0
    correct_predictions, total_predictions = 0, 0
    
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad() 
        outputs = model(images) 
        loss = criterion(outputs, labels) 
        
        loss.backward() 
        optimizer.step() 
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    train_acc = (correct_predictions / total_predictions) * 100
    train_loss = running_loss / len(train_loader)
    
    history_train_loss.append(train_loss)
    history_train_acc.append(train_acc)
    
    # --- VALIDATION & OVERFITTING CHECK (Every 5 Epochs) ---
    if (epoch + 1) % 5 == 0:
        epoch_checkpoints.append(epoch + 1)
        
        model.eval() 
        val_loss = 0.0
        val_correct, val_total = 0, 0
        
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                val_outputs = model(val_images)
                v_loss = criterion(val_outputs, val_labels)
                
                val_loss += v_loss.item()
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_total += val_labels.size(0)
                val_correct += (val_predicted == val_labels).sum().item()
                
        v_acc = (val_correct / val_total) * 100
        v_loss = val_loss / len(val_loader)
        
        history_val_loss.append(v_loss)
        history_val_acc.append(v_acc)
        
        print(f"\n--- EPOCH {epoch+1} CHECKPOINT ---")
        print(f"Train Acc: {train_acc:.2f}% | Val Acc: {v_acc:.2f}%")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {v_loss:.4f}")
        
        if v_loss > train_loss + 0.1:
            print("⚠️ WARNING: Validation Loss is rising. Potential Overfitting Detected!")
            
        save_path = f"./saved_models/mobilevit_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Checkpoint saved to {save_path}")
    else:
        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}] - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")

print("\nTraining Complete!")

# ==========================================
# 4. PLOT OVERFITTING GRAPH
# ==========================================
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, NUM_EPOCHS + 1), history_train_loss, label='Train Loss', color='blue')
plt.plot(epoch_checkpoints, history_val_loss, label='Val Loss', color='red', marker='o')
plt.title('Loss vs Epochs (Overfitting Check)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, NUM_EPOCHS + 1), history_train_acc, label='Train Accuracy', color='blue')
plt.plot(epoch_checkpoints, history_val_acc, label='Val Accuracy', color='green', marker='o')
plt.title('Accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.savefig('overfitting_check_graph.png')
print("Graph saved as 'overfitting_check_graph.png'.")