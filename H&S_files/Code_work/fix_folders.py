import os
import shutil

# This targets your training folder
dataset_path = "./dataset/Training"

print("Scanning for nested Kaggle folders...")

# Look inside glioma, meningioma, etc.
for class_folder in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_folder)
    
    if os.path.isdir(class_path):
        for sub_item in os.listdir(class_path):
            sub_item_path = os.path.join(class_path, sub_item)
            
            # If it finds a folder INSIDE the folder (the error you got)
            if os.path.isdir(sub_item_path):
                print(f"Fixing nested folder: {sub_item_path}...")
                
                # Move all images up one level
                for image in os.listdir(sub_item_path):
                    shutil.move(os.path.join(sub_item_path, image), class_path)
                
                # Delete the annoying empty folder
                os.rmdir(sub_item_path)

print("\nAll folders are perfectly structured!")
print("You can now run: python 01_train_model.py")