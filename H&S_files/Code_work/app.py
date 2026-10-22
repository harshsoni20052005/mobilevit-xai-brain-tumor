import streamlit as st
import torch
import numpy as np
from PIL import Image
import timm
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# 1. Setup the Webpage layout
st.set_page_config(page_title="Brain Tumor AI", layout="centered")
st.title("🧠 Brain Tumor Detection & Fused XAI")
st.write("Upload an MRI scan to predict the tumor class and generate a Dual-Level Fused XAI heatmap.")

# 2. Load the Model (We cache this so it doesn't reload on every single click)
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model('mobilevitv2_100', pretrained=False, num_classes=4)
    # Load YOUR trained brain
    model.load_state_dict(torch.load("./saved_models/mobilevit_epoch_20.pth", map_location=device))
    model = model.to(device)
    model.eval()
    return model, device

model, device = load_model()

# Alphabetical order based on your Kaggle dataset folders
classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# 3. Create the File Uploader Widget
uploaded_file = st.file_uploader("Choose an MRI Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Create two columns for a side-by-side comparison
    col1, col2 = st.columns(2)
    
    # Prepare the Human Image
    image = Image.open(uploaded_file).convert('RGB')
    image = image.resize((256, 256))
    rgb_img_float = np.float32(image) / 255 

    with col1:
        st.subheader("Original MRI")
        st.image(image, use_column_width=True)

    # 4. The "Analyze" Button
    if st.button("Run AI Diagnosis & XAI", type="primary"):
        with st.spinner("Analyzing brain structures and generating Heatmap..."):
            
            # Prepare the AI Tensor
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            input_tensor = transform(image).unsqueeze(0).to(device)

            # A. Make the Prediction
            with torch.no_grad():
                outputs = model(input_tensor)
                _, predicted = torch.max(outputs.data, 1)
                predicted_class = classes[predicted.item()]
            
            # B. Generate the Fused XAI Heatmap
            try:
                target_layers = [model.stages[-1]]
            except AttributeError:
                target_layers = [model.conv_head]
            
            cam = GradCAM(model=model, target_layers=target_layers)
            targets = [ClassifierOutputTarget(predicted.item())]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
            visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)

            with col2:
                st.subheader(f"Prediction: {predicted_class}")
                st.image(visualization, caption="MobileViT Fused Heatmap", use_column_width=True)
            
            st.success(f"Diagnosis Complete! The model detected: **{predicted_class}**") 