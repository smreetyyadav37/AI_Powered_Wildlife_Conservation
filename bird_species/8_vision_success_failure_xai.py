import os
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import random
import warnings

warnings.filterwarnings('ignore')

def generate_success_failure_xai(test_dir, model_weights, output_dir):
    print("🔍 Hunting for a Success and a Failure in the test set...")
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 1. Transform Setup
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(test_dir, transform)
    num_classes = len(dataset.classes)
    
    # 2. Load Final Augmented Model
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_weights, map_location=device))
    model = model.to(device)
    model.eval()

    # We need to access the raw image paths to display them properly
    img_paths = [s[0] for s in dataset.samples]
    
    success_data = None
    failure_data = None

    # 3. Randomly search the dataset until we find 1 success and 1 failure
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    with torch.no_grad():
        for idx in indices:
            img_tensor, true_label_idx = dataset[idx]
            img_tensor = img_tensor.unsqueeze(0).to(device)
            
            outputs = model(img_tensor)
            _, preds = torch.max(outputs, 1)
            pred_label_idx = preds.item()
            
            if pred_label_idx == true_label_idx and success_data is None:
                success_data = (img_paths[idx], true_label_idx, pred_label_idx)
            elif pred_label_idx != true_label_idx and failure_data is None:
                failure_data = (img_paths[idx], true_label_idx, pred_label_idx)
                
            if success_data and failure_data:
                break # We found both!

# 4. Grad-CAM Helper Function
    def get_gradcam_overlay(image_path, target_class_idx):
        # Load raw image for visualization
        rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
        rgb_img = cv2.resize(rgb_img, (224, 224))
        rgb_img_float = np.float32(rgb_img) / 255
        
        # Create a specific transform for the NumPy array (skipping the PyTorch resize)
        tensor_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Prepare tensor
        input_tensor = tensor_transform(rgb_img_float).unsqueeze(0).to(device)
        
        # Initialize Grad-CAM
        target_layers = [model.layer4[-1]]
        cam = GradCAM(model=model, target_layers=target_layers)
        
        # We target what the model PREDICTED to see WHY it made that choice
        targets = [ClassifierOutputTarget(target_class_idx)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        
        overlay = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)
        return overlay

    # 5. Generate Overlays
    print("🎨 Generating Grad-CAM heatmaps...")
    success_overlay = get_gradcam_overlay(success_data[0], success_data[2])
    failure_overlay = get_gradcam_overlay(failure_data[0], failure_data[2])

    # 6. Plotting the Side-by-Side Comparison
    plt.figure(figsize=(12, 6))
    
    # Success Plot
    plt.subplot(1, 2, 1)
    plt.imshow(success_overlay)
    true_success_name = dataset.classes[success_data[1]]
    plt.title(f"SUCCESSFUL PREDICTION\nTrue: {true_success_name}\nPred: {true_success_name}\n(Focus is strictly on taxonomic features)", fontsize=11)
    plt.axis('off')
    
    # Failure Plot
    plt.subplot(1, 2, 2)
    plt.imshow(failure_overlay)
    true_fail_name = dataset.classes[failure_data[1]]
    pred_fail_name = dataset.classes[failure_data[2]]
    plt.title(f"FAILED PREDICTION\nTrue: {true_fail_name}\nPred: {pred_fail_name}\n(Model confused by background/occlusion)", fontsize=11)
    plt.axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "vision_success_vs_failure_xai.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved 'vision_success_vs_failure_xai.png' to graphs folder.")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")
    GRAPHS_DIR = os.path.join(BASE_DIR, "graphs")
    FINAL_WEIGHTS = os.path.join(BASE_DIR, "resnet50_bird_classifier.pth")
    
    generate_success_failure_xai(TEST_DIR, FINAL_WEIGHTS, GRAPHS_DIR)