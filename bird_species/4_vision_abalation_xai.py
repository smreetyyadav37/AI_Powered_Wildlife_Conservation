import os
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import warnings

warnings.filterwarnings('ignore')

def generate_comparative_xai(image_path, baseline_weights, final_weights, num_classes, output_dir):
    print("🔍 Generating Comparative Grad-CAM Ablation Study...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 1. Load the Image
    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img_float = np.float32(rgb_img) / 255
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(rgb_img_float).unsqueeze(0).to(device)

    # Helper function to load model and get CAM
    def get_cam_visualization(weights_path):
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model = model.to(device)
        model.eval()
        
        target_layers = [model.layer4[-1]]
        cam = GradCAM(model=model, target_layers=target_layers)
        
        # We look at the top predicted class for the heatmap
        outputs = model(input_tensor)
        _, preds = torch.max(outputs, 1)
        target_class = preds.item()
        
        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(target_class)])
        visualization = show_cam_on_image(rgb_img_float, grayscale_cam[0, :], use_rgb=True)
        return visualization, target_class

    # 2. Get Visualizations
    vis_baseline, class_base = get_cam_visualization(baseline_weights)
    vis_final, class_final = get_cam_visualization(final_weights)

    # 3. Plot the Ablation Comparison
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(rgb_img)
    plt.title("Original Image (Rare Species)")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(vis_baseline)
    plt.title("Baseline Model (Imbalanced Data)\nNotice focus on background")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(vis_final)
    plt.title("Final Model (Targeted Augmentation)\nFocus is strictly on the bird")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "xai_ablation_comparison9.png"), dpi=300)
    print("✅ Saved 'xai_ablation_comparison9.png' to graphs folder.")

if __name__ == "__main__":
    import os
    import glob
    import random
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    GRAPHS_DIRECTORY = os.path.join(BASE_DIR, "graphs")
    os.makedirs(GRAPHS_DIRECTORY, exist_ok=True)
    
    BASELINE_WEIGHTS = os.path.join(BASE_DIR, "resnet50_baseline_imbalanced.pth")
    FINAL_WEIGHTS = os.path.join(BASE_DIR, "resnet50_bird_classifier.pth")
    
    # --- DYNAMIC IMAGE SELECTION ---
    test_dir = os.path.join(BASE_DIR, "dataset", "test")
    
    print(f"📂 Searching for images in: {test_dir}")
    all_test_images = glob.glob(os.path.join(test_dir, "**/*.jpg"), recursive=True)
    
    if not all_test_images:
        print("❌ ERROR: Could not find any .jpg images in your test folder.")
    else:
        # Pick a random image from the test set
        TEST_IMAGE = random.choice(all_test_images)
        print(f"🎲 Randomly selected test image: {TEST_IMAGE}")
        
        generate_comparative_xai(TEST_IMAGE, BASELINE_WEIGHTS, FINAL_WEIGHTS, 525, GRAPHS_DIRECTORY)