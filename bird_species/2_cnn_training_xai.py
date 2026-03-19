import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Import Grad-CAM for Explainable AI
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import warnings
warnings.filterwarnings('ignore')

def train_and_evaluate_model(data_dir, graphs_dir, model_save_path, num_epochs=4):
    print("🧠 Starting Phase 2: PyTorch Transfer Learning & Grad-CAM XAI (Baseline Run)...")
    
    # 1. Hardware Configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ Computation Device: {device}")
    
    # 2. Data Streaming Setup (High-throughput transformations)
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
        ]),
        'valid': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Load data directly from your pre-split folders
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'valid']
    }
    
    # DataLoader manages the batching and parallel processing
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4)
        for x in ['train', 'valid']
    }
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)
    
    print(f"📚 Dataset Loaded: {dataset_sizes['train']} Train | {dataset_sizes['valid']} Valid")
    print(f"🏷️ Total Bird Species: {num_classes}")

    # 3. Model Architecture (Transfer Learning)
    print("\n🏗️ Initializing ResNet-50 Architecture...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # Freeze early layers to retain basic edge/shape detection
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze the final layer group (layer4) and the fully connected (fc) layer
    for param in model.layer4.parameters():
        param.requires_grad = True
        
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    # 4. Optimizer and Loss Function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    # 5. Training Loop
    print(f"\n🚀 Beginning Training for {num_epochs} Epochs...")
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    history = {'train_loss': [], 'valid_loss': [], 'train_acc': [], 'valid_acc': []}

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 15)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f"✅ Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"🏆 Best Validation Accuracy: {best_acc:4f}")

    # Load best weights and save with custom path
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), model_save_path)
    print(f"💾 Best model weights saved explicitly to: {model_save_path}")

    # 6. Plot Training Curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['valid_loss'], label='Valid Loss')
    plt.title('Cross Entropy Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['valid_acc'], label='Valid Acc')
    plt.title('Classification Accuracy')
    plt.legend()
    
    plt.savefig(os.path.join(graphs_dir, "training_curves_baseline.png"), dpi=300)
    print("✅ Saved 'training_curves_baseline.png' to graphs folder.")
    
    return model, dataloaders, class_names, device

def generate_gradcam_xai(model, dataloaders, class_names, device, graphs_dir):
    print("\n🔍 Generating Grad-CAM Explainability Heatmap...")
    model.eval()
    
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    
    inputs, labels = next(iter(dataloaders['valid']))
    input_tensor = inputs[0:1].to(device) 
    true_label_idx = labels[0].item()
    true_class_name = class_names[true_label_idx]
    
    targets = [ClassifierOutputTarget(true_label_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    img_np = input_tensor[0].cpu().numpy().transpose(1, 2, 0)
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)
    
    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_np)
    plt.title(f"Original: {true_class_name}")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(visualization)
    plt.title("Grad-CAM XAI Focus Area")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "gradcam_sample_baseline.png"), dpi=300)
    print("✅ Saved 'gradcam_sample_baseline.png' to graphs folder.")

if __name__ == "__main__":
    import os
    
    # 1. Dynamically hunt for the folder that contains 'train' and 'valid'
    DATA_DIRECTORY = "/content/dataset" 
    for root, dirs, files in os.walk('/content/dataset'):
        if 'train' in dirs and 'valid' in dirs:
            DATA_DIRECTORY = root
            print(f"✅ Found data folders at: {DATA_DIRECTORY}")
            break
            
    GRAPHS_DIRECTORY = "/content/graphs"
    os.makedirs(GRAPHS_DIRECTORY, exist_ok=True)
    
    # Define exact path for baseline model
    BASELINE_SAVE_PATH = "/content/resnet50_baseline_imbalance.pth"
    
    # 2. Run the pipeline (Restricted to 4 Epochs)
    trained_model, loaders, classes, comp_device = train_and_evaluate_model(
        DATA_DIRECTORY, GRAPHS_DIRECTORY, model_save_path=BASELINE_SAVE_PATH, num_epochs=4
    )
    generate_gradcam_xai(trained_model, loaders, classes, comp_device, GRAPHS_DIRECTORY)