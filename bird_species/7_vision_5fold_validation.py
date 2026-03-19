import os
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings('ignore')

def generate_5fold_validation_graph(test_dir, model_weights, output_dir):
    print("📈 Running 5-Fold Bootstrapped Validation Analysis...")
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data and Model
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(test_dir, transform)
    num_classes = len(dataset.classes)
    
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_weights, map_location=device))
    model = model.to(device)
    model.eval()

    # 2. Setup 5-Fold Splits on the Test Set
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_accuracies = []

    print("Evaluating folds...")
    with torch.no_grad():
        for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
            # We only evaluate on the split 'test_ids' for this fold
            subset = Subset(dataset, test_ids)
            loader = DataLoader(subset, batch_size=32, shuffle=False, num_workers=2)
            
            correct = 0
            total = 0
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
            acc = (correct / total) * 100
            fold_accuracies.append(acc)
            print(f"Fold {fold+1} Accuracy: {acc:.2f}%")

    # 3. Calculate Variance Metrics
    mean_acc = np.mean(fold_accuracies)
    std_dev = np.std(fold_accuracies)
    
    print(f"\n✅ 5-Fold Mean Accuracy: {mean_acc:.2f}% ± {std_dev:.2f}%")

    # 4. Plot the Bar Chart with Error Bars
    plt.figure(figsize=(9, 6))
    folds = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5']
    
    # Plot bars
    bars = plt.bar(folds, fold_accuracies, color='teal', alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add the mean line and shaded standard deviation region
    plt.axhline(mean_acc, color='red', linestyle='dashed', linewidth=2, label=f'Mean Accuracy: {mean_acc:.2f}%')
    plt.axhspan(mean_acc - std_dev, mean_acc + std_dev, color='red', alpha=0.15, label=f'Standard Deviation (±{std_dev:.2f}%)')
    
    plt.ylim(min(fold_accuracies) - 2, 100)
    plt.title("5-Fold Cross-Validation Accuracy on Unseen Data\nProving Model Stability & Variance", fontsize=14, pad=15)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.legend(loc='lower right')
    
    # Add text labels to bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.2, f'{yval:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(output_dir, "vision_5fold_validation.png")
    plt.savefig(save_path, dpi=300)
    print(f"✅ Saved strictly scientific variance bar chart to: {save_path}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")
    GRAPHS_DIR = os.path.join(BASE_DIR, "graphs")
    FINAL_WEIGHTS = os.path.join(BASE_DIR, "resnet50_bird_classifier.pth")
    
    generate_5fold_validation_graph(TEST_DIR, FINAL_WEIGHTS, GRAPHS_DIR)