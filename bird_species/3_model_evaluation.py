import os
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

def run_final_evaluation(test_dir, model_weights_path, graphs_dir, results_dir):
    print("🔬 Starting Phase 3: Final Model Evaluation on Unseen Test Data...")
    
    # 1. Setup Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ Evaluating on: {device}")
    
    # 2. Prepare Test Data
    # Strictly NO data augmentation here, only resize and normalize.
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_dataset = datasets.ImageFolder(test_dir, test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    class_names = test_dataset.classes
    num_classes = len(class_names)
    print(f"📚 Test Dataset Loaded: {len(test_dataset)} images across {num_classes} species.")
    
    # 3. Load Model Architecture & Weights
    print(f"🏗️ Rebuilding ResNet-50 and loading weights from {model_weights_path}...")
    model = models.resnet50(weights=None) 
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    try:
        model.load_state_dict(torch.load(model_weights_path, map_location=device))
    except FileNotFoundError:
        print(f"\nERROR: Could not find '{model_weights_path}'.")
        return
        
    model = model.to(device)
    model.eval()
    
    # 4. Run Inference
    print("Running predictions on test set... (This might take a minute)")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # 5. Calculate Metrics
    print("\n📊 Calculating Research Metrics...")
    
    # Macro F1 is critical for highly multi-class problems
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    overall_accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    print(f"🏆 Overall Test Accuracy: {overall_accuracy * 100:.2f}%")
    print(f"🏆 Macro F1-Score:        {macro_f1:.4f}")
    
    # 6. Extract Top-5 Misclassifications
    cm = confusion_matrix(all_labels, all_preds)
    
    misclassifications = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                misclassifications.append({
                    'True_Class': class_names[i],
                    'Predicted_Class': class_names[j],
                    'Count': cm[i, j]
                })
                
    # Sort by the most frequent errors
    df_errors = pd.DataFrame(misclassifications)
    if not df_errors.empty:
        df_errors = df_errors.sort_values(by='Count', ascending=False).head(5)
        
        print("\n⚠️ Top 5 Species Misclassifications:")
        print(df_errors.to_string(index=False))
        
        # Plot the misclassifications as a bar chart
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_errors, x='Count', y='True_Class', hue='Predicted_Class', dodge=False)
        plt.title('Top 5 AI Misclassifications (True Class vs Predicted)', fontsize=14)
        plt.xlabel('Number of Times Confused')
        plt.ylabel('True Bird Species')
        plt.tight_layout()
        plt.savefig(os.path.join(graphs_dir, "top_misclassifications.png"), dpi=300)
        print("✅ Saved 'top_misclassifications.png' to graphs folder.")
        
    else:
        print("\n🌟 Flawless! No misclassifications found in the test set.")

    # 7. Export Formal Report
    export_file = os.path.join(results_dir, "computer_vision_metrics.txt")
    with open(export_file, "w") as f:
        f.write("====================================================\n")
        f.write("   BIRD SPECIES CLASSIFICATION: TEST SET METRICS\n")
        f.write("====================================================\n\n")
        f.write(f"Total Test Images: {len(test_dataset)}\n")
        f.write(f"Overall Accuracy:  {overall_accuracy * 100:.2f}%\n")
        f.write(f"Macro F1-Score:    {macro_f1:.4f}\n\n")
        
        if not df_errors.empty:
            f.write("TOP 5 MISCLASSIFICATIONS:\n")
            f.write("----------------------------------------------------\n")
            for index, row in df_errors.iterrows():
                f.write(f"The model confused '{row['True_Class']}' with '{row['Predicted_Class']}' ({row['Count']} times)\n")

    print(f"\n✅ Success! Research metrics exported to '{export_file}'")

if __name__ == "__main__":
    import os
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TEST_DIRECTORY = os.path.join(BASE_DIR, "dataset", "test")
    MODEL_WEIGHTS = os.path.join(BASE_DIR, "resnet50_bird_classifier.pth")
    GRAPHS_DIRECTORY = os.path.join(BASE_DIR, "graphs")
    RESULTS_DIRECTORY = os.path.join(BASE_DIR, "results")
    
    os.makedirs(GRAPHS_DIRECTORY, exist_ok=True)
    os.makedirs(RESULTS_DIRECTORY, exist_ok=True)
    
    print(f"📁 Working directory automatically set to: {BASE_DIR}")
    
    run_final_evaluation(TEST_DIRECTORY, MODEL_WEIGHTS, GRAPHS_DIRECTORY, RESULTS_DIRECTORY)