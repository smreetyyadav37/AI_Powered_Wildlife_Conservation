import os
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from statsmodels.stats.contingency_tables import mcnemar
import numpy as np
import warnings

warnings.filterwarnings('ignore')

def run_mcnemars_test(test_dir, baseline_weights, final_weights):
    print("🔬 Running McNemar's Test for Statistical Significance...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_dataset = datasets.ImageFolder(test_dir, test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    num_classes = len(test_dataset.classes)
    
    def get_model_predictions(weights_path):
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model = model.to(device)
        model.eval()
        
        preds_list = []
        labels_list = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                preds_list.extend(preds.cpu().numpy())
                labels_list.extend(labels.cpu().numpy())
        return np.array(preds_list), np.array(labels_list)

    print("Running inference with Baseline Model...")
    baseline_preds, true_labels = get_model_predictions(baseline_weights)
    
    print("Running inference with Final Model...")
    final_preds, _ = get_model_predictions(final_weights)

    # Calculate Correct (1) vs Incorrect (0) for both models
    base_correct = (baseline_preds == true_labels).astype(int)
    final_correct = (final_preds == true_labels).astype(int)

    # Build Contingency Table
    # [[Both Correct, Base Wrong / Final Right],
    #  [Base Right / Final Wrong, Both Wrong]]
    table = [[0, 0], [0, 0]]
    for b, f in zip(base_correct, final_correct):
        if b == 1 and f == 1: table[0][0] += 1
        elif b == 0 and f == 1: table[0][1] += 1
        elif b == 1 and f == 0: table[1][0] += 1
        elif b == 0 and f == 0: table[1][1] += 1

    # Perform McNemar's Test
    result = mcnemar(table, exact=False, correction=True)

    print("\n====================================================")
    print(" 📊 MCNEMAR'S TEST RESULTS (VISION CLASSIFICATION)")
    print("====================================================")
    print("Contingency Table:")
    print(f"Both Models Correct:          {table[0][0]}")
    print(f"Baseline Wrong, Final Right:  {table[0][1]}  <-- (Improvement)")
    print(f"Baseline Right, Final Wrong:  {table[1][0]}  <-- (Degradation)")
    print(f"Both Models Wrong:            {table[1][1]}")
    print("----------------------------------------------------")
    print(f"Chi-Squared Statistic: {result.statistic:.4f}")
    print(f"P-Value:               {result.pvalue:.8f}")
    
    if result.pvalue < 0.05:
        print("\n✅ CONCLUSION: The p-value is < 0.05.")
        print("The targeted data augmentation produced a STATISTICALLY SIGNIFICANT improvement.")
    else:
        print("\n⚠️ CONCLUSION: The p-value is >= 0.05.")
        print("The difference in classifiers is not statistically significant.")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")
    BASELINE_WEIGHTS = os.path.join(BASE_DIR, "resnet50_baseline_imbalanced.pth")
    FINAL_WEIGHTS = os.path.join(BASE_DIR, "resnet50_bird_classifier.pth")
    
    run_mcnemars_test(TEST_DIR, BASELINE_WEIGHTS, FINAL_WEIGHTS)