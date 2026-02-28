import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import albumentations as A
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

def run_data_audit_and_augmentation(train_dir, graphs_dir):
    print("🚀 Starting Phase 1: Data Auditing & Targeted Augmentation...")
    
    # 1. Create Graphs directory if it doesn't exist
    os.makedirs(graphs_dir, exist_ok=True)
    
    # 2. Audit the current dataset (Counting only original images, ignoring past augmentations)
    print("📊 Auditing class distribution...")
    species_list = os.listdir(train_dir)
    original_counts = {}
    
    for species in tqdm(species_list, desc="Scanning Folders"):
        species_path = os.path.join(train_dir, species)
        if os.path.isdir(species_path):
            # Count only original images to prevent infinite loop of augmentations
            original_imgs = [img for img in os.listdir(species_path) if not img.startswith("aug_")]
            original_counts[species] = len(original_imgs)
            
    df_counts = pd.DataFrame(list(original_counts.items()), columns=['Species', 'Original_Count'])
    df_counts = df_counts.sort_values(by='Original_Count').reset_index(drop=True)
    
    # 3. Define "Rare" threshold (25th Percentile)
    threshold = np.percentile(df_counts['Original_Count'], 25)
    rare_species = df_counts[df_counts['Original_Count'] <= threshold]['Species'].tolist()
    
    print(f"\n📈 Audit Complete:")
    print(f"Total Species: {len(species_list)}")
    print(f"Rarity Threshold (25th Percentile): {threshold:.0f} images")
    print(f"Number of 'Rare' Species Identified: {len(rare_species)}")
    
    # 4. Plot "Before" Distribution
    plt.figure(figsize=(15, 5))
    plt.plot(df_counts['Species'], df_counts['Original_Count'], color='red')
    plt.axhline(y=threshold, color='blue', linestyle='--', label=f'Rare Threshold ({threshold:.0f})')
    plt.xticks([]) 
    plt.title("Original Dataset Distribution (Imbalanced)", fontsize=14)
    plt.ylabel("Number of Images")
    plt.xlabel("Bird Species (Sorted by Count)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "distribution_before_aug.png"), dpi=300)
    print(f"✅ Saved 'distribution_before_aug.png' to graphs folder.")
    

    # 5. Define Augmentation Pipeline
    # These specific augmentations simulate real-world camera noise, lighting changes, and bird angles
    augmentation_pipeline = A.Compose([
        A.HorizontalFlip(p=0.7),
        A.Rotate(limit=25, p=0.7),
        A.RandomBrightnessContrast(p=0.5),
        A.Blur(blur_limit=3, p=0.2)
    ])
    
    # 6. Apply Targeted Augmentations
    print("\n🧬 Synthesizing new data for Rare classes...")
    total_augmented = 0
    
    for species in tqdm(rare_species, desc="Augmenting Rare Species"):
        species_path = os.path.join(train_dir, species)
        original_imgs = [img for img in os.listdir(species_path) if not img.startswith("aug_")]
        
        for img_name in original_imgs:
            img_path = os.path.join(species_path, img_name)
            
            try:
                # Read with OpenCV, convert BGR to RGB (standard for Neural Networks)
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Apply augmentation
                augmented = augmentation_pipeline(image=image)
                aug_image = augmented["image"]
                
                # Convert back to BGR for OpenCV saving
                aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                
                save_path = os.path.join(species_path, f"aug_{img_name}")
                cv2.imwrite(save_path, aug_image_bgr)
                total_augmented += 1
                
            except Exception as e:
                # Silently pass corrupted/unreadable images
                continue

    print(f"\n✅ Augmentation Complete! Synthesized {total_augmented} new images.")
    
    # 7. Plot "After" Distribution
    final_counts = {}
    for species in species_list:
        species_path = os.path.join(train_dir, species)
        if os.path.isdir(species_path):
            final_counts[species] = len(os.listdir(species_path))
            
    df_final = pd.DataFrame(list(final_counts.items()), columns=['Species', 'Final_Count'])
    df_final = df_final.set_index('Species').reindex(df_counts['Species']).reset_index()

    plt.figure(figsize=(15, 5))
    plt.plot(df_final['Species'], df_final['Final_Count'], color='teal', label="New Count (After Augmentation)")
    plt.plot(df_counts['Species'], df_counts['Original_Count'], color='red', alpha=0.3, label="Original Count")
    plt.axhline(y=threshold, color='blue', linestyle='--', label=f'Original Threshold ({threshold:.0f})')
    plt.xticks([])
    plt.title("Balanced Dataset Distribution (After Targeted Augmentation)", fontsize=14)
    plt.ylabel("Number of Images")
    plt.xlabel("Bird Species (Sorted by Original Count)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "distribution_after_aug.png"), dpi=300)
    print(f"✅ Saved 'distribution_after_aug.png' to graphs folder.")

if __name__ == "__main__":
    import os
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    TRAIN_DIRECTORY = os.path.join(BASE_DIR, "dataset", "train")
    GRAPHS_DIRECTORY = os.path.join(BASE_DIR, "graphs")
    
    os.makedirs(GRAPHS_DIRECTORY, exist_ok=True)
    
    print(f"📁 Working directory automatically set to: {BASE_DIR}")
    run_data_audit_and_augmentation(TRAIN_DIRECTORY, GRAPHS_DIRECTORY)