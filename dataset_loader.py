import os
import shutil
import subprocess
from pathlib import Path

# --- Configuration ---
TARGET_DIR = Path("dataset_organized")
BASE_DIR_ARNAB = Path("dataset_raw_arnab")
BASE_DIR_ROCKVISION = Path("dataset_raw_rockvision")

SOURCES = {
    "arnab": "https://github.com/Arnab14999/Rock-image-classifier.git",
    "rockvision": "https://github.com/nachomonereo/RockVision_Lite.git"
}

# Mapping: Source Class -> Target Class
CLASS_MAPPING_ARNAB = {
    "Igneous": ["Basalt", "Granite"],
    "Metamorphic": ["Marble", "Quartzite"],
    "Sedimentary": ["Coal", "Limestone", "Sandstone"]
}

CLASS_MAPPING_ROCKVISION = {
    0: "Igneous",
    1: "Metamorphic",
    2: "Sedimentary"
}

def force_remove(path):
    if path.exists():
        print(f"[Info] Removing directory: {path}")
        try:
            shutil.rmtree(path)
        except Exception:
            os.system(f'rmdir /S /Q "{path}"')

def download_datasets():
    for name, url in SOURCES.items():
        target_path = BASE_DIR_ARNAB if name == "arnab" else BASE_DIR_ROCKVISION
        
        if not target_path.exists():
            print(f"[Info] Cloning {name} dataset...")
            try:
                subprocess.run(["git", "clone", url, str(target_path)], check=True)
            except subprocess.CalledProcessError as e:
                print(f"[Error] Failed to clone {name}: {e}")
        else:
            print(f"[Info] {name} dataset already exists.")

def is_valid_image(file_path):
    return file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']

def process_arnab(target_dir):
    print("Processing Arnab14999 dataset...")
    source_arnab = BASE_DIR_ARNAB / "Data"
    
    if not source_arnab.exists():
        return

    for category, rock_types in CLASS_MAPPING_ARNAB.items():
        category_dir = target_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        
        for rock_type in rock_types:
            found_dir = None
            for d in source_arnab.iterdir():
                if d.is_dir() and d.name.lower() == rock_type.lower():
                    found_dir = d
                    break
            
            if found_dir:
                count = 0
                for img_file in found_dir.glob("*"):
                    if is_valid_image(img_file):
                        shutil.copy(img_file, category_dir / f"arnab_{rock_type}_{img_file.name}")
                        count += 1
                print(f"  Merged {count} images from {rock_type} -> {category}")

def process_rockvision(target_dir):
    print("Processing RockVision Lite dataset...")
    base_rv = BASE_DIR_ROCKVISION / "datasets" / "rock_classification"
    count = 0
    
    for subset in ["train", "valid", "test"]:
        subset_dir = base_rv / subset
        labels_dir = subset_dir / "labels"
        images_dir = subset_dir / "images"
        
        if not labels_dir.exists(): continue
        
        for label_file in labels_dir.glob("*.txt"):
            try:
                with open(label_file, 'r') as f:
                    content = f.readline().strip()
                
                if content:
                    class_id = int(content.split()[0])
                    if class_id in CLASS_MAPPING_ROCKVISION:
                        category = CLASS_MAPPING_ROCKVISION[class_id]
                        dest_dir = target_dir / category
                        dest_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Find matching image
                        for ext in ['.jpg', '.png', '.jpeg', '.bmp']:
                            img_path = images_dir / (label_file.stem + ext)
                            if img_path.exists():
                                shutil.copy(img_path, dest_dir / f"rv_{subset}_{img_path.name}")
                                count += 1
                                break
            except Exception:
                continue
                
    print(f"  Merged {count} images from RockVision.")

def organize_dataset():
    print("Starting dataset organization...")
    force_remove(TARGET_DIR)
    TARGET_DIR.mkdir()
    
    process_arnab(TARGET_DIR)
    process_rockvision(TARGET_DIR)
    
    print("\nDataset Statistics:")
    for category in ["Igneous", "Metamorphic", "Sedimentary"]:
        path = TARGET_DIR / category
        count = len(list(path.glob("*"))) if path.exists() else 0
        print(f"  {category}: {count} images")

if __name__ == "__main__":
    download_datasets()
    organize_dataset()
