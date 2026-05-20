
import os
import random
import shutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SOURCE_DIR = os.path.join(BASE_DIR, "dataset_processed")
TARGET_DIR = os.path.join(BASE_DIR, "dataset_final")

classes = ["with_mask", "without_mask", "mask_weared_incorrect"]

splits = ["train", "val", "test"]
ratios = {"train": 0.7, "val": 0.15, "test": 0.15}

# Create folders
for split in splits:
    for c in classes:
        os.makedirs(os.path.join(TARGET_DIR, split, c), exist_ok=True)

def split_files(class_name):
    class_dir = os.path.join(SOURCE_DIR, class_name)
    files = os.listdir(class_dir)
    random.shuffle(files)

    n = len(files)
    train_end = int(n * ratios["train"])
    val_end = train_end + int(n * ratios["val"])

    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]

    return train_files, val_files, test_files

def copy_files(files, class_name, split):
    for f in files:
        src = os.path.join(SOURCE_DIR, class_name, f)
        dst = os.path.join(TARGET_DIR, split, class_name, f)
        shutil.copy2(src, dst)

def main():
    for c in classes:
        train_files, val_files, test_files = split_files(c)

        copy_files(train_files, c, "train")
        copy_files(val_files, c, "val")
        copy_files(test_files, c, "test")

        print(f"{c} split done")

    print("\nDataset splitting complete!")

if __name__ == "__main__":
    main()