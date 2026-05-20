import os
import json
import cv2
from tqdm import tqdm

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGE_DIR = os.path.join(BASE_DIR, "dataset_raw", "images")
ANN_DIR = os.path.join(BASE_DIR, "dataset_raw", "annotations")
OUTPUT_DIR = os.path.join(BASE_DIR, "dataset_processed")

# Create output folders
classes = ["with_mask", "without_mask", "mask_weared_incorrect"]

for c in classes:
    os.makedirs(os.path.join(OUTPUT_DIR, c), exist_ok=True)

def crop_and_save(image_path, ann_path, img_name):
    image = cv2.imread(image_path)
    if image is None:
        return

    with open(ann_path, "r") as f:
        data = json.load(f)

    objects = data.get("objects", [])

    for i, obj in enumerate(objects):
        label = obj.get("classTitle")

        # skip unknown labels
        if label not in classes:
            continue

        points = obj.get("points", {}).get("exterior", [])
        if len(points) != 2:
            continue

        (x1, y1), (x2, y2) = points

        # ensure correct ordering
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        save_path = os.path.join(
            OUTPUT_DIR,
            label,
            f"{img_name}_crop_{i}.jpg"
        )

        cv2.imwrite(save_path, crop)


def main():
    images = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".png")]

    print("Total images:", len(images))

    processed = 0
    missing_json = 0

    for img_name in tqdm(images):
        image_path = os.path.join(IMAGE_DIR, img_name)
        ann_name = img_name + ".json"
        ann_path = os.path.join(ANN_DIR, ann_name)

        if not os.path.exists(ann_path):
            missing_json += 1
            print("Missing JSON:", ann_name)
            continue

        crop_and_save(image_path, ann_path, img_name.replace(".png", ""))
        processed += 1

    print("\nDONE")
    print("Processed images:", processed)
    print("Missing JSON files:", missing_json)


if __name__ == "__main__":
    main()