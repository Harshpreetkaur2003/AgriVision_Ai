
from pathlib import Path
from collections import Counter, defaultdict

import cv2
import pandas as pd
import matplotlib.pyplot as plt


# =====================================
# 1. PROJECT PATHS
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    BASE_DIR / "dataset" / "plantvillage"
    / "PlantVillage" / "PlantVillage"
)

RESULT_DIR = BASE_DIR / "results" / "dataset_exploration"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


# =====================================
# 2. CHECK DATASET
# =====================================

if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_DIR}"
    )

image_paths = [
    p for p in DATASET_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
]

if not image_paths:
    raise ValueError("No images found in dataset!")

print("\n========== DATASET INFORMATION ==========")

print("Dataset location:", DATASET_DIR)
print("Total images:", len(image_paths))


# =====================================
# 3. DETECT CLASSES
# =====================================

class_images = defaultdict(list)

for path in image_paths:
    class_images[path.parent.name].append(path)

print("\n========== DISEASE CLASSES ==========")

for name, images in sorted(class_images.items()):
    print(f"{name}: {len(images)} images")

print("\nTotal classes:", len(class_images))


# =====================================
# 4. CLASS DISTRIBUTION
# =====================================

distribution = pd.DataFrame([
    {
        "Class": name,
        "Image_Count": len(images)
    }
    for name, images in class_images.items()
])

distribution = distribution.sort_values(
    "Image_Count",
    ascending=False
)

distribution.to_csv(
    RESULT_DIR / "class_distribution.csv",
    index=False
)

print("\n", distribution.to_string(index=False))


# =====================================
# 5. IMAGE VALIDATION USING OPENCV
# =====================================

valid_images = []
corrupted_images = []
dimensions = Counter()

for i, path in enumerate(image_paths):

    image = cv2.imread(str(path))

    if image is None:
        corrupted_images.append(str(path))
        continue

    height, width = image.shape[:2]

    valid_images.append(path)
    dimensions[(width, height)] += 1

    if (i + 1) % 1000 == 0:
        print(f"Checked {i + 1}/{len(image_paths)} images")


print("\n========== IMAGE QUALITY ==========")

print("Valid images:", len(valid_images))
print("Corrupted images:", len(corrupted_images))

pd.DataFrame({
    "Corrupted_Image": corrupted_images
}).to_csv(
    RESULT_DIR / "corrupted_images.csv",
    index=False
)


# =====================================
# 6. IMAGE DIMENSIONS
# =====================================

dimension_data = [
    {
        "Width": width,
        "Height": height,
        "Count": count
    }
    for (width, height), count in dimensions.items()
]

pd.DataFrame(dimension_data).to_csv(
    RESULT_DIR / "image_dimensions.csv",
    index=False
)

print("\n========== COMMON DIMENSIONS ==========")

for dimension, count in dimensions.most_common(15):
    print(f"{dimension}: {count} images")


# =====================================
# 7. CLASS DISTRIBUTION GRAPH
# =====================================

plt.figure(figsize=(15, 7))

plt.bar(
    distribution["Class"],
    distribution["Image_Count"]
)

plt.title("PlantVillage Disease Class Distribution")
plt.xlabel("Disease Classes")
plt.ylabel("Number of Images")

plt.xticks(rotation=90)
plt.tight_layout()

plt.savefig(
    RESULT_DIR / "class_distribution.png",
    dpi=150
)

plt.show()


# =====================================
# 8. SAMPLE IMAGE VISUALIZATION
# =====================================

valid_set = set(valid_images)
samples = []

for name, paths in sorted(class_images.items()):

    valid_paths = [
        p for p in paths if p in valid_set
    ]

    samples.extend(valid_paths[:2])

    if len(samples) >= 16:
        break

rows = (len(samples) + 3) // 4

plt.figure(figsize=(16, rows * 4))

for i, path in enumerate(samples):

    image = cv2.imread(str(path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.subplot(rows, 4, i + 1)
    plt.imshow(image)
    plt.title(path.parent.name, fontsize=8)
    plt.axis("off")

plt.tight_layout()

plt.savefig(
    RESULT_DIR / "sample_images.png",
    dpi=150
)

plt.show()


# =====================================
# 9. FINAL SUMMARY
# =====================================

print("\n========== FINAL DATASET SUMMARY ==========")

print("Total images:", len(image_paths))
print("Valid images:", len(valid_images))
print("Corrupted images:", len(corrupted_images))
print("Total classes:", len(class_images))

print("\nResults saved at:")
print(RESULT_DIR)

print("\nDATASET EXPLORATION COMPLETED!")
