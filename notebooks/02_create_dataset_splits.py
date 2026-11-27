
from pathlib import Path
import pandas as pd
import json

from sklearn.model_selection import train_test_split

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    BASE_DIR / "dataset" / "plantvillage"
    / "PlantVillage" / "PlantVillage"
)

RESULT_DIR = BASE_DIR / "results" / "dataset_splits"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Supported image formats
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Collect image paths and class names
records = []

for class_folder in sorted(DATASET_DIR.iterdir()):

    if not class_folder.is_dir():
        continue

    class_name = class_folder.name

    for image_path in class_folder.rglob("*"):

        if image_path.suffix.lower() in IMAGE_EXTENSIONS:

            records.append({
                "image_path": str(
                    image_path.relative_to(BASE_DIR)
                ).replace("\\", "/"),
                "class_name": class_name
            })

# Create DataFrame
df = pd.DataFrame(records)

if df.empty:
    raise ValueError("No images found. Check dataset location.")

# Assign numerical labels
classes = sorted(df["class_name"].unique())

class_indices = {
    class_name: index
    for index, class_name in enumerate(classes)
}

df["label"] = df["class_name"].map(class_indices)

print("\n========== DATASET INFORMATION ==========")
print("Total images:", len(df))
print("Total classes:", len(classes))

# First split: 70% training, 30% temporary
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["class_name"]
)

# Second split: 15% validation, 15% testing
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["class_name"]
)

# Save CSV files
train_df.to_csv(RESULT_DIR / "train.csv", index=False)
val_df.to_csv(RESULT_DIR / "validation.csv", index=False)
test_df.to_csv(RESULT_DIR / "test.csv", index=False)

# Save class-to-number mapping
with open(RESULT_DIR / "class_indices.json", "w") as file:
    json.dump(class_indices, file, indent=4)

# Class distribution summary
summary = pd.DataFrame({
    "Training": train_df["class_name"].value_counts(),
    "Validation": val_df["class_name"].value_counts(),
    "Testing": test_df["class_name"].value_counts()
}).fillna(0).astype(int)

summary.to_csv(RESULT_DIR / "split_summary.csv")

# Display results
print("\n========== DATASET SPLIT SUMMARY ==========")
print("Training images:", len(train_df))
print("Validation images:", len(val_df))
print("Testing images:", len(test_df))

print("\n========== CLASS-WISE DISTRIBUTION ==========")
print(summary)

# Verify no image path appears in multiple sets
assert set(train_df["image_path"]).isdisjoint(
    set(val_df["image_path"])
)

assert set(train_df["image_path"]).isdisjoint(
    set(test_df["image_path"])
)

assert set(val_df["image_path"]).isdisjoint(
    set(test_df["image_path"])
)

assert len(train_df) + len(val_df) + len(test_df) == len(df)

print("\nNo overlapping image paths detected.")
print("\nDataset splitting completed successfully!")
print("Results saved at:", RESULT_DIR)
