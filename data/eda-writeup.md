# Exploratory Data Analysis (EDA) Write-Up

> **Issue:** #24
> **Milestone:** Phase 2 — Foundations
> **Last updated:** 2026-09-01

---

## 1. Objective

Verify the CCMT dataset's suitability for training a crop disease classification model before handing off to the modelling team. This EDA checks image counts, class balance, and flags any issues that need addressing during data preparation.

---

## 2. Tools Used

| Tool | Purpose |
|---|---|
| Python | Image counting, folder traversal |
| pathlib | File system navigation |

---

## 3. Maize — EDA Results

### 3.1 Raw Data (5,289 images, 7 classes)

| Class | Images | % of Total |
|---|---|---|
| Leaf spot | 1,239 | 23.4% |
| Leaf blight | 990 | 18.7% |
| Streak virus | 965 | 18.2% |
| Leaf beetle | 933 | 17.6% |
| Grasshopper | 673 | 12.7% |
| Fall armyworm | 285 | 5.4% |
| Healthy | 204 | 3.9% |
| **Total** | **5,289** | **100%** |

- **Largest class:** Leaf spot (1,239)
- **Smallest class:** Healthy (204)
- **Imbalance ratio:** 6.1x

### 3.2 Augmented Data (24,551 images, 7 classes)

| Class | Images | % of Total |
|---|---|---|
| Streak virus | 5,047 | 20.6% |
| Leaf blight | 5,029 | 20.5% |
| Leaf beetle | 4,739 | 19.3% |
| Leaf spot | 4,285 | 17.5% |
| Grasshopper | 2,986 | 12.2% |
| Fall armyworm | 1,424 | 5.8% |
| Healthy | 1,041 | 4.2% |
| **Total** | **24,551** | **100%** |

- **Largest class:** Streak virus (5,047)
- **Smallest class:** Healthy (1,041)
- **Imbalance ratio:** 4.8x

### 3.3 Maize Key Findings

1. **Class imbalance is the primary concern.** The healthy class is severely underrepresented — 204 raw images and 1,041 augmented. A model trained on this distribution without correction will be biased toward predicting disease even on healthy crops.

2. **Augmentation helped but didn't solve it.** The imbalance ratio dropped from 6.1x to 4.8x after augmentation, but 4.8x is still significant enough to affect model performance.

3. **Fall armyworm is also underrepresented.** At 285 raw / 1,424 augmented, it's the second-smallest class. Since fall armyworm is one of the most destructive pests in sub-Saharan Africa, poor detection of this class would be a real-world problem.

---

## 4. Tomato — EDA Results

### 4.1 Raw Data (5,780 images, 5 classes)

| Class | Images | % of Total |
|---|---|---|
| Septoria leaf spot | 2,743 | 47.5% |
| Leaf blight | 1,288 | 22.3% |
| Verticillium wilt | 772 | 13.4% |
| Leaf curl | 511 | 8.8% |
| Healthy | 466 | 8.1% |
| **Total** | **5,780** | **100%** |

- **Largest class:** Septoria leaf spot (2,743)
- **Smallest class:** Healthy (466)
- **Imbalance ratio:** 5.9x

### 4.2 Augmented Data (27,168 images, 5 classes)

| Class | Images | % of Total |
|---|---|---|
| Septoria leaf spot | 11,713 | 43.1% |
| Leaf blight | 6,509 | 24.0% |
| Verticillium wilt | 3,864 | 14.2% |
| Leaf curl | 2,582 | 9.5% |
| Healthy | 2,500 | 9.2% |
| **Total** | **27,168** | **100%** |

- **Largest class:** Septoria leaf spot (11,713)
- **Smallest class:** Healthy (2,500)
- **Imbalance ratio:** 4.7x

### 4.3 Tomato Key Findings

1. **Septoria leaf spot dominates.** It accounts for nearly half the raw dataset (47.5%) and 43.1% of augmented data. The model will see this class far more than any other, risking over-prediction of septoria leaf spot.

2. **Healthy class is underrepresented again.** Same pattern as maize — 466 raw and 2,500 augmented images make it the smallest class. The same balancing strategies apply.

3. **Imbalance is comparable to maize.** Raw 5.9x → augmented 4.7x. Both crops need the same treatment before training.

---

## 5. Cross-Crop Summary

| Metric | Maize | Tomato |
|---|---|---|
| Raw image count | 5,289 | 5,780 |
| Augmented image count | 24,551 | 27,168 |
| Number of classes | 7 | 5 |
| Includes pest classes | Yes (3) | No |
| Includes healthy baseline | Yes | Yes |
| Imbalance ratio (raw) | 6.1x | 5.9x |
| Imbalance ratio (augmented) | 4.8x | 4.7x |
| Dominant class | Leaf spot → Streak virus | Septoria leaf spot |
| Weakest class | Healthy | Healthy |

---

## 6. Recommendations for Data Preparation

Based on this EDA, the following steps should be taken before model training:

1. **Address class imbalance for both crops** — choose one of:
   - Apply `class_weight='balanced'` during model training (simplest, preserves all data)
   - Oversample the healthy and other small classes with additional augmentation (rotation, flipping, brightness shifts)
   - Downsample larger classes to match the smallest class count (sacrifices data but guarantees balance)

2. **Standardise image dimensions** — check whether all images are the same resolution. If not, decide on a target size (e.g. 224×224 for standard CNN input) and resize during preprocessing.

3. **Train/validation/test split** — use a standard 70/15/15 or 80/10/10 split. Stratify by class to maintain the same class proportions across all three sets.

4. **Confirm image quality** — visually spot-check a random sample from each class to catch any mislabelled, corrupted, or low-quality images before they enter the training pipeline.

5. **Handle folder name typos** — the folders "grasshoper" and "verticulium wilt" contain spelling errors from the source data. Code must reference these exact (misspelled) names. Consider renaming locally and documenting the change, or mapping the misspelled folder names to corrected labels in the preprocessing pipeline.

---

## 7. Code Used

```python
from pathlib import Path

base_path = Path(r'C:\Users\sizwe\OneDrive\Documents\Work\SKOLO\data')
IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

for crop in ['Maize', 'Tomato']:
    for data_type in ['Raw Data', 'Augmented Data']:
        crop_path = base_path / data_type / crop
        
        print(f"\n=== {crop} ({data_type}) ===\n")
        
        counts = {}
        for class_folder in sorted(crop_path.iterdir()):
            if class_folder.is_dir():
                n = sum(1 for f in class_folder.iterdir()
                        if f.is_file() and f.suffix.lower() in IMG_EXT)
                counts[class_folder.name] = n
                print(f"  {class_folder.name}: {n} images")
        
        total = sum(counts.values())
        smallest = min(counts, key=counts.get)
        largest = max(counts, key=counts.get)
        ratio = counts[largest] / max(counts[smallest], 1)
        
        print(f"\n  TOTAL: {total} images across {len(counts)} classes")
        print(f"  Largest:  {largest} ({counts[largest]})")
        print(f"  Smallest: {smallest} ({counts[smallest]})")
        print(f"  Imbalance ratio: {ratio:.1f}x")
```