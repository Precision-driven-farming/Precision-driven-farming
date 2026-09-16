# Dataset Description, Labeling Scheme & Augmentation

> **Issue:** #23
> **Milestone:** Phase 2 — Foundations
> **Last updated:** 2026-09-01

---

## 1. Overview

This document describes the dataset selected for the Precision-Driven Farming crop disease classification system. The AI model needs to identify whether a crop leaf is healthy, diseased, or pest-damaged from an image alone — so the dataset must contain labelled leaf images across those categories.

Both target crops (maize and tomato) are sourced from a single dataset:

| Crop | Dataset | Source | Raw Images | Augmented Images | Classes |
|---|---|---|---|---|---|
| Maize | CCMT | Mendeley / Kaggle | 5,289 | 24,551 | 7 |
| Tomato | CCMT | Mendeley / Kaggle | 5,780 | 27,168 | 5 |

---

## 2. Dataset — CCMT (Cashew, Cassava, Maize, Tomato)

### 2.1 Source & Citation

- **Full name:** CCMT: Dataset for Crop Pest and Disease Detection
- **Authors:** P.K. Mensah, V. Akoto-Adjepong, K. Adu et al.
- **Institution:** University of Energy and Natural Resources, Sunyani, Ghana
- **Published:** June 2023, Data in Brief (Elsevier)
- **DOI:** 10.1016/j.dib.2023.109306
- **Mendeley DOI:** 10.17632/bwh3zbpkpv.1
- **Kaggle mirror:** `ccmt-plant-disease-clean-verified`
- **License:** Freely available for research use

### 2.2 Why This Dataset

- Expert-validated: images were annotated by plant virologists and pathologists, with a consensus conference to eliminate mislabels
- Pest-specific classes: unlike many crop disease datasets that lump all pests into one generic "pest damage" bucket, CCMT separates fall armyworm, grasshopper, and leaf beetle into distinct classes — critical for accurate pest identification
- Real-world conditions: captured on local farms in Ghana using a Canon EOS Rebel T7 DSLR under varied lighting, backgrounds (white, dark, illuminated, real field), and angles
- Both target crops (maize and tomato) are available in a single dataset, ensuring consistent image quality and labeling standards across crops
- Published and peer-reviewed with a clear DOI for academic citation

### 2.3 Labeling Scheme

The CCMT dataset uses a folder-based labeling scheme. Each class has its own subfolder, and every image inside that folder belongs to that class. No separate annotation file is needed — the folder name is the label.

#### Maize Classes (7)

| Class | Category | Description |
|---|---|---|
| Fall armyworm | Pest | Damage caused by Spodoptera frugiperda larvae |
| Grasshopper | Pest | Feeding damage from grasshopper species |
| Healthy | Baseline | No visible disease or pest damage |
| Leaf beetle | Pest | Damage from leaf beetle species |
| Leaf blight | Disease | Fungal infection causing large lesions |
| Leaf spot | Disease | Smaller circular/oval fungal lesions |
| Streak virus | Disease | Viral infection causing yellow streaks along leaf veins |

> **Note:** The folder name in the dataset is spelled "grasshoper" (one p) — this is a typo in the source data, not a different class.

#### Tomato Classes (5)

| Class | Category | Description |
|---|---|---|
| Healthy | Baseline | No visible disease or pest damage |
| Leaf blight | Disease | Fungal infection causing brown/black lesions |
| Leaf curl | Disease | Viral infection causing upward curling and yellowing of leaves |
| Septoria leaf spot | Disease | Fungal infection causing small circular spots with dark borders |
| Verticillium wilt | Disease | Fungal infection causing yellowing, wilting, and vascular discolouration |

> **Note:** The folder name in the dataset is spelled "verticulium wilt" (missing an l) — this is a typo in the source data.

#### Folder Structure

```
data/
├── Raw Data/
│   ├── Maize/
│   │   ├── fall armyworm/
│   │   ├── grasshoper/
│   │   ├── healthy/
│   │   ├── leaf beetle/
│   │   ├── leaf blight/
│   │   ├── leaf spot/
│   │   └── streak virus/
│   └── Tomato/
│       ├── healthy/
│       ├── leaf blight/
│       ├── leaf curl/
│       ├── septoria leaf spot/
│       └── verticulium wilt/
├── Augmented Data/
│   ├── Maize/
│   │   └── (same 7 subfolders)
│   └── Tomato/
│       └── (same 5 subfolders)
```

---

## 3. Image Counts & Class Balance

### 3.1 Maize

**Raw data (5,289 images):**

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

- **Imbalance ratio:** 6.1x (leaf spot ÷ healthy)

**Augmented data (24,551 images):**

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

- **Imbalance ratio:** 4.8x (streak virus ÷ healthy)

### 3.2 Tomato

**Raw data (5,780 images):**

| Class | Images | % of Total |
|---|---|---|
| Septoria leaf spot | 2,743 | 47.5% |
| Leaf blight | 1,288 | 22.3% |
| Verticillium wilt | 772 | 13.4% |
| Leaf curl | 511 | 8.8% |
| Healthy | 466 | 8.1% |
| **Total** | **5,780** | **100%** |

- **Imbalance ratio:** 5.9x (septoria leaf spot ÷ healthy)

**Augmented data (27,168 images):**

| Class | Images | % of Total |
|---|---|---|
| Septoria leaf spot | 11,713 | 43.1% |
| Leaf blight | 6,509 | 24.0% |
| Verticillium wilt | 3,864 | 14.2% |
| Leaf curl | 2,582 | 9.5% |
| Healthy | 2,500 | 9.2% |
| **Total** | **27,168** | **100%** |

- **Imbalance ratio:** 4.7x (septoria leaf spot ÷ healthy)

---

## 4. Augmentation

The dataset authors already applied augmentation (cropping and resizing) to expand the raw images:

| Crop | Raw → Augmented | Growth Factor |
|---|---|---|
| Maize | 5,289 → 24,551 | ~4.6x |
| Tomato | 5,780 → 27,168 | ~4.7x |

For both crops, augmentation reduced the imbalance ratio but did not eliminate it:

| Crop | Raw Imbalance | Augmented Imbalance | Smallest Class (Augmented) |
|---|---|---|---|
| Maize | 6.1x | 4.8x | Healthy (1,041) |
| Tomato | 5.9x | 4.7x | Healthy (2,500) |

In both cases, the "healthy" class remains the smallest. Additional balancing strategies will be needed before model training (see Recommendations in the EDA write-up, issue #24).

---

## 5. Scope & Limitations

### 5.1 What the Dataset Covers

- **Disease detection:** Both crops cover multiple disease types with clearly separated classes
- **Pest detection:** The maize subset includes three named insect pest classes (fall armyworm, grasshopper, leaf beetle), providing real pest identification capability
- **Healthy baseline:** Both crops include a "healthy" class for comparison

### 5.2 Known Limitations

- **No rodent damage data:** No publicly available, pre-labelled image dataset exists for rodent or animal damage to crops. This is a genuine gap in available agricultural datasets. Rodent damage detection is not feasible for version 1 and is noted as a future improvement
- **Class imbalance in both crops:** The healthy class is underrepresented in both maize and tomato. Septoria leaf spot dominates the tomato dataset at nearly half the images. Mitigation strategies (class weighting, oversampling, or downsampling) must be applied before training
- **Tomato pest coverage:** The tomato subset contains only disease classes and no pest-specific classes, unlike maize. Pest detection for tomato is limited to visual symptoms that overlap with disease presentations
- **Geographic origin:** Images were sourced from farms in Ghana, not South Africa. While crop diseases are broadly similar across sub-Saharan Africa, some region-specific disease presentations may differ
- **Folder name typos:** Two folders contain spelling errors ("grasshoper", "verticulium wilt"). Code referencing these folders must use the misspelled names

---

## 6. Data Access

| Crop | Access Method | Size |
|---|---|---|
| Maize | Kaggle: `ccmt-plant-disease-clean-verified` → `Raw Data/Maize` and `Augmented Data/Maize` | Part of full CCMT (~1.22 GB raw, ~6.81 GB augmented) |
| Tomato | Kaggle: `ccmt-plant-disease-clean-verified` → `Raw Data/Tomato` and `Augmented Data/Tomato` | Part of full CCMT (same download) |

**Note:** The CCMT dataset contains all four crops (Cashew, Cassava, Maize, Tomato). Only the Maize and Tomato subfolders are used for this project.