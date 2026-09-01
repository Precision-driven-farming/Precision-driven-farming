Dataset Description, Labeling Scheme & Augmentation

Issue: #23
Milestone: Phase 2 — Foundations
Last updated: 2026-09-01

1. Overview

Here’s the rundown of the dataset we picked for the Precision-Driven Farming crop disease classification system. The aim is pretty clear: the AI model needs to tell if a crop leaf is healthy, diseased, or pest-damaged, just by looking at an image. That means we need labeled leaf images for all three categories.

Both focus crops—maize and tomato—come from one dataset:

Crop | Dataset | Source | Raw Images | Augmented Images | Classes
--- | --- | --- | --- | --- | ---
Maize | CCMT | Mendeley / Kaggle | 5,289 | 24,551 | 7
Tomato | CCMT | Mendeley / Kaggle | 5,435 | 27,178 | 5

2. Dataset — CCMT (Cashew, Cassava, Maize, Tomato)

2.1 Source & Citation

Full name: CCMT: Dataset for Crop Pest and Disease Detection  
Authors: P.K. Mensah, V. Akoto-Adjepong, K. Adu et al.  
Institution: University of Energy and Natural Resources, Sunyani, Ghana  
Published: June 2023, Data in Brief (Elsevier)  
DOI: 10.1016/j.dib.2023.109306  
Mendeley DOI: 10.17632/bwh3zbpkpv.1  
Kaggle mirror: ccmt-plant-disease-clean-verified  
License: Free for research use

2.2 Why This Dataset

- Validation by experts — plant virologists and pathologists labeled the images, and they met to clear up mislabels.
- Detailed pest classes — you’re not getting all pest damage lumped into “other.” CCMT splits out fall armyworm, grasshopper, and leaf beetle individually, which really helps with pest ID.
- Actual farm conditions — they used a Canon DSLR camera, snapped these on Ghanaian farms, and covered a range of lighting, backgrounds (plain, dark, bright, field), and angles.
- Both maize and tomato are included, so image quality and labeling are consistent between crops.
- The dataset’s published, peer-reviewed, and has a DOI—nice and legit for research or citation.

2.3 Labeling Scheme

CCMT uses a folder-based system. Each class gets its own folder, and every image inside belongs to that class. No need for a separate annotation file—the folder name is the label.

Maize Classes (7)
Class | Category | Description
--- | --- | ---
Fall armyworm | Pest | Caused by Spodoptera frugiperda larvae
Grasshopper | Pest | Feeding damage from grasshoppers
Healthy | Baseline | No visible disease or pest damage
Leaf beetle | Pest | Injury from beetle species
Leaf blight | Disease | Fungal, gives big lesions
Leaf spot | Disease | Smaller spots, usually circular/oval, fungal
Streak virus | Disease | Viral, causes yellow streaks along veins

Tomato Classes (5)
Class | Category | Description
--- | --- | ---
Healthy | Baseline | No visible disease or pest damage
Leaf blight | Disease | Fungal, brown/black lesions
Leaf curl | Disease | Viral, leaves curl upward and yellow
Septoria leaf spot | Disease | Fungal, small spots with dark borders
Verticillium wilt | Disease | Fungal, causes yellowing and wilting

Folder Structure
data/
├── Raw Data/
│   ├── Maize/
│   │   ├── fall armyworm/
│   │   ├── grasshopper/
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
│       └── verticillium wilt/
├── Augmented Data/
│   ├── Maize/
│   │   └── (same 7 subfolders)
│   └── Tomato/
│       └── (same 5 subfolders)

3. Image Counts & Class Balance

3.1 Maize

Raw data (5,289 images):

Class | Images | % of Total
--- | --- | ---
Leaf spot (largest) | 1,239 | 23.4%
Healthy (smallest) | 204 | 3.9%
Total | 5,289 | 100%
Imbalance ratio | 6.1 | (largest ÷ smallest)

Augmented data (24,551 images):

Class | Images | % of Total
--- | --- | ---
Streak virus (largest) | 5,047 | 20.6%
Healthy (smallest) | 1,041 | 4.2%
Total | 24,551 | 100%
Imbalance ratio | 4.8 | (largest ÷ smallest)

3.2 Tomato

Raw and augmented image counts per class aren’t settled yet—will be checked during EDA (see issue #24).

4. Augmentation

The dataset team already boosted the data through augmentation (cropping and resizing):

Crop | Raw → Augmented | Growth Factor
--- | --- | ---
Maize | 5,289 → 24,551 | ~4.6x
Tomato | 5,435 → 27,178 | ~5.0x

For maize, there’s still a notable class imbalance even after augmentation—the “healthy” group is just over 1,000 images, pretty small by comparison. Additional re-balancing is on the to-do list (class weighting or downsampling before training).

For tomato, class balance will be checked later during EDA.

5. Scope & Limitations

5.1 What’s Covered

- Disease detection: Several diseases covered for both crops, each with its own class.
- Pest detection: Maize set includes three named pests (fall armyworm, grasshopper, leaf beetle), so you can actually identify pests.
- Healthy baseline: Both crops have a “healthy” class for reference.

5.2 Known Limitations

- No rodent damage data: There’s just no labeled image dataset for rodent damage out there, so we’re missing that. For now, rodent detection isn’t possible—maybe in a future version.
- Maize class imbalance: Healthy maize leaves show up less often, both raw and augmented. We’ll have to actively balance this before training (class weighting, oversampling, or downsampling).
- Tomato pests: The tomato images only categorize diseases, no pest-specific folders. So pest ID in tomato is limited to whatever looks visually distinct and doesn’t overlap with disease symptoms.
- Geographic coverage: All images come from Ghana, not South Africa. Crop diseases tend to be similar across sub-Saharan Africa, but there can be local differences in how diseases show up.

6. Data Access

Crop | Access | Size
--- | --- | ---
Maize | Kaggle: ccmt-plant-disease-clean-verified → Raw Data/Maize and Augmented Data/Maize | Full CCMT (~1.22 GB raw, ~6.81 GB augmented)
Tomato | Kaggle: ccmt-plant-disease-clean-verified → Raw Data/Tomato and Augmented Data/Tomato | Same download

Note: The CCMT dataset actually includes four crops. For this project, we’re only pulling Maize and Tomato out of the set.