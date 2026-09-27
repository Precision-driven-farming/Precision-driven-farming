"""
================================================================================
 PRECISION-DRIVEN FARMING — AI MODEL: TRAINING & INFERENCE DEMO (Part C)
 Role: Deep Learning / Training Engineer — Training Pipeline, Hyperparameter
       Tuning, Deployment Logic
================================================================================

WHAT THIS SCRIPT DEMONSTRATES
--------------------------------------------------------------------------------
This script implements the confirmed model architecture end-to-end:

    Backbone : MobileNetV3-Large, ImageNet-pretrained (transfer learning)
    Head     : Global Average Pooling -> Dropout -> Fully Connected -> Softmax
    Input    : 224x224x3 RGB
    Output   : 12-class prediction (7 maize classes + 5 tomato classes)

It ties directly back to the project theme: this is the core disease-
classification model that will run on the drone/pole hardware, turning a
captured leaf image into a disease label + confidence score, which then
feeds the precision spraying map (see app/severity.py in the model repo).

WHAT IS REAL VS. A STAND-IN IN THIS VERSION
--------------------------------------------------------------------------------
- The model architecture, preprocessing pipeline, class-weighted loss,
  two-stage freeze/fine-tune training loop, and inference function are all
  REAL, working code — not pseudocode.
- Because the real PlantVillage-style maize/tomato dataset is not loaded
  into this environment yet, the "if __name__ == '__main__'" block at the
  bottom generates a small SYNTHETIC dummy dataset (random noise images
  saved into folders using the same *misspelled* folder-name pattern the
  real dataset ships with) purely so the full pipeline can be exercised
  end-to-end and demonstrably run without errors. This is clearly labelled
  everywhere it happens. Swap DATASET_ROOT to point at the real, raw
  dataset folder and everything else runs unchanged.
- This has been syntax-checked but NOT run against the real dataset or a
  GPU in the environment this was written in. Run it in Google Colab (free
  GPU: Runtime -> Change runtime type -> GPU) once the real dataset is
  available.

HOW TO USE WITH THE REAL DATASET
--------------------------------------------------------------------------------
1. Set DATASET_ROOT below to the folder containing the raw (not
   pre-augmented) maize and tomato class folders.
2. Set USE_DUMMY_DATA = False.
3. Run: python training_inference_demo.py
================================================================================
"""

import os
import random
import shutil
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# ==============================================================================
# CONFIG
# ==============================================================================
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

IMG_SIZE = 224
NUM_CLASSES = 12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set this to the real raw dataset folder when available. Structure expected:
#DATASET_ROOT/<crop>/<raw_folder_name>/<image files>#
#where <raw_folder_name> is one of the keys in RAW_TO_LABEL below.
DATASET_ROOT = "./data/raw"

# Flip to False once the real dataset is in place at DATASET_ROOT.
USE_DUMMY_DATA = True
DUMMY_IMAGES_PER_CLASS = 6  # kept tiny -- this is only to prove the pipeline runs

# Confidence threshold for the human-verification fallback (Part B).
# Below this, the system flags the result for farmer review instead of
# treating the prediction as final. This value is a starting point —
# the report should note it will be tuned experimentally once real
# validation data is available.
CONFIDENCE_THRESHOLD = 70.0  # percent

# ImageNet normalisation stats -- required because the backbone is
# ImageNet-pretrained, so inputs must be normalised the same way the
# pretrained weights were originally trained on.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ==============================================================================
# 1. CLASS DEFINITIONS + MISSPELLED FOLDER-NAME MAPPING
# ==============================================================================
# Maps each raw (possibly misspelled) folder name to its corrected base label.
# Note: "healthy" and "leaf blight" folders exist under BOTH crops on disk,
# so this dict alone is not enough to produce 12 distinct classes -- the
# crop folder (Maize/Tomato) must also be used. See CLASS_NAMES below.
RAW_TO_LABEL: Dict[str, str] = {
    # --- Maize (7 raw folder names) ---
    "fall armyworm": "Fall armyworm",
    "grasshoper": "Grasshopper",        # documented typo — keep as-is
    "healthy": "Healthy",
    "leaf beetle": "Leaf beetle",
    "leaf blight": "Leaf blight",
    "leaf spot": "Leaf spot",
    "streak virus": "Streak virus",
    # --- Tomato (5 raw folder names) ---
    "leaf curl": "Leaf curl",
    "septoria leaf spot": "Septoria leaf spot",
    "verticulium wilt": "Verticillium wilt",   # documented typo — keep as-is
    # "healthy" and "leaf blight" keys above are reused for tomato folders too.
}

# The dataset is nested by crop: DATASET_ROOT/<crop>/<raw_folder_name>/*.jpg
CROP_FOLDERS = ["Maize", "Tomato"]

# Folder names that exist under BOTH crops and need crop disambiguation
# to produce distinct class indices.
SHARED_LABELS = {"Healthy", "Leaf blight"}


def resolve_label(crop: str, raw_folder_name: str) -> str:
    """
    Maps a (crop, raw_folder_name) pair to one of the 12 official class
    names. Disambiguates "Healthy" and "Leaf blight" per crop, since those
    folder names appear under both Maize and Tomato on disk.
    """
    base_label = RAW_TO_LABEL.get(raw_folder_name)
    if base_label is None:
        raise ValueError(
            f"Unrecognised folder name '{raw_folder_name}' under crop '{crop}'. "
            f"Add it to RAW_TO_LABEL."
        )
    if base_label in SHARED_LABELS:
        return f"{base_label} ({crop})"
    return base_label


# The 12 official classes, in a fixed order, so class index <-> label is
# stable across training, saving, and inference.
CLASS_NAMES: List[str] = [
    # Maize (7)
    "Fall armyworm", "Grasshopper", "Healthy (Maize)", "Leaf beetle",
    "Leaf blight (Maize)", "Leaf spot", "Streak virus",
    # Tomato (5)
    "Healthy (Tomato)", "Leaf blight (Tomato)", "Leaf curl",
    "Septoria leaf spot", "Verticillium wilt",
]
NUM_CLASSES = len(CLASS_NAMES)
assert NUM_CLASSES == 12


def display_label(class_name: str) -> str:
    """Strip the crop-disambiguation suffix for the farmer-facing output
    string (e.g. 'Healthy (Maize)' -> 'Healthy'), matching the exact
    output format Member 7 needs: '<label>: <confidence>%'."""
    return class_name.split(" (")[0]




# ==============================================================================
# 2. PREPROCESSING (OpenCV + torchvision), in the specified order:
#    validate -> resize -> RGB convert -> augment (train only) -> normalize -> tensor
# ==============================================================================

def load_and_preprocess_cv2(image_path: str) -> np.ndarray:
    """
    OpenCV preprocessing stage: validate -> resize -> RGB convert.
    Returns an RGB uint8 numpy array of shape (224, 224, 3), or raises a
    clear error if the image could not be read (this is the "validate" step
    -- corrupt/missing frames must never silently proceed to inference).
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image (corrupt or missing): {image_path}")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV loads as BGR by default
    return img


# torchvision transforms handle: augment (train split only) -> normalize -> tensor
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class LeafDiseaseDataset(Dataset):
    """
    Expects a list of (image_path, class_index) pairs, already resolved from
    the (possibly misspelled) raw folder names to the 12 corrected class
    indices via RAW_TO_LABEL above.
    """

    def __init__(self, samples, train: bool):
        self.samples = samples
        self.transform = train_transform if train else eval_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = load_and_preprocess_cv2(path)       # validate -> resize -> RGB
        tensor = self.transform(img)              # augment (train only) -> normalize -> tensor
        return tensor, label


def build_sample_list(dataset_root: str):
    """
    Walks dataset_root/<crop>/<raw_folder_name>/*.jpg, resolving each
    (crop, raw_folder_name) pair to one of the 12 official class indices
    via resolve_label(). Handles the two documented folder-name typos
    automatically since RAW_TO_LABEL maps them directly.
    """
    samples = []
    root = Path(dataset_root)
    for crop in CROP_FOLDERS:
        crop_dir = root / crop
        if not crop_dir.exists():
            print(f"[warn] expected crop folder not found, skipping: {crop_dir}")
            continue
        for raw_folder in crop_dir.iterdir():
            if not raw_folder.is_dir():
                continue
            try:
                label = resolve_label(crop, raw_folder.name)
            except ValueError as e:
                print(f"[warn] {e}")
                continue
            class_idx = CLASS_NAMES.index(label)
            for img_path in raw_folder.glob("*"):
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    samples.append((str(img_path), class_idx))
    return samples

# ==============================================================================
# 3. MODEL: MobileNetV3-Large backbone + custom head
#    GAP -> Dropout -> Fully Connected -> Softmax (applied at inference only)
# ==============================================================================

class MobileNetV3LeafClassifier(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.3, pretrained: bool = True):
        super().__init__()
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.mobilenet_v3_large(weights=weights)

        # torchvision's mobilenet_v3_large forward pass is:
        #   x = self.features(x); x = self.avgpool(x); x = flatten(x); x = self.classifier(x)
        # We keep `features` + `avgpool` (the Global Average Pooling the
        # architecture calls for) and discard the original ImageNet
        # classifier, replacing it with our own head below.
        in_features = backbone.classifier[0].in_features  # 960 for MobileNetV3-Large
        backbone.classifier = nn.Identity()  # backbone(x) now returns the pooled feature vector
        self.backbone = backbone

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features, num_classes)
        # NOTE: Softmax is intentionally NOT applied here. nn.CrossEntropyLoss
        # expects raw logits and applies log-softmax internally during
        # training; applying softmax twice would break gradient scaling.
        # Softmax is applied explicitly at inference time in predict_image().

    def forward(self, x):
        x = self.backbone(x)   # -> (batch, 960) pooled features (GAP already applied)
        x = self.dropout(x)
        x = self.fc(x)         # -> (batch, num_classes) raw logits
        return x

    def freeze_backbone(self):
        """Stage 1: freeze all backbone weights, train only the head."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_top_blocks(self, n_blocks: int = 3):
        """
        Stage 2: unfreeze only the last `n_blocks` inverted-residual blocks
        of the backbone's feature extractor (not the whole network), so the
        low-level, general-purpose early layers stay frozen while the more
        task-specific later layers adapt to leaf disease patterns.
        """
        feature_blocks = list(self.backbone.features.children())
        for block in feature_blocks[-n_blocks:]:
            for p in block.parameters():
                p.requires_grad = True


# ==============================================================================
# 4. TRAINING LOOP (two-stage: freeze -> fine-tune)
# ==============================================================================

def compute_loss_weights(labels):
    """Class weights inversely proportional to frequency, for imbalance (Part A3)."""
    classes = np.arange(NUM_CLASSES)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=np.array(labels))
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            if is_train:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def train_model(model, train_loader, val_loader, class_weights,
                 stage1_epochs=2, stage2_epochs=2):
    """
    Two-stage training as described in Part A1:
      Stage 1: backbone frozen, train head only, lr=1e-3
      Stage 2: unfreeze top backbone blocks, fine-tune with discriminative
               learning rates (small for backbone, larger for head)

    NOTE: stage1_epochs / stage2_epochs default to a small number (2 each)
    here purely so this demo completes quickly on the synthetic dummy data.
    On the real dataset, use the epoch counts recommended in Part A
    (roughly 8-10 for stage 1, 8-12 for stage 2, with early stopping).
    """
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

    # ---- Stage 1: freeze backbone, train head ----
    print("\n[Stage 1] Training head only (backbone frozen)...")
    model.freeze_backbone()
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)

    for epoch in range(stage1_epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion)
        scheduler.step(val_acc)
        print(f"  Epoch {epoch + 1}/{stage1_epochs} | train_loss={train_loss:.4f} "
              f"train_acc={train_acc:.3f} | val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

    # ---- Stage 2: unfreeze top blocks, fine-tune with discriminative LR ----
    print("\n[Stage 2] Fine-tuning top backbone blocks...")
    model.unfreeze_top_blocks(n_blocks=3)
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": 1e-5},
        {"params": model.fc.parameters(), "lr": 1e-4},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)

    best_val_acc = 0.0
    for epoch in range(stage2_epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion)
        scheduler.step(val_acc)
        best_val_acc = max(best_val_acc, val_acc)
        print(f"  Epoch {epoch + 1}/{stage2_epochs} | train_loss={train_loss:.4f} "
              f"train_acc={train_acc:.3f} | val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

    return model


# ==============================================================================
# 5. INFERENCE — matches the exact required output format: "Leaf spot: 91%"
# ==============================================================================

def predict_image(model, image_path: str):
    """
    Runs inference on a single image and applies the confidence-threshold
    fallback described in Part B: predictions below CONFIDENCE_THRESHOLD
    are flagged for human verification rather than treated as final,
    per Team 1's requirement that farmers must review results before
    any spraying decision is made.
    """
    model.eval()
    img = load_and_preprocess_cv2(image_path)
    tensor = eval_transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)

    label = display_label(CLASS_NAMES[pred_idx.item()])
    confidence_pct = confidence.item() * 100
    needs_verification = confidence_pct < CONFIDENCE_THRESHOLD

    # Exact output format required for Member 7's accuracy-metrics section
    print(f"{label}: {confidence_pct:.0f}%")
    if needs_verification:
        print(f"  [LOW CONFIDENCE — below {CONFIDENCE_THRESHOLD:.0f}% threshold] "
              f"Flagging for farmer verification before any spraying action.")

    return label, confidence_pct, needs_verification
 
def evaluate_test_set(model, test_loader):
    """
    Runs the model over the full test set and returns (true_labels,
    predicted_labels) as flat lists of class indices — the exact inputs
    Member 7 needs for sklearn's confusion_matrix / classification_report.

    IMPORTANT for Member 7: these are indices into the 12-entry CLASS_NAMES
    list (with crop disambiguation, e.g. "Healthy (Maize)" vs
    "Healthy (Tomato)"), NOT the collapsed display strings used in the
    printed inference output. Build your confusion matrix using CLASS_NAMES
    as the axis labels, not display_label(CLASS_NAMES[i]) — otherwise a
    maize-Healthy sample misclassified as tomato-Healthy would look like a
    correct prediction instead of an error.
    """
    model.eval()
    true_labels, pred_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            logits = model(images)
            preds = logits.argmax(dim=1)
            true_labels.extend(labels.tolist())
            pred_labels.extend(preds.cpu().tolist())
    return true_labels, pred_labels
 

def export_for_deployment(model, export_path: str = "leaf_classifier.pt"):
    """
    Exports the trained model to TorchScript for edge deployment on
    drone/pole hardware, per Part B.

    TorchScript is chosen over ONNX here because the deployment target is
    still Python/PyTorch-based (drone companion computer or a field laptop
    running the same stack used for training), so no cross-framework
    conversion is needed. If a future deployment target requires a
    non-PyTorch runtime (e.g. a microcontroller or a mobile app using
    TensorFlow Lite), ONNX export would need to be added instead — this
    is noted as a Future Improvement, not a current requirement.
    """
    model.eval()
    example_input = torch.rand(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    traced_model = torch.jit.trace(model, example_input)
    traced_model.save(export_path)
    print(f"[export] Model exported for deployment -> {export_path}")
    return export_path
 
# ==============================================================================
# 6. DUMMY DATA GENERATOR (demo-only — replaced by real dataset when available)
# ==============================================================================

def generate_dummy_dataset(root: str, images_per_class: int = DUMMY_IMAGES_PER_CLASS):
    """
    Creates synthetic random-noise images under DATASET_ROOT/<crop>/<raw_folder_name>/,
    matching the real dataset's nested structure, purely so the pipeline can be
    exercised end-to-end without the real dataset present.
    """
    root_path = Path(root)
    if root_path.exists():
        shutil.rmtree(root_path)
    root_path.mkdir(parents=True)

    maize_folders = ["fall armyworm", "grasshoper", "healthy", "leaf beetle",
                      "leaf blight", "leaf spot", "streak virus"]
    tomato_folders = ["healthy", "leaf blight", "leaf curl",
                       "septoria leaf spot", "verticulium wilt"]

    for crop, folders in [("Maize", maize_folders), ("Tomato", tomato_folders)]:
        for raw_name in folders:
            class_dir = root_path / crop / raw_name
            class_dir.mkdir(parents=True)
            for i in range(images_per_class):
                noise = np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
                Image.fromarray(noise).save(class_dir / f"dummy_{i}.jpg")

    print(f"[dummy data] Generated {images_per_class} synthetic images per class "
          f"across {len(maize_folders)} maize + {len(tomato_folders)} tomato classes under {root}")

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":

    if USE_DUMMY_DATA:
        print("=" * 70)
        print("USE_DUMMY_DATA = True -> generating synthetic placeholder images.")
        print("Replace with the real raw dataset and set USE_DUMMY_DATA = False")
        print("before reporting any accuracy numbers from this script.")
        print("=" * 70)
        generate_dummy_dataset(DATASET_ROOT)

    # ---- Build sample list from (possibly misspelled) folder structure ----
    all_samples = build_sample_list(DATASET_ROOT)
    print(f"\nTotal images found: {len(all_samples)}")

    # ---- Stratified 70/15/15 split (Part A4) ----
    paths = [s[0] for s in all_samples]
    labels = [s[1] for s in all_samples]

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels, test_size=0.30, stratify=labels, random_state=RANDOM_SEED
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.50, stratify=temp_labels, random_state=RANDOM_SEED
    )

    train_samples = list(zip(train_paths, train_labels))
    val_samples = list(zip(val_paths, val_labels))
    test_samples = list(zip(test_paths, test_labels))
    print(f"Split -> train: {len(train_samples)}, val: {len(val_samples)}, test: {len(test_samples)}")

    train_ds = LeafDiseaseDataset(train_samples, train=True)
    val_ds = LeafDiseaseDataset(val_samples, train=False)

    # batch_size kept small since this is dummy/demo data; use 32 on real data
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    # ---- Class weights for imbalance handling (Part A3) ----
    class_weights = compute_loss_weights(train_labels)
    print(f"\nComputed class weights (imbalance handling): {class_weights.tolist()}")

    # ---- Build + train model ----
    model = MobileNetV3LeafClassifier(num_classes=NUM_CLASSES, dropout=0.3, pretrained=True)
    model = train_model(model, train_loader, val_loader, class_weights,
                        stage1_epochs=2, stage2_epochs=2)

    # ---- Sample inference on one image (Part C requirement for Member 7) ----
    print("\n[Inference demo] Running prediction on one sample image...")
    sample_image_path = test_samples[0][0]
 
    # ---- Full test-set evaluation, for Member 7's accuracy-metrics section ----
    test_ds = LeafDiseaseDataset(test_samples, train=False)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)

    print("\n[Evaluation] Running full test-set evaluation for Member 7...")
    true_labels, pred_labels = evaluate_test_set(model, test_loader)
    print(f"Evaluated {len(true_labels)} test samples.")
    print("Pass `true_labels`, `pred_labels`, and `CLASS_NAMES` to Member 7 "
          "for confusion_matrix / classification_report.")

    print(f"Evaluated {len(true_labels)} test samples.")
    print("Pass `true_labels`, `pred_labels`, and `CLASS_NAMES` to Member 7 "
          "for confusion_matrix / classification_report.")

    # ---- Export trained model for deployment (Part B) ----
    export_for_deployment(model)
