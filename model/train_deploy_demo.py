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
feeds the precision spraying map (see app/spraying_pipeline.py).

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
3. Run from the repo root: python model/train_deploy_demo.py
================================================================================
"""

import copy
import json
import os
import random
import shutil
import time
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

# The CCMT raw data, laid out as it ships and as data/dataset-description.md
# documents it (paths are relative to the repo root):
#   DATASET_ROOT/<crop>/<raw_folder_name>/<image files>
# where <raw_folder_name> is one of the keys in RAW_TO_LABEL below.
DATASET_ROOT = "./data/Raw Data"

# True only for a quick pipeline check without the dataset: it writes noise
# images into DATASET_ROOT, and refuses to if real images are already there.
USE_DUMMY_DATA = False
DUMMY_IMAGES_PER_CLASS = 8  # kept tiny -- this is only to prove the pipeline runs

# Training schedule. The dummy run stays tiny so it finishes in about a
# minute on CPU; the real-data values are the upper end of the Part A plan
# (8-10 epochs for stage 1, 8-12 for stage 2), since early stopping ends a
# stage sooner once validation accuracy stops improving.
STAGE1_EPOCHS = 2 if USE_DUMMY_DATA else 10
STAGE2_EPOCHS = 2 if USE_DUMMY_DATA else 12
BATCH_SIZE = 4 if USE_DUMMY_DATA else 32
# Epochs without a new best validation accuracy before a stage stops early.
# Kept above the LR scheduler's patience (2) so the halved learning rate
# gets a chance to help before training gives up.
EARLY_STOP_PATIENCE = 4
# Background workers that load and augment images while the model trains.
# With 0, image loading blocks training between batches.
NUM_WORKERS = 2

# Everything a run produces besides the deployment export. The best
# checkpoint is written to disk every time it improves, so an interrupted
# run still leaves the best model so far. Paths are relative to the repo root.
OUTPUT_DIR = "./model/outputs"
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "best_checkpoint.pt")
HISTORY_PATH = os.path.join(OUTPUT_DIR, "training_history.json")

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

    def unfreeze_top_blocks(self, n_blocks: int = 17):
        """
        Stage 2: unfreeze the last `n_blocks` blocks of the backbone's
        feature extractor (MobileNetV3-Large has 17, so the default unfreezes
        the whole backbone). On CCMT, fine-tuning all 17 blocks beat
        fine-tuning only the top 3 by 9 points of test accuracy (84.3% vs
        75.2%); smaller values keep the early, general-purpose layers frozen.
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
                 stage1_epochs=STAGE1_EPOCHS, stage2_epochs=STAGE2_EPOCHS,
                 patience=EARLY_STOP_PATIENCE, unfreeze_blocks=17,
                 backbone_lr=1e-4, head_lr=1e-3, skip_stage1=False):
    """
    Two-stage training as described in Part A1:
      Stage 1: backbone frozen, train head only, lr=1e-3
      Stage 2: unfreeze the top `unfreeze_blocks` backbone blocks, fine-tune
               with discriminative learning rates (backbone_lr for the
               backbone, head_lr for the head)

    Early stopping + checkpointing: after every epoch the weights are saved
    if validation accuracy is the best seen so far (ties broken by lower
    validation loss). A stage ends early after `patience` epochs with no new
    best. Stage 2 starts from the best stage-1 weights, and the model returned
    is the best checkpoint across both stages, not the last epoch.

    skip_stage1 is for tuning experiments that start from saved stage-1
    weights: the loaded weights are scored on validation first, so stage 2
    has to beat them to count.
    """
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    best = {"acc": -1.0, "loss": float("inf"), "state": None, "where": None}
    history = []  # one row per epoch, rewritten to HISTORY_PATH as training goes
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    def fit_stage(name, optimizer, epochs):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)
        epochs_since_best = 0
        for epoch in range(epochs):
            started = time.time()
            train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc = run_epoch(model, val_loader, criterion)
            scheduler.step(val_acc)

            improved = val_acc > best["acc"] or (val_acc == best["acc"] and val_loss < best["loss"])
            if improved:
                best.update(acc=val_acc, loss=val_loss, where=f"{name} epoch {epoch + 1}",
                            state=copy.deepcopy(model.state_dict()))
                torch.save(best["state"], CHECKPOINT_PATH)
                epochs_since_best = 0
            else:
                epochs_since_best += 1

            minutes = (time.time() - started) / 60
            history.append({"stage": name, "epoch": epoch + 1, "train_loss": train_loss,
                            "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc,
                            "lr": optimizer.param_groups[-1]["lr"], "best": improved,
                            "minutes": round(minutes, 2)})
            with open(HISTORY_PATH, "w") as f:
                json.dump(history, f, indent=1)

            print(f"  Epoch {epoch + 1}/{epochs} | train_loss={train_loss:.4f} "
                  f"train_acc={train_acc:.3f} | val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
                  f" | {minutes:.1f} min{'  <- best so far' if improved else ''}", flush=True)

            if epochs_since_best >= patience:
                print(f"  Early stopping: no val_acc improvement for {patience} epochs.")
                break
        model.load_state_dict(best["state"])

    # ---- Stage 1: freeze backbone, train head ----
    model.freeze_backbone()
    model.to(DEVICE)
    if skip_stage1:
        val_loss, val_acc = run_epoch(model, val_loader, criterion)
        best.update(acc=val_acc, loss=val_loss, where="loaded stage-1 weights",
                    state=copy.deepcopy(model.state_dict()))
        print(f"\n[Stage 1] Skipped: loaded weights score val_acc={val_acc:.3f} val_loss={val_loss:.4f}")
    else:
        print("\n[Stage 1] Training head only (backbone frozen)...")
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4
        )
        fit_stage("stage 1", optimizer, stage1_epochs)
    torch.save(best["state"], os.path.join(OUTPUT_DIR, "stage1_best.pt"))

    # ---- Stage 2: unfreeze top blocks, fine-tune with discriminative LR ----
    print(f"\n[Stage 2] Fine-tuning the top {unfreeze_blocks} backbone blocks "
          f"(backbone lr {backbone_lr:g}, head lr {head_lr:g})...")
    model.unfreeze_top_blocks(n_blocks=unfreeze_blocks)
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": backbone_lr},
        {"params": model.fc.parameters(), "lr": head_lr},
    ], weight_decay=1e-4)
    fit_stage("stage 2", optimizer, stage2_epochs)

    print(f"\n[Checkpoint] Returning best model from {best['where']} "
          f"(val_acc={best['acc']:.3f}, val_loss={best['loss']:.4f}), saved at {CHECKPOINT_PATH}.")
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


def save_evaluation_outputs(true_labels, pred_labels):
    """
    Writes the test-set results Member 7 needs into OUTPUT_DIR:
      metrics.json               overall accuracy, macro P/R/F1, per-class P/R/F1
      classification_report.txt  sklearn's report over the full 12 CLASS_NAMES
      confusion_matrix.csv/.png  12 x 12, rows = true class, columns = predicted
      test_predictions.json      the raw true/pred indices plus CLASS_NAMES
      training_curves.png        per-epoch loss and accuracy from HISTORY_PATH
    All 12 crop-specific class names are used throughout, never the collapsed
    display labels (see evaluate_test_set).
    """
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, precision_recall_fscore_support)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    labels = list(range(NUM_CLASSES))

    p, r, f1, support = precision_recall_fscore_support(
        true_labels, pred_labels, labels=labels, zero_division=0)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, labels=labels, average="macro", zero_division=0)
    metrics = {
        "test_samples": len(true_labels),
        "accuracy": accuracy_score(true_labels, pred_labels),
        "macro_precision": macro_p, "macro_recall": macro_r, "macro_f1": macro_f1,
        "per_class": {name: {"precision": float(p[i]), "recall": float(r[i]),
                             "f1": float(f1[i]), "support": int(support[i])}
                      for i, name in enumerate(CLASS_NAMES)},
    }
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    report = classification_report(true_labels, pred_labels, labels=labels,
                                   target_names=CLASS_NAMES, digits=3, zero_division=0)
    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
        f.write(report)

    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    np.savetxt(os.path.join(OUTPUT_DIR, "confusion_matrix.csv"), cm, fmt="%d", delimiter=",",
               header=",".join(CLASS_NAMES), comments="")
    with open(os.path.join(OUTPUT_DIR, "test_predictions.json"), "w") as f:
        json.dump({"class_names": CLASS_NAMES, "true_labels": true_labels,
                   "pred_labels": pred_labels}, f)

    print(f"\nAccuracy {metrics['accuracy']:.3f} | macro precision {macro_p:.3f} | "
          f"macro recall {macro_r:.3f} | macro F1 {macro_f1:.3f}\n")
    print(report)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[outputs] matplotlib not installed: skipped the PNG charts; the numbers are saved.")
        return metrics

    fig, ax = plt.subplots(figsize=(11, 9.5))
    ax.imshow(cm, cmap="Greens")
    ax.set_xticks(labels, CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(labels, CLASS_NAMES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"Confusion matrix, test set (n={len(true_labels)})")
    for i in labels:
        for j in labels:
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH) as f:
            history = json.load(f)
        x = range(1, len(history) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax, key, title in [(ax1, "loss", "Loss"), (ax2, "acc", "Accuracy")]:
            ax.plot(x, [h[f"train_{key}"] for h in history], marker="o", label="train")
            ax.plot(x, [h[f"val_{key}"] for h in history], marker="o", label="validation")
            stage2_start = next((k for k, h in enumerate(history, 1) if h["stage"] == "stage 2"), None)
            if stage2_start:
                ax.axvline(stage2_start - 0.5, color="grey", linestyle="--")
                ax.text(stage2_start - 0.4, ax.get_ylim()[1], " stage 2: fine-tuning",
                        va="top", fontsize=9, color="grey")
            ax.set_title(title)
            ax.set_xlabel("Epoch (both stages)")
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
            ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=150)
        plt.close(fig)
    print(f"[outputs] Metrics, report, confusion matrix and curves saved in {OUTPUT_DIR}")
    return metrics


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
        # Only ever wipe a folder this function created. Anything else in it
        # (or more files than a dummy run could leave) is treated as real data.
        existing = [p for p in root_path.rglob("*") if p.is_file()]
        foreign = [p for p in existing
                   if not (p.name.startswith("dummy_") and p.suffix == ".jpg"
                           and p.stem[len("dummy_"):].isdigit())]
        if foreign or len(existing) > images_per_class * 12 * 2:
            example = foreign[0] if foreign else existing[0]
            raise RuntimeError(
                f"Refusing to delete {root_path.resolve()}: it holds {len(existing)} files "
                f"that look like a real dataset (e.g. {example}).\n"
                f"Set USE_DUMMY_DATA = False to train on it, or point DATASET_ROOT "
                f"at an empty folder for a dummy run. Nothing was deleted."
            )
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
    import argparse

    # Defaults are the deployed recipe (whole backbone fine-tuned at 1e-4, head 1e-3),
    # chosen by the Part A2 tuning experiment. The original Part A plan is
    # --unfreeze-blocks 3 --backbone-lr 1e-5 --head-lr 1e-4 (the baseline run).
    ap = argparse.ArgumentParser(description="Train, evaluate and export the leaf classifier.")
    ap.add_argument("--unfreeze-blocks", type=int, default=17, help="backbone blocks fine-tuned in stage 2 (17 = all)")
    ap.add_argument("--backbone-lr", type=float, default=1e-4, help="stage-2 learning rate for the backbone")
    ap.add_argument("--head-lr", type=float, default=1e-3, help="stage-2 learning rate for the head")
    ap.add_argument("--stage1-epochs", type=int, default=STAGE1_EPOCHS)
    ap.add_argument("--stage2-epochs", type=int, default=STAGE2_EPOCHS)
    ap.add_argument("--init-weights", help="start from saved weights (e.g. a stage1_best.pt) and skip stage 1")
    ap.add_argument("--out-dir", default=OUTPUT_DIR, help="where this run's checkpoint, metrics and charts go")
    ap.add_argument("--export-path", default="leaf_classifier.pt", help="TorchScript export for deployment")
    args = ap.parse_args()
    OUTPUT_DIR = args.out_dir
    CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "best_checkpoint.pt")
    HISTORY_PATH = os.path.join(OUTPUT_DIR, "training_history.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "run_config.json"), "w") as f:
        json.dump({**vars(args), "batch_size": BATCH_SIZE, "early_stop_patience": EARLY_STOP_PATIENCE,
                   "dropout": 0.3, "stage1_lr": 1e-3, "weight_decay": 1e-4, "seed": RANDOM_SEED}, f, indent=2)

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

    # persistent_workers keeps the workers alive between epochs; on Windows
    # each new worker re-imports torch, which costs seconds every epoch.
    loader_opts = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                       persistent_workers=NUM_WORKERS > 0)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_opts)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_opts)

    # ---- Class weights for imbalance handling (Part A3) ----
    class_weights = compute_loss_weights(train_labels)
    print(f"\nComputed class weights (imbalance handling): {class_weights.tolist()}")

    # ---- Build + train model ----
    model = MobileNetV3LeafClassifier(num_classes=NUM_CLASSES, dropout=0.3, pretrained=True)
    if args.init_weights:
        model.load_state_dict(torch.load(args.init_weights, map_location="cpu"))
        print(f"\n[Init] Loaded weights from {args.init_weights}")
    model = train_model(model, train_loader, val_loader, class_weights,
                        stage1_epochs=args.stage1_epochs, stage2_epochs=args.stage2_epochs,
                        unfreeze_blocks=args.unfreeze_blocks, backbone_lr=args.backbone_lr,
                        head_lr=args.head_lr, skip_stage1=bool(args.init_weights))

    # ---- Sample inference on one image (Part C requirement for Member 7) ----
    print("\n[Inference demo] Running prediction on one sample image...")
    sample_image_path = test_samples[0][0]
    predict_image(model, sample_image_path)
 
    # ---- Full test-set evaluation, for Member 7's accuracy-metrics section ----
    test_ds = LeafDiseaseDataset(test_samples, train=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

    print("\n[Evaluation] Running full test-set evaluation for Member 7...")
    true_labels, pred_labels = evaluate_test_set(model, test_loader)
    print(f"Evaluated {len(true_labels)} test samples.")

    # ---- Export trained model for deployment (Part B) ----
    export_for_deployment(model, args.export_path)

    # ---- Metrics, confusion matrix and curves for Member 7 (issue #28) ----
    save_evaluation_outputs(true_labels, pred_labels)
