"""
================================================================================
 PRECISION DRIVEN FARMING AI - TRAINING, EVALUATION & DEPLOYMENT SCRIPT
================================================================================
Model     : MobileNetV3-Large (ImageNet pre-trained, torchvision) + custom head
Head      : Global Average Pooling -> Dropout -> Fully Connected (12 logits)
Input     : 224 x 224 x 3 RGB leaf image
Classes   : 12 crop-specific classes (7 maize + 5 tomato) from the CCMT dataset
Training  : two-stage transfer learning
              Stage 1 - backbone frozen, head only, lr 1e-3
              Stage 2 - last 3 inverted-residual blocks unfrozen, lr 1e-5,
                        head lr 1e-4
            AdamW, class-weighted cross-entropy, ReduceLROnPlateau,
            EARLY STOPPING on validation accuracy, best-epoch checkpointing.


--------------------------------------------------------------------------------

HOW TO RUN
--------------------------------------------------------------------------------
    pip install torch torchvision opencv-python scikit-learn pandas pillow matplotlib
    python training_inference_demo.py

  * Set DATASET_ROOT below (or the DATASET_ROOT environment variable).
  * Expected layouts (detected automatically):
        A) DATASET_ROOT/train/<Crop>/<class>/*.jpg
           DATASET_ROOT/val/...  and  DATASET_ROOT/test/...   (already split)
        B) DATASET_ROOT/train/<Crop>/<class>/*.jpg only
           -> everything found is pooled and split 70/15/15 (stratified)
        C) DATASET_ROOT/<Crop>/<class>/*.jpg
           -> pooled and split 70/15/15 (stratified)
  * USE THE RAW IMAGES, NOT THE AUTHOR-AUGMENTED ONES. Augmented images are
    cropped/resized copies of the same photos and leak into the test set.
  * A GPU is strongly recommended (Google Colab: Runtime > Change runtime type
    > GPU). On CPU this will take hours.

OUTPUTS (written to ./outputs next to this script)
--------------------------------------------------------------------------------
    best_model.pth            best weights (state_dict)
    leaf_classifier.pt        TorchScript export for deployment
    split_train/val/test.csv  exact images used in each split
    training_history.csv      per-epoch loss/accuracy/learning rate
    results_table.csv / .md   per-class precision, recall, F1, support
    success_criteria.csv      checks against the business success criteria
    confusion_matrix.csv/.png confusion matrix (12 x 12, crop-specific labels)
    results_summary.md        everything above in one readable file

STATUS OF THE NUMBERS
--------------------------------------------------------------------------------
No accuracy figure exists until this script is run on the real dataset.
USE_DUMMY_DATA = True only proves the pipeline runs (random-noise images);
its outputs are marked DUMMY and must never be reported.
================================================================================
"""

import copy
import hashlib
import os
import random
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# ==============================================================================
# CONFIG
# ==============================================================================
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Folder that contains train/ (and optionally val/ and test/). Raw string so the
# Windows backslashes are kept. Override with an environment variable on Colab.
DATASET_ROOT = os.environ.get(
    "DATASET_ROOT",
    r"C:\Users\nhlan\Documents\My Coding Projects\Precision driven farming\resources\Datasets",
)

USE_DUMMY_DATA = False  # True = random-noise smoke test only (results are meaningless)
PRETRAINED = True       # needs internet the first time (downloads ImageNet weights)

IMG_SIZE = 224
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)  # train / validation / test

BATCH_SIZE = 32          # 16-32 per the report; lower it if you run out of memory
NUM_WORKERS = 0          # 0 is the safe choice on Windows; try 2-4 on Colab/Linux
DROPOUT = 0.3
N_UNFROZEN_BLOCKS = 3

STAGE1_MAX_EPOCHS = 10
STAGE2_MAX_EPOCHS = 12
EARLY_STOP_PATIENCE = 3  # epochs without validation-accuracy improvement
LR_SCHEDULER_PATIENCE = 2

VALIDATE_IMAGES = True   # try to read every image once and drop unreadable ones
CHECK_DUPLICATES = True  # drop exact duplicate photos found across splits

CONFIDENCE_THRESHOLD = 70.0  # percent; below this -> flagged for farmer verification

# Business success criteria (Precision Driven Farming AI)
TARGET_ACCURACY = 0.90       # criterion 1
TARGET_RECALL = 0.90         # criterion 2
TARGET_SECONDS_PER_IMAGE = 10.0  # criterion 4
LATENCY_SAMPLE_SIZE = 50

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}  # same as the EDA

# Raw image counts from the dataset description (used to detect augmented data)
EXPECTED_RAW_COUNTS = {"Maize": 5289, "Tomato": 5780}

# ==============================================================================
# 1. CLASSES AND FOLDER-NAME HANDLING
# ==============================================================================
# 12 crop-specific classes in a FIXED order (index <-> label stays stable).
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
CROP_OF_CLASS = ["Maize"] * 7 + ["Tomato"] * 5
CROP_FOLDERS = {"maize": "Maize", "tomato": "Tomato"}  # lower-case folder -> crop name

# Normalised folder name -> base label. Normalising = lower case, "_" and "-"
# turned into spaces, so  fall_armyworm == "Fall Armyworm" == "fall armyworm".
# Both the source typos and the correct spellings are accepted.
FOLDER_TO_LABEL: Dict[str, str] = {
    "fall armyworm": "Fall armyworm",
    "grasshoper": "Grasshopper",      # typo in the source dataset
    "grasshopper": "Grasshopper",
    "healthy": "Healthy",
    "leaf beetle": "Leaf beetle",
    "leaf blight": "Leaf blight",
    "leaf spot": "Leaf spot",
    "streak virus": "Streak virus",
    "leaf curl": "Leaf curl",
    "septoria leaf spot": "Septoria leaf spot",
    "verticulium wilt": "Verticillium wilt",   # typo in the source dataset
    "verticillium wilt": "Verticillium wilt",
}
SHARED_LABELS = {"Healthy", "Leaf blight"}  # exist under BOTH crops


def normalise_folder_name(name: str) -> str:
    return " ".join(name.lower().replace("_", " ").replace("-", " ").split())


def resolve_label(crop: str, folder_name: str) -> str:
    """(crop, folder name) -> one of the 12 official class names."""
    base = FOLDER_TO_LABEL.get(normalise_folder_name(folder_name))
    if base is None:
        raise ValueError(f"Unrecognised class folder '{folder_name}' under '{crop}'. "
                         f"Add it to FOLDER_TO_LABEL.")
    label = f"{base} ({crop})" if base in SHARED_LABELS else base
    if label not in CLASS_NAMES or CROP_OF_CLASS[CLASS_NAMES.index(label)] != crop:
        raise ValueError(f"'{folder_name}' under '{crop}' maps to '{label}', "
                         f"which is not one of the {crop} classes.")
    return label


def display_label(class_name: str) -> str:
    """'Healthy (Maize)' -> 'Healthy' for the farmer-facing output string."""
    return class_name.split(" (")[0]


# ==============================================================================
# 2. DATA DISCOVERY, VALIDATION, SPLITTING
# ==============================================================================
SPLIT_ALIASES = {"train": ["train"], "val": ["val", "valid", "validation"], "test": ["test"]}


def find_split_dirs(root: Path) -> Dict[str, Path]:
    """Find train/val/test sub-folders (or the crop folders directly under root)."""
    found = {}
    for key, aliases in SPLIT_ALIASES.items():
        for alias in aliases:
            for child in root.iterdir():
                if child.is_dir() and child.name.lower() == alias:
                    found[key] = child
    if not found and any(c.is_dir() and c.name.lower() in CROP_FOLDERS for c in root.iterdir()):
        found["all"] = root
    return found


def collect_samples(split_dir: Path) -> List[Tuple[str, int]]:
    """Walk split_dir/<Crop>/<class>/*.<image> -> list of (path, class index)."""
    samples = []
    for crop_dir in sorted(split_dir.iterdir()):
        if not crop_dir.is_dir() or crop_dir.name.lower() not in CROP_FOLDERS:
            continue
        crop = CROP_FOLDERS[crop_dir.name.lower()]
        for class_dir in sorted(crop_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            try:
                label = resolve_label(crop, class_dir.name)
            except ValueError as e:
                print(f"[warn] {e} - skipped.")
                continue
            idx = CLASS_NAMES.index(label)
            for f in sorted(class_dir.iterdir()):
                if f.is_file() and f.suffix.lower() in IMG_EXT:
                    samples.append((str(f), idx))
    return samples


def check_looks_raw(samples: List[Tuple[str, int]]) -> None:
    """Warn if the pooled image counts look like the AUGMENTED dataset."""
    counts = {"Maize": 0, "Tomato": 0}
    for _, idx in samples:
        counts[CROP_OF_CLASS[idx]] += 1
    print("\nImages found per crop (pooled):", counts,
          "| expected RAW:", EXPECTED_RAW_COUNTS)
    for crop, n in counts.items():
        exp = EXPECTED_RAW_COUNTS[crop]
        if n > exp * 1.15:
            print(f"[WARNING] {crop}: {n} images found but the RAW dataset has about {exp}. "
                  f"This looks like the AUGMENTED data. Augmented images are copies of "
                  f"the same photos, so splitting them randomly LEAKS test images into "
                  f"training and inflates accuracy. Use the raw folders.")
        elif n < exp * 0.85:
            print(f"[WARNING] {crop}: only {n} images found, fewer than the raw {exp}. "
                  f"Check that the whole dataset was copied.")


def validate_images(samples):
    """Drop images OpenCV cannot read (corrupted / unsupported)."""
    good, bad = [], []
    for path, idx in samples:
        (good if cv2.imread(path) is not None else bad).append((path, idx))
    if bad:
        print(f"[validate] {len(bad)} unreadable image(s) removed, e.g. {bad[0][0]}")
    else:
        print(f"[validate] all {len(good)} images readable.")
    return good


def file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def drop_cross_split_duplicates(train, val, test):
    """Remove from val/test any image that is byte-identical to an earlier split's image."""
    seen = {file_md5(p) for p, _ in train}
    new_val, dropped_val = [], 0
    for p, y in val:
        h = file_md5(p)
        if h in seen:
            dropped_val += 1
        else:
            new_val.append((p, y))
    seen |= {file_md5(p) for p, _ in new_val}
    new_test, dropped_test = [], 0
    for p, y in test:
        if file_md5(p) in seen:
            dropped_test += 1
        else:
            new_test.append((p, y))
    print(f"[duplicates] removed {dropped_val} from validation and {dropped_test} from test "
          f"(identical to images in an earlier split).")
    return train, new_val, new_test


def stratified_split(samples, seed=RANDOM_SEED):
    """70/15/15 split, stratified by class, on the pooled RAW images."""
    paths = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    _, val_frac, test_frac = SPLIT_FRACTIONS
    tr_p, tmp_p, tr_l, tmp_l = train_test_split(
        paths, labels, test_size=val_frac + test_frac, stratify=labels, random_state=seed)
    va_p, te_p, va_l, te_l = train_test_split(
        tmp_p, tmp_l, test_size=test_frac / (val_frac + test_frac),
        stratify=tmp_l, random_state=seed)
    return list(zip(tr_p, tr_l)), list(zip(va_p, va_l)), list(zip(te_p, te_l))


def build_splits(root: str):
    root_path = Path(root)
    if not root_path.exists():
        raise SystemExit(f"DATASET_ROOT does not exist: {root_path}")
    found = find_split_dirs(root_path)
    print("Split folders found:", {k: str(v) for k, v in found.items()} or "none")
    if not found:
        raise SystemExit("No train/val/test folders and no Maize/Tomato folders under DATASET_ROOT.")

    if {"train", "val", "test"} <= set(found):
        print("Layout A: using the dataset's own train / val / test folders.")
        train, val, test = (collect_samples(found[k]) for k in ("train", "val", "test"))
        check_looks_raw(train + val + test)
    else:
        print("Layout B/C: pooling everything found and splitting 70/15/15 (stratified).")
        pooled = []
        for d in found.values():
            pooled += collect_samples(d)
        check_looks_raw(pooled)
        if VALIDATE_IMAGES:
            pooled = validate_images(pooled)
        train, val, test = stratified_split(pooled)

    if VALIDATE_IMAGES and {"train", "val", "test"} <= set(found):
        train, val, test = validate_images(train), validate_images(val), validate_images(test)
    if CHECK_DUPLICATES:
        train, val, test = drop_cross_split_duplicates(train, val, test)
    return train, val, test


def save_split_lists(train, val, test):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, data in (("train", train), ("val", val), ("test", test)):
        pd.DataFrame({"path": [p for p, _ in data],
                      "label_index": [y for _, y in data],
                      "label_name": [CLASS_NAMES[y] for _, y in data]}
                     ).to_csv(OUTPUT_DIR / f"split_{name}.csv", index=False)


def print_split_table(train, val, test):
    df = pd.DataFrame({
        "Class": CLASS_NAMES,
        "Train": [sum(1 for _, y in train if y == i) for i in range(NUM_CLASSES)],
        "Val": [sum(1 for _, y in val if y == i) for i in range(NUM_CLASSES)],
        "Test": [sum(1 for _, y in test if y == i) for i in range(NUM_CLASSES)],
    })
    print("\nImages per class and split:")
    print(df.to_string(index=False))
    print(f"TOTAL: train {len(train)} | val {len(val)} | test {len(test)}")


# ==============================================================================
# 3. PREPROCESSING:  validate -> resize -> RGB -> augment (train only) -> normalise -> tensor
# ==============================================================================
def load_and_preprocess_cv2(image_path: str) -> np.ndarray:
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image (corrupt or missing): {image_path}")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV loads BGR


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
    """(image_path, class_index) pairs. Augmentation only when train=True."""

    def __init__(self, samples, train: bool):
        self.samples = samples
        self.transform = train_transform if train else eval_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return self.transform(load_and_preprocess_cv2(path)), label


# ==============================================================================
# 4. MODEL
# ==============================================================================
class MobileNetV3LeafClassifier(nn.Module):
    """MobileNetV3-Large backbone (features + GAP) -> Dropout -> Linear(960 -> 12).

    torchvision's MobileNetV3-Large `features` has 17 children:
        [0]      stem conv
        [1..15]  the 15 inverted-residual blocks
        [16]     final 1x1 conv (960 channels)
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = DROPOUT,
                 pretrained: bool = PRETRAINED):
        super().__init__()
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.mobilenet_v3_large(weights=weights)
        in_features = backbone.classifier[0].in_features  # 960
        backbone.classifier = nn.Identity()  # backbone(x) -> pooled 960-d vector (GAP included)
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features, num_classes)
        # No Softmax here: CrossEntropyLoss expects raw logits. Softmax is applied
        # only at inference time.

    def forward(self, x):
        return self.fc(self.dropout(self.backbone(x)))

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_top_blocks(self, n_blocks: int = N_UNFROZEN_BLOCKS):
        """Unfreeze the last n inverted-residual blocks (stem and final conv stay frozen)."""
        feats = self.backbone.features
        block_indices = list(range(1, len(feats) - 1))  # exclude stem [0] and final conv [-1]
        chosen = block_indices[-n_blocks:]
        for i in chosen:
            for p in feats[i].parameters():
                p.requires_grad = True
        print(f"[unfreeze] features has {len(feats)} children; unfroze inverted-residual "
              f"blocks at indices {chosen} ({type(feats[chosen[0]]).__name__}).")
        return chosen

    def apply_bn_policy(self):
        """Keep BatchNorm layers whose parameters are frozen in eval mode so their
        pre-trained running statistics are not changed by small batches."""
        for m in self.backbone.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                if not any(p.requires_grad for p in m.parameters()):
                    m.eval()

    def count_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ==============================================================================
# 5. TRAINING (two stages, early stopping, best-epoch checkpoint)
# ==============================================================================
def compute_loss_weights(labels):
    """Weights inversely proportional to class frequency (imbalance handling)."""
    present = set(labels)
    missing = [CLASS_NAMES[i] for i in range(NUM_CLASSES) if i not in present]
    if missing:
        raise SystemExit(f"These classes have NO training images: {missing}. "
                         f"Check DATASET_ROOT and the folder names.")
    w = compute_class_weight(class_weight="balanced", classes=np.arange(NUM_CLASSES),
                             y=np.array(labels))
    return torch.tensor(w, dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    if is_train:
        model.train()
        model.apply_bn_policy()  # frozen layers' BatchNorm stays in eval mode
    else:
        model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with (torch.enable_grad() if is_train else torch.no_grad()):
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
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += images.size(0)
    return total_loss / total, correct / total


def fit_stage(model, train_loader, val_loader, criterion, optimizer, scheduler,
              stage, max_epochs, patience, history, best_acc=-1.0, best_state=None):
    """Train one stage with early stopping. Returns (best_val_acc, best_state, epochs_run, stopped_early).
    The model is restored to the best validation epoch before returning."""
    bad_epochs, epochs_run, stopped_early = 0, 0, False
    for epoch in range(1, max_epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion)
        scheduler.step(va_acc)
        lr = optimizer.param_groups[-1]["lr"]
        epochs_run = epoch
        history.append({"stage": stage, "epoch": epoch, "train_loss": tr_loss,
                        "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc,
                        "head_lr": lr})
        improved = va_acc > best_acc + 1e-6
        if improved:
            best_acc, bad_epochs = va_acc, 0
            best_state = copy.deepcopy(model.state_dict())
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, OUTPUT_DIR / "best_model.pth")
        else:
            bad_epochs += 1
        print(f"  [Stage {stage}] epoch {epoch}/{max_epochs} | train_loss={tr_loss:.4f} "
              f"train_acc={tr_acc:.3f} | val_loss={va_loss:.4f} val_acc={va_acc:.3f}"
              f"{'  <- best' if improved else ''}")
        if bad_epochs >= patience:
            print(f"  [Stage {stage}] early stopping: no improvement for {patience} epochs.")
            stopped_early = True
            break
    if best_state is not None:
        model.load_state_dict(best_state)  # restore best epoch
    return best_acc, best_state, epochs_run, stopped_early


def train_model(model, train_loader, val_loader, class_weights,
                stage1_epochs=STAGE1_MAX_EPOCHS, stage2_epochs=STAGE2_MAX_EPOCHS,
                patience=EARLY_STOP_PATIENCE):
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    model.to(DEVICE)
    history: List[dict] = []
    info = {}

    # ---- Stage 1: head only ----
    print("\n[Stage 1] Training the head only (backbone frozen, BatchNorm in eval mode)...")
    model.freeze_backbone()
    info["stage1_trainable_params"] = model.count_trainable()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", patience=LR_SCHEDULER_PATIENCE, factor=0.5)
    best_acc, best_state, n1, stop1 = fit_stage(
        model, train_loader, val_loader, criterion, opt, sched, 1, stage1_epochs, patience, history)
    info.update(stage1_epochs_run=n1, stage1_early_stopped=stop1, stage1_best_val_acc=best_acc)

    # ---- Stage 2: fine-tune the last blocks ----
    print("\n[Stage 2] Fine-tuning the top backbone blocks...")
    info["unfrozen_block_indices"] = model.unfreeze_top_blocks(N_UNFROZEN_BLOCKS)
    info["stage2_trainable_params"] = model.count_trainable()
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": backbone_params, "lr": 1e-5},
                             {"params": model.fc.parameters(), "lr": 1e-4}],
                            weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", patience=LR_SCHEDULER_PATIENCE, factor=0.5)
    # Stage 2 starts from the best Stage 1 weights and only replaces them if it does better.
    best_acc, best_state, n2, stop2 = fit_stage(
        model, train_loader, val_loader, criterion, opt, sched, 2, stage2_epochs, patience,
        history, best_acc=best_acc, best_state=best_state)
    info.update(stage2_epochs_run=n2, stage2_early_stopped=stop2, best_val_acc=best_acc)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(OUTPUT_DIR / "training_history.csv", index=False)
    return model, history, info


# ==============================================================================
# 6. INFERENCE, EVALUATION, LATENCY, EXPORT
# ==============================================================================
def _infer(model, image_path: str):
    """Full single-image pipeline: load -> preprocess -> model -> softmax."""
    img = load_and_preprocess_cv2(image_path)
    tensor = eval_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)
    conf, idx = torch.max(probs, dim=1)
    return idx.item(), conf.item() * 100


def predict_image(model, image_path: str):
    """Prints  '<label>: <confidence>%'  and flags low-confidence results for
    farmer verification (no spraying decision is made from the AI result alone)."""
    model.eval()
    idx, confidence_pct = _infer(model, image_path)
    label = display_label(CLASS_NAMES[idx])
    needs_verification = confidence_pct < CONFIDENCE_THRESHOLD
    print(f"{label}: {confidence_pct:.0f}%   (full class: {CLASS_NAMES[idx]})")
    if needs_verification:
        print(f"  [LOW CONFIDENCE - below {CONFIDENCE_THRESHOLD:.0f}%] "
              f"Flagged for farmer verification before any spraying action.")
    return label, confidence_pct, needs_verification


def evaluate_test_set(model, test_loader):
    """Returns (true labels, predicted labels, confidences in %) as flat lists.
    Labels are indices into CLASS_NAMES (crop-specific), not display strings."""
    model.eval()
    true_labels, pred_labels, confs = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            probs = torch.softmax(model(images.to(DEVICE)), dim=1)
            conf, pred = torch.max(probs, dim=1)
            true_labels.extend(labels.tolist())
            pred_labels.extend(pred.cpu().tolist())
            confs.extend((conf.cpu() * 100).tolist())
    return true_labels, pred_labels, confs


def measure_latency(model, sample_paths: List[str], n: int = LATENCY_SAMPLE_SIZE) -> dict:
    """Seconds per image for the whole pipeline (read + preprocess + inference)."""
    model.eval()
    paths = sample_paths[:n]
    for p in paths[:3]:  # warm-up
        _infer(model, p)
    times = []
    for p in paths:
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _infer(model, p)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return {"n_images": len(times), "mean_s": float(np.mean(times)),
            "p95_s": float(np.percentile(times, 95)), "max_s": float(np.max(times)),
            "device": str(DEVICE)}


def export_for_deployment(model, export_path: Path = None):
    """TorchScript export (CPU). Checks the exported model gives the same output."""
    export_path = export_path or (OUTPUT_DIR / "leaf_classifier.pt")
    m = copy.deepcopy(model).cpu().eval()
    example = torch.rand(1, 3, IMG_SIZE, IMG_SIZE)
    traced = torch.jit.trace(m, example)
    with torch.no_grad():
        same = torch.allclose(m(example), traced(example), atol=1e-4)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    traced.save(str(export_path))
    print(f"[export] TorchScript model saved -> {export_path} | matches eager model: {same}")
    return export_path


# ==============================================================================
# 7. RESULTS TABLES
# ==============================================================================
def df_to_markdown(df: pd.DataFrame, float_fmt: str = "{:.3f}") -> str:
    def fmt(v):
        return float_fmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(df.columns) + " |",
             "|" + "|".join("---" for _ in df.columns) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def build_results(true_labels, pred_labels, confs, latency, info, dummy: bool):
    tag = "_DUMMY" if dummy else ""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    banner = ("**DUMMY DATA - random noise images. These numbers are meaningless and must "
              "not be reported.**\n\n" if dummy else "")

    rep = classification_report(true_labels, pred_labels, labels=list(range(NUM_CLASSES)),
                                target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    rows = []
    for name in CLASS_NAMES:
        r = rep[name]
        rows.append({"Class": name, "Precision": r["precision"], "Recall": r["recall"],
                     "F1": r["f1-score"], "Support": int(r["support"]),
                     f"Recall >= {TARGET_RECALL:.0%}":
                         "PASS" if r["recall"] >= TARGET_RECALL else "FAIL"})
    accuracy = rep["accuracy"]
    rows.append({"Class": "ACCURACY (overall)", "Precision": "", "Recall": "",
                 "F1": accuracy, "Support": len(true_labels), f"Recall >= {TARGET_RECALL:.0%}": ""})
    for key, nice in (("macro avg", "Macro average"), ("weighted avg", "Weighted average")):
        r = rep[key]
        rows.append({"Class": nice, "Precision": r["precision"], "Recall": r["recall"],
                     "F1": r["f1-score"], "Support": int(r["support"]),
                     f"Recall >= {TARGET_RECALL:.0%}": ""})
    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUTPUT_DIR / f"results_table{tag}.csv", index=False)

    # Confusion matrix (crop-specific labels)
    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(NUM_CLASSES)))
    cm_df = pd.DataFrame(cm, index=[f"true: {c}" for c in CLASS_NAMES],
                         columns=[f"pred: {c}" for c in CLASS_NAMES])
    cm_df.to_csv(OUTPUT_DIR / f"confusion_matrix{tag}.csv")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
        ax.set_xticklabels(CLASS_NAMES, rotation=60, ha="right", fontsize=8)
        ax.set_yticklabels(CLASS_NAMES, fontsize=8)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title("Confusion matrix (test set)" + (" - DUMMY DATA" if dummy else ""))
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                if cm[i, j]:
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                            color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"confusion_matrix{tag}.png", dpi=150)
        plt.close(fig)
    except Exception as e:  # matplotlib is optional
        print(f"[results] confusion-matrix image skipped ({e}); CSV was saved.")

    # Success-criteria check (only the criteria this script can measure)
    macro_recall = rep["macro avg"]["recall"]
    per_class_recall = [rep[c]["recall"] for c in CLASS_NAMES if rep[c]["support"] > 0]
    min_recall = min(per_class_recall) if per_class_recall else 0.0
    worst = CLASS_NAMES[int(np.argmin([rep[c]["recall"] if rep[c]["support"] > 0 else 9
                                       for c in CLASS_NAMES]))]
    crit = pd.DataFrame([
        {"Criterion": "1. Classification accuracy", "Target": f">= {TARGET_ACCURACY:.0%}",
         "Result": f"{accuracy:.1%}", "Status": "PASS" if accuracy >= TARGET_ACCURACY else "FAIL"},
        {"Criterion": "2a. Macro-average recall", "Target": f">= {TARGET_RECALL:.0%}",
         "Result": f"{macro_recall:.1%}", "Status": "PASS" if macro_recall >= TARGET_RECALL else "FAIL"},
        {"Criterion": "2b. Lowest per-class recall", "Target": f">= {TARGET_RECALL:.0%}",
         "Result": f"{min_recall:.1%} ({worst})",
         "Status": "PASS" if min_recall >= TARGET_RECALL else "FAIL"},
        {"Criterion": "4. Prediction time per image (mean)", "Target": f"<= {TARGET_SECONDS_PER_IMAGE:.0f} s",
         "Result": f"{latency['mean_s']:.3f} s on {latency['device']} "
                   f"(p95 {latency['p95_s']:.3f} s, n={latency['n_images']})",
         "Status": "PASS" if latency["mean_s"] <= TARGET_SECONDS_PER_IMAGE else "FAIL"},
    ])
    crit.to_csv(OUTPUT_DIR / f"success_criteria{tag}.csv", index=False)

    # Confidence-threshold analysis (supports the 70% starting threshold)
    conf = np.array(confs); correct = np.array(true_labels) == np.array(pred_labels)
    low = conf < CONFIDENCE_THRESHOLD
    def acc_of(mask):
        return f"{correct[mask].mean():.1%}" if mask.any() else "n/a"
    conf_df = pd.DataFrame([
        {"Group": f"Confidence >= {CONFIDENCE_THRESHOLD:.0f}% (shown as final)",
         "Images": int((~low).sum()), "Share": f"{(~low).mean():.1%}", "Accuracy": acc_of(~low)},
        {"Group": f"Confidence < {CONFIDENCE_THRESHOLD:.0f}% (flagged for farmer check)",
         "Images": int(low.sum()), "Share": f"{low.mean():.1%}", "Accuracy": acc_of(low)},
    ])

    # Training summary
    ts = pd.DataFrame([
        {"Item": "Stage 1 epochs run (max %d)" % STAGE1_MAX_EPOCHS,
         "Value": f"{info.get('stage1_epochs_run')}{' (early stopped)' if info.get('stage1_early_stopped') else ''}"},
        {"Item": "Stage 2 epochs run (max %d)" % STAGE2_MAX_EPOCHS,
         "Value": f"{info.get('stage2_epochs_run')}{' (early stopped)' if info.get('stage2_early_stopped') else ''}"},
        {"Item": "Best validation accuracy", "Value": f"{info.get('best_val_acc', float('nan')):.3f}"},
        {"Item": "Trainable parameters, Stage 1", "Value": f"{info.get('stage1_trainable_params'):,}"},
        {"Item": "Trainable parameters, Stage 2", "Value": f"{info.get('stage2_trainable_params'):,}"},
        {"Item": "Unfrozen inverted-residual block indices", "Value": str(info.get("unfrozen_block_indices"))},
    ])

    md = [f"# Results summary\n\n{banner}", "## Per-class results (test set)\n",
          df_to_markdown(results_df), "\n## Success criteria check\n",
          df_to_markdown(crit),
          "\n*Criteria 3 (early detection), 5, 6, 7, 8 and 9 are not measured by this script.*\n",
          "\n## Confidence threshold analysis\n", df_to_markdown(conf_df),
          "\n## Training summary\n", df_to_markdown(ts)]
    (OUTPUT_DIR / f"results_summary{tag}.md").write_text("\n".join(md), encoding="utf-8")
    (OUTPUT_DIR / f"results_table{tag}.md").write_text(
        banner + df_to_markdown(results_df), encoding="utf-8")

    print("\n" + "=" * 70 + "\nRESULTS TABLE (test set)\n" + "=" * 70)
    print(results_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nSUCCESS CRITERIA CHECK\n" + crit.to_string(index=False))
    print("\nCONFIDENCE THRESHOLD ANALYSIS\n" + conf_df.to_string(index=False))
    if dummy:
        print("\n*** DUMMY DATA: ignore these numbers. ***")
    return results_df, crit


# ==============================================================================
# 8. DUMMY DATA (smoke test only)
# ==============================================================================
def generate_dummy_dataset(root: Path, images_per_class: int = 12):
    """Random-noise images in <root>/train/<Crop>/<class>/ using underscore folder names
    (like the real dataset on disk). Only proves the pipeline runs."""
    if root.exists():
        shutil.rmtree(root)
    maize = ["fall_armyworm", "grasshoper", "healthy", "leaf_beetle", "leaf_blight",
             "leaf_spot", "streak_virus"]
    tomato = ["healthy", "leaf_blight", "leaf_curl", "septoria_leaf_spot", "verticulium_wilt"]
    for crop, folders in (("Maize", maize), ("Tomato", tomato)):
        for name in folders:
            d = root / "train" / crop / name
            d.mkdir(parents=True)
            for i in range(images_per_class):
                noise = np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
                Image.fromarray(noise).save(d / f"dummy_{i}.jpg")
    print(f"[dummy data] {images_per_class} noise images per class created under {root}")


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    dataset_root, batch_size = DATASET_ROOT, BATCH_SIZE
    s1_epochs, s2_epochs, patience = STAGE1_MAX_EPOCHS, STAGE2_MAX_EPOCHS, EARLY_STOP_PATIENCE
    if USE_DUMMY_DATA:
        print("=" * 70 + "\nUSE_DUMMY_DATA = True: random-noise smoke test. "
              "Results are NOT real.\n" + "=" * 70)
        dummy_root = OUTPUT_DIR.parent / "dummy_data"
        generate_dummy_dataset(dummy_root)
        dataset_root, batch_size = str(dummy_root), 8
        s1_epochs, s2_epochs, patience = 2, 2, 5

    print(f"Device: {DEVICE}")
    if DEVICE.type == "cpu" and not USE_DUMMY_DATA:
        print("[WARNING] No GPU found. Training on ~11,000 images on CPU can take hours. "
              "Use Google Colab (GPU) if possible.")

    train, val, test = build_splits(dataset_root)
    save_split_lists(train, val, test)
    print_split_table(train, val, test)

    train_loader = DataLoader(LeafDiseaseDataset(train, train=True), batch_size=batch_size,
                              shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=DEVICE.type == "cuda")
    val_loader = DataLoader(LeafDiseaseDataset(val, train=False), batch_size=batch_size,
                            shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(LeafDiseaseDataset(test, train=False), batch_size=batch_size,
                             shuffle=False, num_workers=NUM_WORKERS)

    class_weights = compute_loss_weights([y for _, y in train])
    print("\nClass weights (inverse frequency):")
    for name, w in zip(CLASS_NAMES, class_weights.tolist()):
        print(f"  {name:<22} {w:.3f}")

    model = MobileNetV3LeafClassifier(num_classes=NUM_CLASSES, dropout=DROPOUT,
                                      pretrained=PRETRAINED)
    model, history, info = train_model(model, train_loader, val_loader, class_weights,
                                       s1_epochs, s2_epochs, patience)

    # ---- Full test-set evaluation (touched once, after training) ----
    print("\n[Evaluation] Evaluating the best model on the held-out test set...")
    true_labels, pred_labels, confs = evaluate_test_set(model, test_loader)
    print(f"Evaluated {len(true_labels)} test images.")

    # ---- Inference demo on real test images ----
    print("\n[Inference demo] Predictions on 3 test images:")
    for path, true_idx in test[:3]:
        print(f"  true class: {CLASS_NAMES[true_idx]}")
        predict_image(model, path)

    latency = measure_latency(model, [p for p, _ in test])
    print(f"\n[Latency] mean {latency['mean_s']:.3f} s, p95 {latency['p95_s']:.3f} s per image "
          f"on {latency['device']}")

    build_results(true_labels, pred_labels, confs, latency, info, dummy=USE_DUMMY_DATA)
    export_for_deployment(model)
    print(f"\nAll outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
