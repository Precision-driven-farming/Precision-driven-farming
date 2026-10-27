"""
spraying_pipeline.py
====================
End-to-end pipeline: drone/pole image batch -> CNN classification ->
severity scoring -> zone aggregation -> resource optimization ->
farmer verification gate -> structured spraying map.

Sits on top of Member 6's exported TorchScript model (leaf_classifier.pt).
It does NOT retrain or replace the classifier.

Legend used in the comments below (same idea as Member 6's dummy-data flags):
    [REAL]        = real logic that will be kept in the final system
    [PLACEHOLDER] = simulated data/behaviour, replace when hardware exists

Run from the repo root (where leaf_classifier.pt is exported):
      python app/spraying_pipeline.py
Needs: numpy, opencv-python  (+ torch if leaf_classifier.pt is available)
"""

import json
import math
import os
import random
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np

# =============================================================================
# 0. CONFIGURATION  [REAL]  (all thresholds are starting points, to be tuned)
# =============================================================================
MODEL_PATH = "leaf_classifier.pt"      # Member 6's TorchScript export
IMG_SIZE = 224                         # must match Member 6's IMG_SIZE
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CONFIDENCE_THRESHOLD = 70.0            # percent, same as Member 6's B3

# Must stay in sync with CLASS_NAMES in Member 6's train_deploy_demo.py
CLASS_NAMES = [
    "Fall armyworm", "Grasshopper", "Healthy (Maize)", "Leaf beetle",
    "Leaf blight (Maize)", "Leaf spot", "Streak virus",
    "Healthy (Tomato)", "Leaf blight (Tomato)", "Leaf curl",
    "Septoria leaf spot", "Verticillium wilt",
]

# Base severity weight per class (0 = harmless, 1 = most destructive).
# ASSUMPTION: these are placeholder expert guesses. An agronomist should
# confirm them before real use.
CLASS_SEVERITY_WEIGHT = {
    "Healthy": 0.00,
    "Leaf spot": 0.40,
    "Leaf beetle": 0.40,
    "Grasshopper": 0.50,
    "Septoria leaf spot": 0.60,
    "Leaf blight": 0.70,
    "Leaf curl": 0.70,
    "Fall armyworm": 0.90,
    "Streak virus": 0.90,
    "Verticillium wilt": 0.90,
}

CELL_SIZE_M = 20.0            # zone = 20 m x 20 m grid cell
MIN_DETECTIONS_PER_ZONE = 3   # fewer than this = weak evidence -> verify
MEAN_WEIGHT, MAX_WEIGHT = 0.7, 0.3   # zone severity blend (see aggregate)

SPRAY_THRESHOLD = 30.0        # zone severity >= this is a spray candidate
HIGH_THRESHOLD = 60.0         # zone severity >= this is high priority
MONITOR_THRESHOLD = 15.0      # between this and SPRAY_THRESHOLD = monitor
MAX_SPRAY_FRACTION = 0.50     # budget: never spray more than 50% of field
DOSE_HIGH_PCT, DOSE_STANDARD_PCT = 100, 60

GPS_ACCURACY_LIMIT_M = 5.0    # worse than this -> farmer must check position
LOW_CONF_ZONE_SHARE = 0.30    # >30% low-confidence detections -> verify
POOR_GPS_ZONE_SHARE = 0.30    # >30% poor GPS fixes in a zone -> verify
UNUSABLE_ZONE_SHARE = 0.30    # >30% unusable images in a zone -> verify

# Input check before classification. A fine-tuned CNN can be confidently wrong
# on input that is not a leaf photo (on random noise the deployed model says
# "Streak virus" at ~89%), so confidence alone cannot catch it. Measured on all
# 11,069 CCMT images: neighbour-pixel correlation is never below 0.525 (noise
# is below 0.2), mean brightness never below 8, contrast never below 2.8.
MIN_PIXEL_CORRELATION = 0.5   # below this: noise / sensor fault, not a photo
MIN_CONTRAST = 2.0            # grey-level std below this: blank frame
MIN_BRIGHTNESS, MAX_BRIGHTNESS = 5.0, 250.0


# =============================================================================
# 1. MODEL LOADING + INFERENCE
# =============================================================================
def display_label(class_name):
    """'Healthy (Maize)' -> 'Healthy' (same as Member 6's display_label)."""
    return class_name.split(" (")[0]


class UnusableImage(ValueError):
    """The image was read but is not fit to classify (see image_quality_issue)."""


def image_quality_issue(img):
    """[REAL] Returns why a (resized, BGR) image is unfit to classify, or None."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if g.std() < MIN_CONTRAST:
        return "blank frame (no contrast)"
    if not MIN_BRIGHTNESS <= g.mean() <= MAX_BRIGHTNESS:
        return "under- or over-exposed"
    corr = np.corrcoef(g[:, :-1].ravel(), g[:, 1:].ravel())[0, 1]
    if corr < MIN_PIXEL_CORRELATION:
        return "not a camera photo (pixel noise)"
    return None


def preprocess(image_path, check_quality=True):
    """[REAL] Same steps as Member 6: validate -> resize -> RGB -> normalise.
    Equivalent to their eval_transform (ToTensor + ImageNet Normalize)."""
    img = cv2.imread(str(image_path))
    if img is None:                       # "validate" step
        raise ValueError(f"Could not read image: {image_path}")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    issue = image_quality_issue(img) if check_quality else None
    if issue:
        raise UnusableImage(issue)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    arr = img.astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(arr, (2, 0, 1))   # HWC -> CHW


class TorchScriptClassifier:
    """[REAL] Loads Member 6's exported model and classifies one image."""

    def __init__(self, path):
        import torch                      # imported here so the script still
        self.torch = torch                # runs (in simulation) without torch
        self.model = torch.jit.load(path, map_location="cpu")
        self.model.eval()

    def predict(self, image_path):
        t = self.torch.from_numpy(preprocess(image_path)).unsqueeze(0)
        with self.torch.no_grad():
            probs = self.torch.softmax(self.model(t), dim=1)
            conf, idx = self.torch.max(probs, dim=1)
        return display_label(CLASS_NAMES[idx.item()]), conf.item() * 100


class SimulatedClassifier:
    """[PLACEHOLDER] Stand-in used only when leaf_classifier.pt or torch is
    missing. Returns pre-decided (label, confidence) pairs so the demo has a
    realistic spatial pattern. Delete once the real model is available."""

    def __init__(self, truth):
        self.truth = truth                # {image_path: (label, conf_pct)}

    def predict(self, image_path):
        # still exercises the validate step; the quality check is skipped
        # because the placeholder images are noise standing in for photos
        preprocess(image_path, check_quality=False)
        return self.truth[image_path]


def load_classifier(sim_truth):
    if os.path.exists(MODEL_PATH):
        try:
            clf = TorchScriptClassifier(MODEL_PATH)
            print(f"[model] Loaded real TorchScript model: {MODEL_PATH}")
            return clf
        except ImportError:
            print("[model] torch not installed -> using SIMULATED classifier")
    else:
        print(f"[model] {MODEL_PATH} not found -> using SIMULATED classifier")
    return SimulatedClassifier(sim_truth)


# =============================================================================
# 2. PLACEHOLDER INPUTS: dummy image batch + simulated GPS/timestamps
# =============================================================================
def simulate_batch(folder, n_images=120, seed=7):
    """[PLACEHOLDER] Creates dummy images and fake GPS/timestamps.
    In production, GPS + timestamp come from the drone/pole metadata.
    The disease 'hotspot' in the north-east is invented for the demo."""
    rng = random.Random(seed)
    origin_lat, origin_lon = -26.6700, 27.9300       # simulated field corner
    field_m = 60.0                                   # 60 m x 60 m field
    start = datetime(2026, 10, 8, 6, 0, tzinfo=timezone.utc)
    batch, truth = [], {}

    for i in range(n_images):
        x, y = rng.uniform(0, field_m), rng.uniform(0, field_m)   # metres
        lat = origin_lat + y / 111_320.0
        lon = origin_lon + x / (111_320.0 * math.cos(math.radians(origin_lat)))

        in_hotspot = x > 40 and y > 40               # north-east corner
        in_mild = 20 < x <= 40 and y > 40            # mild patch next to it
        if in_hotspot:
            label = rng.choice(["Fall armyworm", "Leaf blight", "Streak virus"])
            conf = rng.uniform(75, 98)
        elif in_mild:
            label = rng.choice(["Leaf spot", "Leaf spot", "Leaf curl", "Healthy"])
            conf = rng.uniform(70, 92)
        else:
            label = "Healthy" if rng.random() < 0.9 else "Leaf spot"
            conf = rng.uniform(70, 99)
        if rng.random() < 0.06:                      # occasional unsure result
            conf = rng.uniform(40, 69)

        path = os.path.join(folder, f"img_{i:03d}.jpg")
        cv2.imwrite(path, np.random.randint(0, 255, (256, 256, 3), np.uint8))
        truth[path] = (label, conf)
        batch.append({
            "image_path": path,
            "latitude": lat,
            "longitude": lon,
            # fake GPS quality: good almost everywhere, poor in the south-east
            # corner so the verification flag has something to catch
            "gps_accuracy_m": 8.0 if (x > 40 and y < 20) else rng.choice([1.0, 1.5, 2.0, 3.0]),
            "timestamp": (start + timedelta(seconds=10 * i)).isoformat(),
        })
    return batch, truth, (origin_lat, origin_lon)


# =============================================================================
# 3. PART A LOGIC  [REAL]
# =============================================================================
def severity_score(label, confidence_pct):
    """A1. One detection -> severity 0-100.
    severity = class weight x model confidence. Healthy is always 0, and a
    shaky prediction counts for less than a confident one."""
    weight = CLASS_SEVERITY_WEIGHT.get(label, 0.5)   # unknown class: middle
    return weight * confidence_pct               # (conf is already 0-100)


def to_local_metres(lat, lon, origin):
    """Converts GPS to metres east/north of the field origin (flat-earth
    approximation, fine for a single farm)."""
    o_lat, o_lon = origin
    y = (lat - o_lat) * 111_320.0
    x = (lon - o_lon) * 111_320.0 * math.cos(math.radians(o_lat))
    return x, y


def aggregate_zones(detections, origin):
    """A2. Groups detections into grid zones and scores each zone.
    zone severity = 0.7 x mean + 0.3 x max
      - the mean shows how widespread the problem is in the zone
      - the max stops one severe hotspot being averaged away by healthy leaves
    """
    cells = defaultdict(list)
    for d in detections:
        x, y = to_local_metres(d["latitude"], d["longitude"], origin)
        col, row = int(x // CELL_SIZE_M), int(y // CELL_SIZE_M)
        cells[(row, col)].append(d)

    zones = []
    for (row, col), images in sorted(cells.items()):
        # unusable images count towards the zone's coverage but not its score
        items = [d for d in images if d["label"] is not None]
        sev = [d["severity"] for d in items] or [0.0]
        zone_sev = MEAN_WEIGHT * (sum(sev) / len(sev)) + MAX_WEIGHT * max(sev)
        diseased = [d["label"] for d in items if d["label"] != "Healthy"]
        # zone centre converted back to GPS so the farmer can check placement
        cx, cy = (col + 0.5) * CELL_SIZE_M, (row + 0.5) * CELL_SIZE_M
        c_lat = origin[0] + cy / 111_320.0
        c_lon = origin[1] + cx / (111_320.0 * math.cos(math.radians(origin[0])))
        zones.append({
            "zone_id": f"R{row}C{col}",
            "row": row, "col": col,
            "centre_latitude": round(c_lat, 6),
            "centre_longitude": round(c_lon, 6),
            "severity": round(zone_sev, 1),
            "n_detections": len(items),
            "n_unusable": len(images) - len(items),
            "dominant_class": Counter(diseased).most_common(1)[0][0]
                              if diseased else ("Healthy" if items else "Unknown"),
            "low_conf_share": sum(d["low_confidence"] for d in items) / max(1, len(items)),
            "unusable_share": (len(images) - len(items)) / len(images),
            "poor_gps_share": sum(d["gps_accuracy_m"] > GPS_ACCURACY_LIMIT_M
                                  for d in images) / len(images),
            "bounds_m": {"x_min": col * CELL_SIZE_M, "x_max": (col + 1) * CELL_SIZE_M,
                         "y_min": row * CELL_SIZE_M, "y_max": (row + 1) * CELL_SIZE_M},
        })
    return zones


def decide_actions(zones):
    """A3. Resource optimization: spray ONLY zones above the threshold,
    worst first, and never more than MAX_SPRAY_FRACTION of the field.
    Dose is scaled to severity so mild zones get less chemical."""
    budget = max(1, math.floor(MAX_SPRAY_FRACTION * len(zones)))
    candidates = sorted([z for z in zones if z["severity"] >= SPRAY_THRESHOLD],
                        key=lambda z: z["severity"], reverse=True)
    funded = {z["zone_id"] for z in candidates[:budget]}

    for z in zones:
        s = z["severity"]
        if z["n_detections"] == 0:
            z["recommended_action"] = "RE-SCAN - no usable images"
            z["recommended_dose_pct"] = 0
        elif z["zone_id"] in funded:
            high = s >= HIGH_THRESHOLD
            z["recommended_action"] = "SPRAY - HIGH PRIORITY" if high else "SPRAY - STANDARD"
            z["recommended_dose_pct"] = DOSE_HIGH_PCT if high else DOSE_STANDARD_PCT
        elif s >= SPRAY_THRESHOLD:
            z["recommended_action"] = "SPRAY DEFERRED (over budget) - re-scan"
            z["recommended_dose_pct"] = 0
        elif s >= MONITOR_THRESHOLD:
            z["recommended_action"] = "MONITOR - re-scan in 2-3 days"
            z["recommended_dose_pct"] = 0
        else:
            z["recommended_action"] = "NO ACTION"
            z["recommended_dose_pct"] = 0
    return zones


def flag_for_verification(zones):
    """A4. Risk register: 'incorrect spraying map'. Every zone is checked for
    reasons the map might be wrong. Zones with reasons need extra scrutiny.
    NOTE: even zones with NO reasons still need farmer approval (see gate)."""
    for z in zones:
        reasons = []
        if z["poor_gps_share"] > POOR_GPS_ZONE_SHARE:
            reasons.append(f"{z['poor_gps_share']:.0%} of GPS fixes worse than "
                           f"{GPS_ACCURACY_LIMIT_M} m: confirm zone position/boundary")
        if z["n_detections"] < MIN_DETECTIONS_PER_ZONE:
            reasons.append(f"Only {z['n_detections']} image(s): too little evidence")
        if z["unusable_share"] > UNUSABLE_ZONE_SHARE:
            reasons.append(f"{z['unusable_share']:.0%} of images unusable "
                           f"(noise, blank or badly exposed): re-capture")
        if z["low_conf_share"] > LOW_CONF_ZONE_SHARE:
            reasons.append(f"{z['low_conf_share']:.0%} of results below "
                           f"{CONFIDENCE_THRESHOLD:.0f}% confidence: review images")
        z["needs_verification"] = bool(reasons)
        z["verification_reasons"] = reasons
        z["status"] = "AWAITING_FARMER_REVIEW"     # nothing is ever auto-approved
    return zones


def release_for_spraying(zones, farmer_decisions):
    """A4 gate: the ONLY way a zone becomes part of a spray plan.
    farmer_decisions = {zone_id: "approve" | "reject"}.
    A zone with a spray action but no explicit 'approve' is NOT released."""
    plan = []
    for z in zones:
        if not z["recommended_action"].startswith("SPRAY -"):
            continue
        if farmer_decisions.get(z["zone_id"]) == "approve":
            z["status"] = "APPROVED_BY_FARMER"
            plan.append(z)
        else:
            z["status"] = "HELD_NOT_APPROVED"
    return plan


# =============================================================================
# 4. PIPELINE
# =============================================================================
def run_pipeline():
    with tempfile.TemporaryDirectory() as folder:
        batch, sim_truth, origin = simulate_batch(folder)     # [PLACEHOLDER]
        classifier = load_classifier(sim_truth)

        detections = []
        for item in batch:                                    # [REAL] loop
            try:
                label, conf = classifier.predict(item["image_path"])
            except UnusableImage as err:                       # read, but unfit
                detections.append({**item, "label": None, "confidence_pct": None,
                                    "low_confidence": False, "severity": None,
                                    "quality_issue": str(err)})
                continue
            except ValueError as err:                          # corrupt image
                print(f"[skip] {err}")
                continue
            detections.append({
                **item,
                "label": label,
                "confidence_pct": round(conf, 1),
                "low_confidence": conf < CONFIDENCE_THRESHOLD,
                "severity": severity_score(label, conf),
            })

    zones = aggregate_zones(detections, origin)
    zones = decide_actions(zones)
    zones = flag_for_verification(zones)
    return detections, zones


def summarise(zones, plan):
    sprayed = [z for z in zones if z["recommended_action"].startswith("SPRAY -")]
    # a zone with no usable images was not assessed, so it is not a "saving"
    assessed = [z for z in zones if z["n_detections"] > 0]
    saving = round(100 * (1 - len(sprayed) / len(assessed)), 1) if assessed else None
    return {
        "zones_total": len(zones),
        "zones_needing_rescan": len(zones) - len(assessed),
        "zones_recommended_for_spray": len(sprayed),
        "zones_needing_extra_verification": sum(z["needs_verification"] for z in zones),
        "zones_released_after_farmer_approval": len(plan),
        "estimated_chemical_saving_vs_blanket_spray_pct": saving,
    }


if __name__ == "__main__":
    detections, zones = run_pipeline()

    # ---- Final structured spraying map (what the dashboard would show) ----
    spray_map = [{
        "zone_id": z["zone_id"],
        "severity": z["severity"],
        "recommended_action": z["recommended_action"],
        "recommended_dose_pct": z["recommended_dose_pct"],
        "needs_verification": z["needs_verification"],
        "verification_reasons": z["verification_reasons"],
        "dominant_class": z["dominant_class"],
        "n_detections": z["n_detections"],
        "n_unusable": z["n_unusable"],
        "centre_latitude": z["centre_latitude"],
        "centre_longitude": z["centre_longitude"],
        "bounds_m": z["bounds_m"],
        "status": z["status"],
    } for z in zones]

    # ---- [PLACEHOLDER] simulated farmer: approves every spray zone that has
    # no verification reasons and rejects the rest. In the real system this
    # is the farmer clicking approve/reject on the dashboard map.
    decisions = {z["zone_id"]: ("reject" if z["needs_verification"] else "approve")
                 for z in zones}
    plan = release_for_spraying(zones, decisions)

    result = {"spraying_map": spray_map, "summary": summarise(zones, plan)}
    # refresh status values after the gate ran
    for entry, z in zip(result["spraying_map"], zones):
        entry["status"] = z["status"]

    print(json.dumps(result, indent=2))
    with open("spraying_map.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nSaved -> spraying_map.json")
