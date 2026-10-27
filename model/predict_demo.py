"""
predict_demo.py -- live prediction on a real leaf image with the exported model.

Shows the trained classifier on a real CCMT image from the held-out TEST split
(never seen during training or validation), in the farmer-facing format that
predict_image() prints: "<label>: <confidence>%", plus the low-confidence
warning when it applies. The spraying demo on the dashboard and poster stays
simulated; this is how the real model is shown.

Run from the repo root:
    python model/predict_demo.py                      a random test image
    python model/predict_demo.py --class "Fall armyworm"
    python model/predict_demo.py --image "data/Raw Data/Maize/leaf spot/leaf spot12_.jpg"
"""
import argparse
import random
import sys
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_deploy_demo as t  # noqa: E402  (same preprocessing, labels and threshold as training)


def test_split():
    """Rebuild the training run's exact test split (same seed, same stratified 70/15/15)."""
    samples = t.build_sample_list(t.DATASET_ROOT)
    paths, labels = [s[0] for s in samples], [s[1] for s in samples]
    _, temp_p, _, temp_l = train_test_split(paths, labels, test_size=0.30, stratify=labels,
                                            random_state=t.RANDOM_SEED)
    _, test_p, _, test_l = train_test_split(temp_p, temp_l, test_size=0.50, stratify=temp_l,
                                            random_state=t.RANDOM_SEED)
    return list(zip(test_p, test_l))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="leaf_classifier.pt", help="TorchScript export from train_deploy_demo.py")
    ap.add_argument("--image", help="a specific image instead of a random test image")
    ap.add_argument("--class", dest="cls", choices=t.CLASS_NAMES, help="pick a random test image of this class")
    ap.add_argument("--seed", type=int, default=None, help="repeat the same random pick")
    a = ap.parse_args()

    if not Path(a.model).exists():
        sys.exit(f"{a.model} not found. Run model/train_deploy_demo.py first (from the repo root).")
    model = torch.jit.load(a.model, map_location=t.DEVICE)

    truth = None
    if a.image:
        path = a.image
    else:
        pool = test_split()
        if a.cls:
            pool = [s for s in pool if t.CLASS_NAMES[s[1]] == a.cls]
        path, idx = random.Random(a.seed).choice(pool)
        truth = t.CLASS_NAMES[idx]

    print(f"Image: {path}")
    if truth:
        print(f"True class (from the dataset folder): {truth}  -- a held-out test image\n")
    print("Model output:")
    t.predict_image(model, path)

    # Top 3, so the audience can see how sure the model is and what it nearly said.
    with torch.no_grad():
        x = t.eval_transform(t.load_and_preprocess_cv2(path)).unsqueeze(0).to(t.DEVICE)
        probs = torch.softmax(model(x), dim=1)[0]
    top = torch.topk(probs, 3)
    print("\nTop 3:")
    for p, i in zip(top.values.tolist(), top.indices.tolist()):
        print(f"  {t.CLASS_NAMES[i]:<22} {p * 100:5.1f}%")


if __name__ == "__main__":
    main()
