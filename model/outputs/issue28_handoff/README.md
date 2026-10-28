# Issue #28 hand-off: model evaluation results

Test-set results of the **deployed** leaf classifier (`leaf_classifier.pt`, the tuned run in `model/outputs/tuned/`), for the evaluation write-up.

| File | Contents |
|---|---|
| `test_predictions.json` | `class_names` (the 12 `CLASS_NAMES`, in label order), `true_labels` and `pred_labels` (1,661 integers each, one per test image) |
| `classification_report.txt` | Per-class precision, recall, F1 and support (sklearn) |
| `confusion_matrix.png`, `confusion_matrix.csv` | 12 × 12 confusion matrix; rows are the true class, columns the predicted class |
| `metrics.json` | Accuracy, macro precision, recall and F1, and the per-class numbers |
| `training_curves.png` | Training and validation loss and accuracy per epoch |

## Headline numbers (held-out test set, 1,661 images)

| Metric | Target | Result | Baseline run |
|---|---|---|---|
| Accuracy | ≥ 90% | 84.3% | 75.2% |
| Macro recall | ≥ 92% | 85.3% | 79.8% |
| Macro precision | ≥ 87% | 83.8% | 74.1% |
| Macro F1 | ≥ 89.5% | 84.3% | 76.2% |

The targets are not yet met. The baseline's files are in `model/outputs/baseline/`.

## Recomputing the metrics

```python
import json
from sklearn.metrics import classification_report, confusion_matrix

d = json.load(open("model/outputs/issue28_handoff/test_predictions.json"))
print(classification_report(d["true_labels"], d["pred_labels"],
                            labels=list(range(12)), target_names=d["class_names"], digits=3))
print(confusion_matrix(d["true_labels"], d["pred_labels"], labels=list(range(12))))
```

## Points worth reporting

* No maize leaf was predicted as a tomato class, or the reverse: every error is within one crop.
* The main confusions are between look-alike lesions: maize leaf spot and maize leaf blight (39 and 25 images), tomato Septoria leaf spot and tomato leaf blight (24 and 22), and Verticillium wilt mistaken for Septoria (23).
* The weakest class is Verticillium wilt (recall 64.7%). The strongest are grasshopper (98.0%), healthy tomato (97.1%) and leaf beetle (95.7%).
* Results at 70% confidence or more cover 86.6% of test images and are 89.4% right. The other 13.4% are flagged for the farmer to check.

Full discussion: `docs/LeafSight_Project_Report.md`, section 5.3.
