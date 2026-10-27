# Training, Hyperparameter Tuning & Deployment Logic
**Issues:** #26 (Training & Hyperparameter Tuning), #27 (Deployment Logic)
**Author:** Member 6

---

## Part A — Training & Hyperparameter Tuning

### A1. Training Strategy

The model was trained with **two-stage transfer learning** on the MobileNetV3-Large backbone (pre-trained on ImageNet), matching the architecture confirmed in Member 5's document:

- **Stage 1 — Head training:** The backbone was frozen entirely, and only the new classification head (Dropout → Fully Connected) was trained, using a learning rate of `1e-3`. This let the randomly-initialised head learn to interpret the pre-trained backbone's features before any backbone weights are disturbed.
- **Stage 2 — Fine-tuning with discriminative learning rates.** The plan was to unfreeze only the last 3 of the backbone's 17 blocks, at `1e-5` for the backbone and `1e-4` for the head, keeping the early, general-purpose features (edges, textures) fixed. That run is the **baseline**. The tuning experiment (A2, A5) showed that unfreezing **all 17 blocks** at `1e-4` (backbone) and `1e-3` (head) is much better on this data, so that is the **deployed** recipe and the script's default. The backbone still learns ten times more slowly than the head, so the pre-trained features are adapted rather than overwritten.

**Loss function:** `CrossEntropyLoss`, weighted by inverse class frequency (see A3). Raw logits were used during training — Softmax is applied only at inference time — because `CrossEntropyLoss` applies log-softmax internally, and applying Softmax twice would distort gradient scaling.

**Optimizer:** `AdamW`, chosen over plain Adam for its decoupled weight decay (`1e-4`), which reduces overfitting risk — relevant given the dataset's moderate size (~11k raw images across 12 classes).

**Learning rate schedule:** `ReduceLROnPlateau`, monitoring validation accuracy, halving the learning rate after 2 epochs without improvement. This was chosen over a fixed schedule because the real dataset's convergence behaviour was unknown. In both real runs validation accuracy kept improving often enough that the rate never had to be halved.

### A2. Hyperparameter Tuning

Given the scope of a student project (limited compute time and no GPU), we used a **manual/guided search** rather than an automated method like grid search or Bayesian optimization, which would require many full training runs to be worthwhile.

Parameters considered, in priority order (the experiment that was run changed items 2 and 3; results in A5):
1. **Dropout rate** (starting at 0.3) — adjusted based on the gap between training and validation accuracy (a widening gap signals overfitting and a need for higher dropout).
2. **Number of unfrozen backbone blocks** (starting at 3) — increased if Stage 2 fine-tuning underperforms, decreased if it overfits quickly.
3. **Learning rates** for Stage 1 and Stage 2 — adjusted based on loss curve behaviour (oscillating loss suggests the rate is too high; flat loss suggests too low).
4. **Batch size** — kept small (32, dropping to 16 if memory runs out) for this project given expected hardware constraints (no dedicated GPU cluster), acknowledging this is a trade-off against training stability.

Each stage runs up to its epoch limit (10 for Stage 1, 12 for Stage 2) and ends early after 4 epochs without a new best validation accuracy. After every epoch the weights are saved if validation accuracy is the best so far, Stage 2 starts from the best Stage 1 weights, and the model exported for deployment is the best checkpoint rather than the final epoch.

### A3. Class Imbalance Handling

Team 1's dataset documentation shows meaningful imbalance in both crops — for example, the Healthy class is the smallest in both maize and tomato, while Septoria leaf spot makes up roughly 43–47% of the tomato subset.

We addressed this with **class-weighted loss**: weights are computed as inversely proportional to class frequency (`sklearn.utils.class_weight.compute_class_weight`, `balanced` mode), so the loss function penalises mistakes on minority classes more heavily than mistakes on majority classes. This was chosen over oversampling/undersampling because it doesn't discard data or risk overfitting to duplicated minority-class images, and doesn't require re-generating the (already large) augmented dataset.

Per Team 1's dataset documentation, accuracy alone was not used to judge performance — precision, recall, and F1 per class (Member 7's task) are needed specifically because they reveal poor performance on minority classes that overall accuracy would hide.

### A4. Train/Val/Test Split

We used a **stratified 70/15/15 split** (train/validation/test), stratified by class label so that each split preserves the same class proportions as the full dataset — important given the imbalance noted above. A held-out test set (separate from validation, which is used only during training to monitor performance and tune hyperparameters) ensures the final accuracy metrics Member 7 reports reflect genuinely unseen data.

### A5. Training Runs and Results

Both runs used the real CCMT dataset (11,069 images; 7,748 train, 1,660 validation, 1,661 test), batch size 32, dropout 0.3, AdamW with weight decay `1e-4`, seed 42, on a laptop CPU (Intel Core i3-1115G4, 4 threads, 7.7 GB RAM, no GPU). Google Colab was unavailable, so everything ran locally.

| | Baseline (original plan) | Tuned (deployed) |
|---|---|---|
| Stage 1 | 10 epochs, 59 min, best val 72.9% | shared: started from the same Stage 1 weights |
| Stage 2 blocks unfrozen; learning rates | top 3; `1e-5` / `1e-4` | all 17; `1e-4` / `1e-3` |
| Stage 2 | 12 epochs, 66 min; best val **76.1%** at epoch 12 | 12 epochs, 124 min; best val **83.1%** at epoch 11 |
| Early stopping | not triggered (still improving) | not triggered; best checkpoint (epoch 11) exported |
| Test accuracy | 75.2% | **84.3%** |
| Test macro recall / precision / F1 | 79.8% / 74.1% / 76.2% | **85.3% / 83.8% / 84.3%** |

The tuned model was chosen on validation accuracy; the test set was scored once per model. The baseline ended with training accuracy (75.2%) below its validation accuracy, so it was underfitting: three blocks at `1e-5` gave the network too little room to adapt to leaf images. The tuned run ended at 92.7% training against 82.7% validation accuracy, a mild overfit that the best-checkpoint rule absorbs. Dropout and batch size were not changed, because each experiment costs two to three hours on this CPU.

Tuning helped most on the tomato lesion classes (tomato leaf blight recall 51.8% → 78.2%, Verticillium wilt F1 52.0% → 70.8%). Fall armyworm recall fell from 97.6% to 88.1% while its precision rose from 75.9% to 88.1%. In both models no maize leaf was ever predicted as a tomato class or the reverse; the remaining errors are between look-alike lesions (maize leaf spot and maize leaf blight, Septoria and tomato leaf blight, Verticillium wilt mistaken for Septoria). Full per-class results, the confusion matrix and the training curves are in `model/outputs/tuned/` (deployed) and `model/outputs/baseline/`.

**Next experiments:** a larger input size (fine lesion detail is lost at 224 pixels), training on a GPU for more epochs, stronger augmentation, per-class confidence thresholds.

---

## Part B — Deployment Logic

### B1. Deployment Flow

The end-to-end flow from image capture to spraying-map input is:

```
Camera/drone image capture
        ↓
OpenCV preprocessing (validate → resize 224×224 → RGB convert)
        ↓
Torchvision transform (normalize → tensor)
        ↓
PyTorch model inference (MobileNetV3-Large + custom head)
        ↓
Softmax → predicted class + confidence score
        ↓
Confidence threshold check
        ↓
Result passed to precision-spraying-map logic (GPS + severity, handled by the wider system)
```

OpenCV is used for the validate/resize/RGB-convert steps specifically because Team 1's functional requirements specify OpenCV for image processing; PyTorch handles everything from tensor conversion onward, matching the requirement that the AI model itself be built in PyTorch.

### B2. Export Format

The trained model is exported using **TorchScript** (`torch.jit.trace`), not ONNX. This is because the current deployment target is assumed to be a Python/PyTorch-capable environment (a drone companion computer or field laptop running the same stack used for training/development), so no cross-framework conversion is needed. TorchScript also allows the model to run without the original Python class definition present, which is useful for a standalone deployment script.

If a future version needs to target a non-PyTorch runtime (e.g. a mobile app using TensorFlow Lite, or a microcontroller), ONNX export would need to be added — this is noted as a Future Improvement rather than a current requirement, since no such hardware constraint has been specified yet.

### B3. Confidence Threshold & Human Verification

Per Team 1's requirement that farmers must be able to check results before making a spraying decision, predictions below a **70% confidence threshold** are flagged for human verification rather than treated as final. This threshold is a starting point, not a tuned value. On the test set it behaves as follows: the deployed model is at least 70% confident on 86.6% of images and is right on 89.4% of those; the remaining 13.4% are flagged, and they contain 41.8% of all its errors (the baseline flagged 40.7% of images). The next step is a per-class threshold chosen from the validation set, starting with fall armyworm.

Confidence cannot catch input that is not a leaf: on random-noise images the model answers "Streak virus" at about 89%. The spraying pipeline therefore rejects blank, badly exposed and noise images before classification (`image_quality_issue()` in `app/spraying_pipeline.py`); on all 11,069 CCMT images it rejects none.

This directly addresses two risks from Team 1's risk assessment: **incorrect disease classification** (mitigated by allowing the farmer to check the result) and **false disease detection** (mitigated by flagging low-confidence results rather than acting on them automatically).

### B4. Latency & Real-Time Constraints

MobileNetV3-Large was selected specifically for its lower computational footprint relative to larger CNN architectures, in anticipation of eventual edge deployment (Member 5's architecture document, §9). Measured on the team's laptop CPU (Intel Core i3-1115G4, no GPU), the exported model takes a median of **13.8 ms per image** (95th percentile 16.9 ms, one thread, model inference only), roughly 70 leaf images per second. The model has 2.98 million parameters and the TorchScript file is 12.5 MB. Latency on an actual drone companion computer still has to be measured once hardware is available.

---

## Summary

This document covers the reasoning behind the training strategy, hyperparameter approach, class-imbalance handling, and deployment logic implemented in the accompanying `train_deploy_demo.py` script, and the results of the two training runs, satisfying the rubric's "Solution Techniques" (appropriate techniques and how the AI Model will improve its accuracy) and supporting the "Practical Solution" section.
