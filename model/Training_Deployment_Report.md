# Training, Hyperparameter Tuning & Deployment Logic
**Issues:** #26 (Training & Hyperparameter Tuning), #27 (Deployment Logic)
**Author:** Member 6

---

## Part A — Training & Hyperparameter Tuning

### A1. Training Strategy

The model uses **two-stage transfer learning** on the MobileNetV3-Large backbone (pre-trained on ImageNet), matching the architecture confirmed in Member 5's document:

- **Stage 1 — Head training:** The backbone is frozen entirely, and only the new classification head (Dropout → Fully Connected) is trained, using a learning rate of `1e-3`. This lets the randomly-initialised head learn to interpret the pre-trained backbone's features before any backbone weights are disturbed.
- **Stage 2 — Selective fine-tuning:** Only the last 3 inverted-residual blocks of the backbone are unfrozen (not the whole network), using **discriminative learning rates** — `1e-5` for the unfrozen backbone layers and `1e-4` for the head. This keeps the early, general-purpose visual features (edges, textures) intact while allowing the later, more task-specific layers to adapt to leaf-disease patterns specifically.

**Loss function:** `CrossEntropyLoss`, weighted by inverse class frequency (see A3). Raw logits are used during training — Softmax is applied only at inference time — because `CrossEntropyLoss` applies log-softmax internally, and applying Softmax twice would distort gradient scaling.

**Optimizer:** `AdamW`, chosen over plain Adam for its decoupled weight decay (`1e-4`), which reduces overfitting risk — relevant given the dataset's moderate size (~11k raw images across 12 classes).

**Learning rate schedule:** `ReduceLROnPlateau`, monitoring validation accuracy, halving the learning rate after 2 epochs without improvement. This was chosen over a fixed schedule because we don't yet know the real dataset's convergence behaviour, and an adaptive schedule reduces the risk of overshooting or stalling.

### A2. Hyperparameter Tuning Plan

Given the scope of a student project (limited compute time and no dedicated GPU cluster), we use a **manual/guided search** rather than an automated method like grid search or Bayesian optimization, which would require many full training runs to be worthwhile.

Parameters to tune, in priority order:
1. **Dropout rate** (starting at 0.3) — adjusted based on the gap between training and validation accuracy (a widening gap signals overfitting and a need for higher dropout).
2. **Number of unfrozen backbone blocks** (starting at 3) — increased if Stage 2 fine-tuning underperforms, decreased if it overfits quickly.
3. **Learning rates** for Stage 1 and Stage 2 — adjusted based on loss curve behaviour (oscillating loss suggests the rate is too high; flat loss suggests too low).
4. **Batch size** — kept small (16–32) for this project given expected hardware constraints (no dedicated GPU cluster), acknowledging this is a trade-off against training stability.

On the real dataset, we plan **8–10 epochs for Stage 1** and **8–12 epochs for Stage 2**, with early stopping based on validation accuracy, rather than a fixed epoch count.

### A3. Class Imbalance Handling

Team 1's dataset documentation shows meaningful imbalance in both crops — for example, the Healthy class is the smallest in both maize and tomato, while Septoria leaf spot makes up roughly 43–47% of the tomato subset.

We address this with **class-weighted loss**: weights are computed as inversely proportional to class frequency (`sklearn.utils.class_weight.compute_class_weight`, `balanced` mode), so the loss function penalises mistakes on minority classes more heavily than mistakes on majority classes. This was chosen over oversampling/undersampling because it doesn't discard data or risk overfitting to duplicated minority-class images, and doesn't require re-generating the (already large) augmented dataset.

Per Team 1's dataset documentation, accuracy alone will not be used to judge performance — precision, recall, and F1 per class (Member 7's task) are needed specifically because they reveal poor performance on minority classes that overall accuracy would hide.

### A4. Train/Val/Test Split

We use a **stratified 70/15/15 split** (train/validation/test), stratified by class label so that each split preserves the same class proportions as the full dataset — important given the imbalance noted above. A held-out test set (separate from validation, which is used only during training to monitor performance and tune hyperparameters) ensures the final accuracy metrics Member 7 reports reflect genuinely unseen data.

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

Per Team 1's requirement that farmers must be able to check results before making a spraying decision, predictions below a **70% confidence threshold** are flagged for human verification rather than treated as final. This threshold is a starting point, not a tuned value — it will be adjusted experimentally once real validation data is available, likely by examining the confidence distribution of correct vs. incorrect predictions (a task that overlaps with Member 7's evaluation work).

This directly addresses two risks from Team 1's risk assessment: **incorrect disease classification** (mitigated by allowing the farmer to check the result) and **false disease detection** (mitigated by flagging low-confidence results rather than acting on them automatically).

### B4. Latency & Real-Time Constraints

MobileNetV3-Large was selected specifically for its lower computational footprint relative to larger CNN architectures, in anticipation of eventual edge deployment (Member 5's architecture document, §9). Actual latency has not yet been benchmarked on target hardware, since the current demo runs on synthetic data without a physical drone/camera in the loop — this is flagged as a task for once real hardware is available, rather than a gap in the current design.

---

## Summary

This document covers the reasoning behind the training strategy, hyperparameter approach, class-imbalance handling, and deployment logic implemented in the accompanying `training_inference_demo.py` script, satisfying the rubric's "Solution Techniques" (appropriate techniques and how the AI Model will improve its accuracy) and supporting the "Practical Solution" section.
