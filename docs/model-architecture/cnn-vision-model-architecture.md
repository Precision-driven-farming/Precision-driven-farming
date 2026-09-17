# CNN / Vision Model Architecture

## 1. Purpose

This document defines the deep learning architecture for the Precision-Driven Farming crop disease classification system.

The system uses computer vision to analyse crop leaf images and classify visible disease, pest, or healthy conditions. The predictions will support precision agriculture by identifying affected crop areas and providing information that can later be used when generating treatment or spraying maps.

The model is intended to support two deployment environments:

- Agricultural drones
- Fixed cameras positioned around crop fields

The AI system is designed as a decision-support tool. A farmer must review the prediction before any pesticide treatment is applied.

---

## 2. Machine Learning Problem

This project is a supervised image classification problem.

The input is a crop leaf image and the output is a predicted disease, pest, or healthy class together with a confidence score.

The selected CCMT dataset contains maize and tomato images.

### Maize classes

1. Fall armyworm
2. Grasshopper
3. Healthy
4. Leaf beetle
5. Leaf blight
6. Leaf spot
7. Streak virus

### Tomato classes

1. Healthy
2. Leaf blight
3. Leaf curl
4. Septoria leaf spot
5. Verticillium wilt

The model therefore needs to distinguish between visually different crop conditions while handling differences in lighting, background, orientation, image quality and disease appearance.

The project uses a total of **12 classification classes**.

---

## 3. Dataset Input

The CCMT dataset contains crop images collected in Ghana.

The project uses only the maize and tomato portions of the dataset.

The original dataset contains both raw and augmented images. The augmented dataset was produced by the dataset authors. Additional controlled augmentation may be applied during model training to improve generalisation.

The dataset contains class imbalance. Therefore, class weighting and/or controlled sampling will be considered during training.

The dataset also contains folder-name spelling inconsistencies such as:

- `grasshoper` instead of `grasshopper`
- `verticulium wilt` instead of `verticillium wilt`

These names will be standardised during preprocessing without changing the underlying image data.
"The folder names themselves will not be changed (code will load images using the original, misspelled folder names: grasshoper, verticulium wilt). Only the class label used in model output and display will use the corrected spelling."

---

## 4. Image Preprocessing Pipeline

Each image will pass through the following preprocessing pipeline:

```text
Raw image
    ↓
Image validation
    ↓
Resize to 224 × 224
    ↓
RGB conversion
    ↓
Training augmentation
    ↓
Normalisation
    ↓
Tensor conversion
    ↓
CNN model
```

### 4.1 Image validation

Images will be checked before entering the training pipeline.

The validation process should identify:

- Corrupted image files
- Unsupported image formats
- Images with missing data
- Incorrect image dimensions
- Duplicate or unusable images where possible

### 4.2 Image resizing

Images will be resized to:

**224 × 224 pixels**

This provides a consistent input size for the selected MobileNetV3 architecture while keeping computational requirements suitable for potential edge deployment.

### 4.3 RGB conversion

Images will be converted to RGB format so that the model receives three colour channels:

```text
Red
Green
Blue
```

### 4.4 Data augmentation

Training images may be augmented using controlled transformations such as:

- Horizontal flipping
- Small rotations
- Cropping
- Scaling
- Brightness changes
- Contrast changes

Augmentation will only be applied to the training data.

Validation and test images will not receive random augmentation because they should represent the model's performance on unseen data.

---

## 5. Candidate CNN Architectures

Several CNN approaches may be considered during development.

### 5.1 Custom CNN

A custom CNN can be developed specifically for this project.

A simple custom CNN may contain:

```text
Input image
    ↓
Convolution
    ↓
ReLU activation
    ↓
Pooling
    ↓
Convolution
    ↓
ReLU activation
    ↓
Pooling
    ↓
Flatten
    ↓
Fully connected layers
    ↓
12-class output
```

A custom CNN provides greater control over the architecture but may require more training data and tuning.

### 5.2 EfficientNet-Lite

EfficientNet-Lite is another lightweight architecture suitable for computer vision applications.

It can provide a useful comparison against MobileNetV3, particularly when considering model accuracy and computational requirements.

### 5.3 MobileNetV3

MobileNetV3 is selected as the primary candidate because it is designed for efficient image classification with reduced computational requirements.

Its lightweight design makes it suitable for applications where the model may eventually need to operate close to the camera or drone rather than relying entirely on a powerful remote server.

---

## 6. Selected Architecture — MobileNetV3

The primary CNN architecture for this project is:

**MobileNetV3-Large**

The model will use transfer learning with pre-trained weights as the starting point.

The proposed architecture is:

```text
224 × 224 × 3 RGB image
        ↓
Image validation
        ↓
Training augmentation
        ↓
Normalisation
        ↓
MobileNetV3-Large backbone
(pre-trained weights)
        ↓
Global Average Pooling
        ↓
Dropout
        ↓
Fully Connected Layer
        ↓
12-class Softmax
        ↓
Predicted Class + Confidence
```

Transfer learning allows the model to begin with visual features learned from a large image dataset. The final classification layer will be adapted to the 12 crop-condition classes used by this project.

---

## 7. Layer-by-Layer Architecture

The proposed model consists of the following major components.

### 7.1 Input layer

The input layer receives an RGB image with dimensions:

```text
224 × 224 × 3
```

The three channels represent the RGB colour channels.

### 7.2 MobileNetV3 backbone

The MobileNetV3-Large network acts as the feature extractor.

The backbone contains convolutional operations and lightweight building blocks that learn visual patterns from the input image.

These patterns may include:

- Leaf edges
- Spots
- Discolouration
- Lesions
- Texture
- Pest damage
- Shape patterns

### 7.3 Global Average Pooling

Global Average Pooling reduces the spatial feature maps produced by the backbone into a smaller feature representation.

This reduces the number of parameters before classification and helps the model focus on the learned features.

### 7.4 Dropout

A dropout layer may be used before the final classification layer.

Dropout helps reduce overfitting by randomly disabling some neurons during training.

The dropout rate will be determined during model experimentation.

### 7.5 Fully Connected Classification Layer

The extracted features are passed to a fully connected layer.

The final layer contains:

**12 output neurons**

because the project contains 12 classification classes.

### 7.6 Softmax output

The final layer uses Softmax to produce a probability for each class.

For example:

```text
Leaf spot       0.82
Leaf blight     0.08
Healthy         0.05
Streak virus    0.03
Other classes   0.02
```

The class with the highest probability becomes the predicted class.

The confidence value will be shown to the user as part of the AI result.

---

## 8. Confidence and Human Verification

The AI system will provide a confidence score with each prediction.

A confidence threshold may be introduced during testing.

For example:

```text
High confidence
        ↓
Display prediction

Low confidence
        ↓
Display warning
        ↓
Request human verification
```

The exact threshold will be determined experimentally.

The system will not automatically instruct the farmer to apply pesticides solely based on an AI prediction.

The farmer remains responsible for reviewing the result before treatment.

---

## 9. Edge Deployment

The model is being designed with future edge deployment in mind.

Potential deployment environments include:

- Agricultural drones
- Fixed cameras
- Field computers
- Edge devices

MobileNetV3 is considered because its lightweight architecture can reduce computational requirements compared with larger CNN models.

However, actual deployment performance will depend on the selected hardware, camera quality, image resolution, power availability and processing requirements.

---

## 10. Risks and Trade-offs

### Accuracy versus speed

A larger or more complex model may improve classification performance but require more computational resources.

A lightweight model may provide faster inference but could require additional experimentation to achieve the required accuracy.

### Dataset limitations

The dataset originates from Ghana and may not represent every agricultural environment.

Differences in:

- Climate
- Soil
- Crop variety
- Lighting
- Camera type
- Disease development stage

may affect model performance.

### Class imbalance

Minority classes may be harder to learn.

Class weighting, augmentation and per-class evaluation will therefore be considered.

### False predictions

An incorrect prediction could lead to an incorrect interpretation of crop health.

For this reason, confidence scores and farmer verification are important parts of the system.

---

## 11. Relationship to the Business and AI System

The CNN model supports the business requirement of helping farmers identify crop-health problems faster.

The expected information flow is:

```text
Crop image
    ↓
AI classification
    ↓
Disease / pest information
    ↓
Location information
    ↓
Affected-area information
    ↓
Farmer dashboard
    ↓
Decision support
```

The AI output can later be used to support:

- Crop-health monitoring
- Historical comparisons
- Disease hotspot identification
- Precision treatment planning
- Reduction of unnecessary blanket spraying

The system is therefore intended to support the farmer's decision-making process rather than replace it.

---

## 12. Future Improvements

Future versions of the system may include:

- Larger and more geographically diverse datasets
- Additional crop types
- Additional diseases
- Additional pest classes
- Object detection for locating individual leaves
- Disease severity estimation
- Image segmentation for affected-area measurement
- GPS-based disease hotspot mapping
- Automated historical crop-health comparison
- Model quantisation
- Edge-device optimisation
- Continuous model retraining with verified field data

A future version could combine classification, object detection and segmentation to provide more detailed information about where disease occurs within a field.

---

## 13. Conclusion

The proposed computer vision solution uses MobileNetV3-Large as the primary CNN architecture for classifying maize and tomato crop conditions.

The model receives 224 × 224 RGB images and produces predictions across 12 classes.

The architecture is designed to balance classification performance with computational efficiency so that future deployment on agricultural drones or field-based edge devices can be considered.

The CNN model forms the computer-vision component of the wider Precision-Driven Farming system. Its predictions can be combined with GPS, affected-area analysis and farmer verification to support precision agriculture and treatment planning.

--
