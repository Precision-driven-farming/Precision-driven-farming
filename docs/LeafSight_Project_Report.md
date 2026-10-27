# LeafSight: AI Crop Disease Detection and Precision Spraying

**Theme:** An AI Solution for Industries — **Industry:** Agriculture (maize and tomato farming)
**Repository:** Precision-driven-farming · **Due:** 02 November 2026

> **How to use this document.** It is written in the order of the marking rubric, so each
> section can be pasted into the group's Word document under the same heading. Word counts for
> the limited sections are given in brackets and were counted by script. Run the final Word
> document through Grammarly and attach the report, as the brief requires.

| Rubric line | Marks | Section |
|---|---|---|
| AI Solution relevant to the theme | 5 | 1 |
| Business objectives, success criteria, background; requirements, constraints, risks; tools and techniques | 25 | 3 |
| Problem definition | 10 | 2 |
| Poster | 10 | 7, `project_poster.html` |
| Machine learning approach | 5 | 5.1 |
| Data | 5 | 5.2 |
| Model evaluation and time-series analysis | 10 | 5.3, 5.4 |
| Solution techniques and improving accuracy | 5 | 5.5 |
| NLP / speech | 5 | 5.6 |
| Deep learning | 5 | 5.7 |
| Other features: chatbot | 5 | 5.8 |
| Practical solution | 10 | 6 |

---

## 1. AI Solution and its relevance to the theme

*(one paragraph, 194 words)*

LeafSight is an AI solution for the agriculture industry, the sector the 4IR has so far done least for in developing countries. Cameras on a drone or on fixed poles photograph maize and tomato leaves as the field is scanned. A deep learning model (MobileNetV3, trained on 11,069 expert-validated CCMT images from Ghana) sorts each leaf into one of 12 classes: nine different pests and diseases, or healthy. The system then turns those results into a precision spraying map: a 20 m grid that shows which zones need treatment, how urgently and at what dose, so that only those zones are sprayed. A time-series analysis of repeated scans warns of zones that are heading towards trouble before they get there, and a chatbot lets the farmer ask about the results in plain English. Nothing is sprayed until the farmer has checked and approved each zone. The model is light enough to run on a laptop or a drone's onboard computer, so the solution can reach small and emerging farmers, not only large commercial farms, which is exactly the gap between the promise of the 4IR and people on the ground that the theme describes.

## 2. Problem definition

*(246 words including the three question headings, limit 250; "What exactly is the problem?" 148 words, limit 190)*

**What exactly is the problem?**
Farmers find crop pests and diseases by walking the field and looking at leaves. This is slow and often late: by the time damage is visible, fall armyworm or leaf blight has usually spread, the yield loss is locked in, and treatment costs more or fails. Because farmers cannot see which plants are affected, they spray the whole field "to be safe", paying for chemicals that land mostly on healthy plants and run off into soil and rivers. The losses are large. FAO estimates that up to 40% of the world's crop production is lost to pests and diseases each year, and fall armyworm alone costs African maize farmers an estimated USD 1 to 4.6 billion a year. Small-scale farmers are hit hardest: South Africa has about 2.4 million, and they cannot afford agronomists, scouts or wasted chemicals. The problem is therefore late detection combined with untargeted spraying.

**How is it relevant to the theme?**
The problem sits in agriculture, a core industry, and is solved with AI: computer vision, deep learning, time-series analysis and natural language processing.

**How will solving it with AI benefit the local municipality?**
For the local municipality, Emfuleni in the Vaal, where the demo field is located, less blanket spraying means less chemical runoff into the Vaal River, already under pollution pressure. Earlier detection protects local food supply and farm jobs, and a low-cost tool lets small farmers produce enough to sell, not only to eat.

## 3. Business understanding

### 3.1 Business background

Maize is South Africa's staple food crop and tomato one of its main vegetable crops, and both are grown by large commercial farms and by small-scale farmers. Agriculture employed about 920,000 people in South Africa at the end of 2024 (Stats SA Quarterly Labour Force Survey), and the 2022 census counted about 2.4 million small-scale farmers, of whom only about 12% sell any produce. For these farmers a pest outbreak decides whether there is anything to sell.

Today, crop protection on most farms works like this: someone walks the rows, inspects leaves by eye, and if a problem is found, the whole field is sprayed. Commercial farms can pay agronomists and scouts; small farms usually cannot. Precision agriculture already shows what targeting can save: in a peer-reviewed maize trial, drone-guided spot spraying used 47% less herbicide than blanket spraying with the same weed control (Allmendinger et al., 2024). What has been missing is an affordable way to find *which* plants need treatment early enough to matter, for pests and diseases rather than weeds.

LeafSight's target customer is a maize and tomato farm of a few hectares to a few hundred hectares, from emerging farmers supported by an agricultural cooperative or the municipality's agricultural extension services, to commercial growers. The demo field in this project is a 60 m × 60 m maize block (a 3 × 3 grid of 20 m zones) at Vereeniging, in Emfuleni Local Municipality.

### 3.2 Business objectives

1. **Detect pests and diseases earlier** than walking inspections, so farmers can act before damage spreads.
2. **Cut chemical use and cost** by spraying only the zones that need it, at a dose matched to severity.
3. **Reduce yield losses** from the nine pests and diseases the system recognises on maize and tomato.
4. **Reduce environmental harm**: less pesticide runoff into soil and the Vaal River catchment.
5. **Keep the farmer in control**: every AI result is explained, its confidence shown, and nothing is sprayed without the farmer's approval.
6. **Be affordable and usable for small farmers**: low-cost hardware, works offline, plain-language explanations.

### 3.3 Business success criteria

Each objective has a measurable test. "Status" is what this project has measured so far (see sections 5.3 and 5.4 for details).

| # | Objective | Success criterion | How it is measured | Status in this project |
|---|---|---|---|---|
| 1 | Earlier detection | Warn at least 7 days before a zone falls below the review level (60% healthy leaves) | Time-series early-warning lead time, replaying a season scan by scan | **11 days** on the simulated season (5.4) |
| 2 | Less chemical | At least 40% less chemical than blanket spraying | Spraying map dose total vs spraying every zone at full dose | **77.8%** less on the simulated scan; field trials report 47% |
| 3 | Fewer losses | Disease classifier catches sick plants: macro recall ≥ 92% | Held-out test set, 12 classes | **Not yet met: 85.3%** (baseline 79.8%) (5.3) |
| 3 | Fewer losses | Overall accuracy ≥ 90%, macro precision ≥ 87%, macro F1 ≥ 89.5% | Held-out test set | **Not yet met:** accuracy 84.3%, precision 83.8%, F1 84.3% (5.3) |
| 4 | Less runoff | Untreated area ≥ 50% of the field on every run | Spraying map: at most 50% of zones may be sprayed per run (enforced in code) | Enforced; 2 of 9 zones sprayed in the demo |
| 5 | Farmer in control | 100% of spray zones need explicit farmer approval; low-confidence results always flagged | Pipeline verification gate; chatbot safety tests | Enforced in code; 13.4% of test results flagged; 22/22 chatbot tests pass |
| 6 | Usable | Chatbot understands farmer questions correctly ≥ 90% of the time | 113 held-out phrasings | **97.3%** (110/113) |

The model targets (rows 3) are deliberately high, and recall comes first: missing a sick plant costs more than a false alarm, because a missed outbreak spreads, while a false alarm is caught when the farmer reviews the zone. Tuning closed about half of the gap (accuracy from 75.2% to 84.3%). The remaining gap comes from look-alike lesion classes and from training on a laptop CPU, which allowed one tuning experiment of about three hours (section 5.5). Until the targets are met, the farmer-approval gate is what keeps a wrong result from becoming a wrong spray.

### 3.4 Requirements

**Functional requirements**

| ID | Requirement | Where it is met |
|---|---|---|
| F1 | Classify a leaf image into one of 12 classes (7 maize, 5 tomato) with a confidence score | `model/train_deploy_demo.py`, `predict_image()` |
| F2 | Flag predictions below 70% confidence for farmer verification | `CONFIDENCE_THRESHOLD`, both in the model and the pipeline |
| F3 | Turn a batch of geotagged leaf results into a zone-by-zone spraying map with severity, action and dose | `app/spraying_pipeline.py` |
| F4 | Never spray more than 50% of the field in one run; prioritise the most severe zones | `MAX_SPRAY_FRACTION`, `decide_actions()` |
| F5 | Require farmer approval of every zone before it is released for spraying | `release_for_spraying()`; dashboard status "Awaiting farmer review" |
| F6 | Track each zone's health over time, forecast it, and warn early | `model/time_series_analysis.py` |
| F7 | Answer farmer questions in plain English, without recommending products or doses | `app/chatbot/agriassist.py` |
| F8 | Show the map, zone details, trends and the chat on one screen | `farmer_dashboard.html` |
| F9 | Refuse to classify images that are not usable leaf photos (blank, badly exposed, sensor noise) and ask for a re-scan | `image_quality_issue()` in the spraying pipeline |

**Non-functional requirements**

| ID | Requirement | How it is addressed |
|---|---|---|
| N1 | Runs without a GPU and without internet in the field | MobileNetV3 model with about 3.0 million parameters (12 MB file); TorchScript export; chatbot runs locally |
| N2 | Compatible with Python, PyTorch and OpenCV | The whole stack is Python; `requirements.txt` |
| N3 | Explainable to non-experts | Confidence shown on every result; plain-language disease descriptions with sources |
| N4 | Accessible | Colour-blind-safe palette, symbols as well as colours on the map, screen-reader labels on the dashboard |
| N5 | Reproducible | Fixed random seed; stratified split; one script produces the model, metrics and charts |

### 3.5 Constraints

* **Budget:** no paid hardware or cloud; the team has no GPU, so training ran on a laptop CPU (section 5.5).
* **Time:** about ten weeks from kickoff to submission (22 August to 2 November 2026).
* **Data:** only public data. The CCMT dataset covers 12 maize and tomato classes from one region (Ghana); there are no South African field images and no labelled time series.
* **Hardware:** no drone was available, so drone flights, GPS and timestamps are simulated in the pipeline and clearly labelled as such.
* **Regulation and safety:** the system must not recommend products, doses or mixing rates; pesticide use follows product labels and local regulations, and the farmer makes every spraying decision.

### 3.6 Risks and contingencies

| Risk | Likelihood | Impact | Mitigation in this project |
|---|---|---|---|
| Model misclassifies a disease (false negative) | Medium | High: an outbreak spreads | Recall prioritised; class-weighted loss; confidence threshold; farmer review; time-series early warning catches what one scan misses |
| Model raises a false alarm (false positive) | Medium | Low–medium: wasted inspection | Farmer approves every zone; `flag_incorrect` feedback goes to the team |
| Model is confidently wrong on an image that is not a leaf (camera fault, blank frame) | Medium | High: a spray decided by noise | Input check before classification rejects such images (0 of 11,069 real images rejected, every noise image caught); the zone is marked for re-scan |
| Dataset differs from local fields (domain shift) | High | High | Results reported honestly as CCMT test results; next step is collecting and labelling local images |
| Class imbalance (Septoria 2,743 vs Healthy maize 204 images) | Certain | Medium | Class weights; per-class recall reported; stratified split |
| Farmers over-trust the AI | Medium | High | Confidence first in every reply; tiers with warnings; the chatbot never says "spray" |
| Poor GPS puts a result in the wrong zone | Medium | Medium | Zones with >30% poor GPS fixes are flagged for the farmer to confirm |
| Drone battery or weather stops a scan | Medium | Low | The chatbot reports uncovered zones; the time series tolerates missing scans |
| No GPU for training | Certain | Medium | Lightweight model, two-stage transfer learning, checkpoints saved every epoch |
| Data loss (dataset overwritten by the demo mode) | Low | High | Guard in the code refuses to delete real data; dataset kept out of git |

### 3.7 Initial assessment of tools and techniques

| Need | Chosen | Why | Alternatives considered |
|---|---|---|---|
| Language | Python 3 | Required by the brief; best ML ecosystem | — |
| Deep learning framework | PyTorch + torchvision | Pretrained MobileNetV3; TorchScript export for edge devices | TensorFlow/Keras (equally capable; the team chose one stack) |
| Image processing | OpenCV | Fast image validation, resizing and colour conversion | Pillow alone (fewer tools) |
| Image model | MobileNetV3-Large, transfer learning | Small and fast enough for edge devices, strong ImageNet features | Custom CNN (too little data to train from scratch); EfficientNet-Lite (similar, slower to fine-tune on CPU) |
| Classical ML / evaluation | scikit-learn | Stratified splits, class weights, metrics, TF-IDF and logistic regression for the chatbot | — |
| Time series | NumPy (EWMA, OLS trend, damped Holt) | Transparent, no extra dependencies, works on short noisy series | ARIMA (needs longer series than one season provides) |
| NLP | TF-IDF + logistic regression + rule-based entities | Small labelled set (733 phrases), fast, explainable, offline | Large language model (needs internet or a GPU; harder to make safe) |
| Dashboard | Single HTML file | Runs in any browser, offline | Web framework (more setup for no gain at prototype stage) |
| Charts | Matplotlib | Confusion matrix, training curves, forecasts | — |
| Dataset | CCMT (Kaggle) | Expert-validated, maize and tomato, African fields | PlantVillage (lab photos on plain backgrounds, less realistic) |

## 4. Main objective of the AI solution and how AI is applied

**Main objective:** give farmers an early, zone-by-zone picture of pests and diseases in their maize and tomato fields, and use it to spray only where needed, with the farmer approving every action.

AI is applied at four points, each feeding the next:

1. **See** — *computer vision with deep learning.* Each leaf photo from the drone or pole camera is checked, resized and normalised (OpenCV), then classified by a fine-tuned MobileNetV3 into one of 12 classes with a confidence score.
2. **Decide** — *rule-based optimisation on the AI output.* Each detection becomes a severity score (how destructive the pest or disease is × how confident the model is). Detections are grouped into 20 m zones by GPS, each zone gets a severity and a recommended action (no action, monitor, standard spray at 60% dose, high-priority spray at full dose), and at most half the field is sprayed per run.
3. **Track and forecast** — *time-series analysis.* Each scan adds one health value per zone. Smoothing, trend and a damped Holt forecast warn about zones heading below the review level, and a before/after comparison checks whether a spray actually worked.
4. **Explain and listen** — *natural language processing.* The farmer asks questions in plain English; an intent classifier and entity extraction find what they want, and the chatbot answers with confidence levels and safety rules built in.

The farmer reviews the map on the dashboard, checks flagged zones in the field, and approves or rejects each one. Only approved zones are released for spraying.

## 5. AI Solution — theoretical aspect

### 5.1 Machine learning approach

| Task | Type of learning | Algorithm | Why it fits |
|---|---|---|---|
| Leaf classification | Supervised, multi-class (12 classes) | Convolutional neural network, MobileNetV3-Large, transfer learning from ImageNet | Images need learned visual features; transfer learning works with ~11,000 images |
| Class imbalance | — | Class-weighted cross-entropy loss | Minority classes (Healthy maize, Fall armyworm) would otherwise be ignored |
| Spray decision | Rule-based optimisation on model output | Severity = class weight × confidence; zone severity = 0.7 × mean + 0.3 × max; greedy selection under a 50% budget | Transparent and auditable, which matters when chemicals are involved |
| Crop-health forecasting | Time-series analysis | EWMA smoothing, least-squares trend, damped Holt exponential smoothing | Short, noisy, irregular series; Holt models level and trend |
| Farmer questions | Supervised text classification | TF-IDF (word and character n-grams) + logistic regression | Small labelled set, fast, gives a probability used for clarification |
| Entities in questions | Rule-based | Regular expressions over fixed vocabularies | Zones, crops, diseases and dates are a small closed set |

### 5.2 Data

**Source.** CCMT: Dataset for crop pest and disease detection (Mensah et al., 2023, *Data in Brief*), expert-validated by plant virologists and pathologists. The project uses the raw maize and tomato images only (the dataset also has cashew and cassava, which are out of scope). The training script applies its own augmentation, so the pre-augmented copies are not needed.

| Crop | Classes | Images |
|---|---|---|
| Maize | Fall armyworm 285, Grasshopper 673, Healthy 204, Leaf beetle 933, Leaf blight 990, Leaf spot 1,239, Streak virus 965 | 5,289 |
| Tomato | Healthy 466, Leaf blight 1,288, Leaf curl 511, Septoria leaf spot 2,743, Verticillium wilt 772 | 5,780 |
| **Total** | **12 classes** | **11,069** |

These counts were checked three ways and match exactly: the team's EDA write-up, the files on disk, and the training script's own data loader. Two folder names carry spelling errors in the source data ("grasshoper", "verticulium wilt"); the code maps them to the correct names.

**Split.** Stratified 70 / 15 / 15: 7,748 training, 1,660 validation and 1,661 test images, with a fixed random seed. Stratification keeps every class's share the same in each part. The test set is used once, at the end.

**All forms of data in the solution, with examples:**

| Form | Example | Where |
|---|---|---|
| Leaf image (RGB JPEG, 400 × 400, resized to 224 × 224) | `data/Raw Data/Maize/fall armyworm/fall armyworm174_.jpg` | CCMT |
| Label (from the folder) | `Maize/leaf blight` → class 4, "Leaf blight (Maize)" | `RAW_TO_LABEL`, `CLASS_NAMES` |
| Model output | `Leaf spot: 91%`, plus `[LOW CONFIDENCE …]` below 70% | `predict_image()` |
| Detection record (image + GPS + time + result) | `{"image_path": "img_041.jpg", "latitude": -26.66955, "longitude": 27.93050, "gps_accuracy_m": 1.5, "timestamp": "2026-10-08T06:19:50+00:00", "label": "Fall armyworm", "confidence_pct": 88.6, "low_confidence": false, "severity": 79.7}` | spraying pipeline |
| Spraying map (per zone) | `{"zone_id": "R2C2", "severity": 74.9, "recommended_action": "SPRAY - HIGH PRIORITY", "recommended_dose_pct": 100, "dominant_class": "Fall armyworm", "status": "AWAITING_FARMER_REVIEW"}` | `spraying_map.json`, dashboard |
| Scan history (time series) | `2026-09-17,R2C2,34,18,52.9,1` (date, zone, images, healthy images, health %, sprayed) | `data/sample_scan_history.csv` |
| Farmer question (text) | "How confident is the AI about R2C1?" | chatbot |
| Parsed query (JSON) | `{"intent": "get_confidence", "entities": {"zone": "R2C1", …}, "requires_farmer_verification": true}` | chatbot |

**Limitations.** One region (Ghana) and two crops; the dataset does not name the pathogen behind its general labels ("leaf blight", "leaf spot", "leaf curl"); no time dimension, so the time-series sample is simulated and labelled as such.

### 5.3 Model and how its accuracy is evaluated

**Evaluation design.** The model is trained on the training set, the best epoch is chosen on the validation set, and the held-out test set (1,661 images the model never saw) is scored once at the end. Metrics:

* **Accuracy** — share of test images classified correctly.
* **Macro recall** — the average, over the 12 classes, of how many sick (or healthy) leaves of that class were found. This is the priority, because a missed outbreak is the costliest error.
* **Macro precision** — the average share of each class's predictions that were right (false alarms lower it).
* **Macro F1** — the balance of the two. *Macro* averages treat every class equally, so the small classes count as much as Septoria leaf spot.
* **Per-class precision, recall and F1**, and a **12 × 12 confusion matrix** built with the full crop-specific class names (so a maize leaf predicted as a tomato leaf counts as an error even when both are "Healthy").

**Results on the held-out test set.** Two models were trained (section 5.5). The deployed model was chosen on validation accuracy; the test set was then scored once per model.

| Metric | Target (3.3) | Baseline: top 3 blocks fine-tuned | **Deployed: whole backbone fine-tuned** |
|---|---|---|---|
| Accuracy | ≥ 90% | 75.2% | **84.3%** |
| Macro recall | ≥ 92% | 79.8% | **85.3%** |
| Macro precision | ≥ 87% | 74.1% | **83.8%** |
| Macro F1 | ≥ 89.5% | 76.2% | **84.3%** |
| Validation accuracy (best epoch) | | 76.1% | 83.1% |

Per class, deployed model (1,661 test images):

| Class | Precision | Recall | F1 | Test images |
|---|---|---|---|---|
| Fall armyworm | 88.1% | 88.1% | 88.1% | 42 |
| Grasshopper | 98.0% | 98.0% | 98.0% | 101 |
| Healthy (Maize) | 73.0% | 90.0% | 80.6% | 30 |
| Leaf beetle | 96.4% | 95.7% | 96.1% | 140 |
| Leaf blight (Maize) | 72.4% | 79.2% | 75.6% | 149 |
| Leaf spot | 77.6% | 71.0% | 74.2% | 186 |
| Streak virus | 91.5% | 89.0% | 90.2% | 145 |
| Healthy (Tomato) | 93.2% | 97.1% | 95.1% | 70 |
| Leaf blight (Tomato) | 77.8% | 78.2% | 78.0% | 193 |
| Leaf curl | 70.7% | 84.4% | 76.9% | 77 |
| Septoria leaf spot | 88.4% | 88.6% | 88.5% | 412 |
| Verticillium wilt | 78.1% | 64.7% | 70.8% | 116 |

**What the confusion matrix shows** (`model/outputs/tuned/confusion_matrix.png`):

* **No crop mix-ups.** Not one maize leaf was predicted as a tomato class or the other way round, in either model. Every error is between classes of the same crop.
* **Errors sit between look-alike lesions.** Maize leaf spot and maize leaf blight are confused in both directions (39 and 25 images); tomato Septoria leaf spot and tomato leaf blight likewise (24 and 22); Verticillium wilt is mostly mistaken for Septoria (23) or leaf blight (14). Pest damage (grasshopper 98%, leaf beetle 96% recall) and streak virus (89%) are recognised well because their patterns are distinctive.
* **Weakest class:** Verticillium wilt, 64.7% recall. Its leaf symptoms (yellowing and browning) overlap with the other tomato lesion classes.
* **What tuning changed:** the largest gains are on the tomato lesion classes that the baseline found hardest: tomato leaf blight recall rose from 51.8% to 78.2%, and F1 for Verticillium wilt from 52.0% to 70.8% and for leaf curl from 59.0% to 76.9%. One class went the other way: fall armyworm recall fell from 97.6% to 88.1% (5 of 42 missed instead of 1), while its precision rose from 75.9% to 88.1%. Fall armyworm is the costliest pest, so per-class thresholds (5.5) start with it.

**Confidence and the 70% verification flag.** The test set was also scored by confidence, because the flag decides how much checking the farmer must do:

| | Baseline | Deployed |
|---|---|---|
| Results at ≥ 70% confidence (share of images, and how many are right) | 59.3%, 89.9% right | **86.6%, 89.4% right** |
| Results flagged below 70% for the farmer to check | 40.7% | **13.4%** |
| Share of all errors that the flag catches | 76.0% | 41.8% |

The deployed model asks the farmer to check about one image in seven instead of two in five, and its confident results are as reliable as the baseline's. The cost is that more of its mistakes are confident (152 of 261), which is why no zone is ever sprayed without the farmer's approval and why the threshold is tuned per class next.

**Input that is not a leaf.** On random-noise images the deployed model answers "Streak virus" at about 89% confidence, so confidence alone cannot detect a camera fault or a frame with no leaf in it. The spraying pipeline therefore checks every image before classifying it: a blank frame, bad exposure, or pixel noise (neighbouring pixels uncorrelated) is rejected. The limits were calibrated on all 11,069 CCMT images: the check rejects none of them and every noise image. A zone with no usable images is marked "RE-SCAN" and is never sprayed.

### 5.4 Time-series analysis

**Why.** One scan shows how a zone looks today; repeated scans show where it is heading. Every scan gives each zone a **health index**: the percentage of its leaf images classified Healthy. Over a season this is one time series per zone. `model/time_series_analysis.py` analyses it.

**Data.** A CSV with one row per zone per scan (sample in section 5.2). CCMT has no time dimension, so until real scans exist the script uses a **simulated** 8-week season: scans every Monday and Thursday, 9 zones, ~30 leaf images per zone per scan. The number of healthy images is drawn at random around a true weekly curve, so the noise is the real kind that comes from photographing a limited number of leaves. The story matches the poster: disease starts in R2C2, spreads to its neighbour R2C1, a spray is approved on day 31 (week 5), and both zones recover.

**Methods.**

1. *Smoothing:* exponentially weighted moving average (α = 0.4), because one scan of ~30 leaves is noisy.
2. *Trend:* least-squares slope over the last 14 days, in percentage points per day, with a 95% confidence interval.
3. *Forecast:* Holt's exponential smoothing with a **damped trend**, its level, trend and damping constants chosen on each zone's own history. A recorded spray resets the trend, because treatment changes the zone's direction.
4. *Backtest:* rolling-origin evaluation. At every point in the season the forecast is made using only earlier scans, and compared with what the next scan, and the scan a week later, actually showed. Holt is compared with three baselines (last value, moving average of 3, EWMA).
5. *Early warning:* a zone forecast to fall below 60% healthy within 7 days is flagged for an extra scan and farmer review.
6. *Treatment check:* the trend in the 21 days before a spray against the 21 days after.
7. *Spread:* the lagged correlation between a diseased zone and its neighbour.

**Results on the simulated season.**

| Analysis | Result |
|---|---|
| Early warning | R2C2 was flagged on **day 17, 11 days before** its smoothed health actually fell below 60% (day 28) |
| Treatment check | R2C2: trend changed from −1.88 to **+2.34 points/day** after the spray, 95% CI above zero: recovering. R2C1: −1.14 → +0.93, interval includes zero, so the system asks for a review rather than claiming success |
| Spread | R2C1 follows R2C2 with a lag of about **4 days** (correlation 0.82) |
| Forecast accuracy (mean absolute error, % points) | Last value 9.31, damped Holt 9.73, moving average 10.49, EWMA 10.50. The sampling-noise floor is 5.6: even a perfect forecast scores about that, because each scan of ~30 leaves is itself that noisy |

**What this shows.** With ~30 images per zone, the noise of a single scan is as large as a week's change in health, so no forecaster beats simply repeating the last scan by much; the honest conclusion is that point forecasts need more images per zone. The time series earns its place through the **early warning** (11 days of lead time, which is the business objective of earlier detection) and the **treatment check**, which tells the farmer whether a spray worked. Charts: `model/outputs/time_series/zone_health_trends.png` and `forecast_backtest.png`.

### 5.5 Solution techniques and how the model's accuracy improves

**Techniques used to reach the solution**

* *Transfer learning in two stages.* Stage 1 freezes the ImageNet-pretrained backbone and trains only the new classification head (learning rate 10⁻³). Stage 2 unfreezes the whole backbone and fine-tunes it with a learning rate ten times lower than the head's (10⁻⁴ against 10⁻³), so the general visual features are adapted to leaves without being destroyed.
* *Class-weighted loss* against imbalance (weights from inverse class frequency on the training set).
* *Data augmentation* on the training set only: random horizontal flips, rotations up to 20°, and brightness, contrast and saturation changes, so the model copes with different angles and light.
* *Adaptive learning rate:* the rate halves when validation accuracy stops improving for 2 epochs (ReduceLROnPlateau).
* *Early stopping and checkpointing:* the best weights (by validation accuracy) are saved to disk every time they improve, a stage stops after 4 epochs without improvement, and the model exported for deployment is the best checkpoint, not the last epoch.
* *Confidence threshold:* below 70% confidence a prediction is never treated as final; the farmer is asked to verify.
* *Input check:* images that are blank, badly exposed or pure sensor noise are rejected before they reach the model (5.3).

**How accuracy was improved: the tuning experiment.** The original plan (the baseline) fine-tuned only the top 3 of the backbone's 17 blocks, at learning rates of 10⁻⁵ (backbone) and 10⁻⁴ (head). The experiment fine-tuned all 17 blocks at rates ten times higher. Both runs started stage 2 from the same stage-1 weights, so only the fine-tuning recipe differed.

| | Baseline | Tuned (deployed) |
|---|---|---|
| Blocks fine-tuned, learning rates | top 3; 10⁻⁵ / 10⁻⁴ | all 17; 10⁻⁴ / 10⁻³ |
| Best validation accuracy | 76.1% (epoch 12 of 12) | **83.1%** (epoch 11 of 12) |
| Test accuracy / macro F1 | 75.2% / 76.2% | **84.3% / 84.3%** |
| Stage-2 training time on a laptop CPU (Intel Core i3, no GPU) | 66 minutes | 124 minutes |

Stage 1 (10 epochs, 59 minutes) reached 72.9% validation accuracy and is shared by both. The baseline was still improving slowly when it ended, because three blocks at a very low rate leave the model little room to adapt to leaves, which look nothing like most ImageNet photos. In the tuned run, training accuracy ended at 92.7% against 82.7% on validation, a mild overfit that the best-checkpoint rule handles: the exported model is epoch 11, not the last epoch. The tuned recipe is now the script's default.

**How accuracy is improved over time**

1. *Further tuning on the validation set,* each experiment one run of the same script: a larger input size (224 pixels loses fine lesion detail, the main source of the remaining confusions), longer training on a GPU, stronger augmentation, and dropout.
2. *Farmer feedback loop:* every result a farmer flags as wrong ("the R2C1 result is wrong") is logged; reviewed flags become new labelled training images from the farmer's own fields, which also tackles the regional limitation of the dataset.
3. *More images per zone per scan* reduce the noise in the health index (section 5.4) and make forecasts sharper.
4. *Threshold tuning:* the 70% confidence threshold is a starting value, to be set from the validation set's precision–recall trade-off per class, starting with fall armyworm.

### 5.6 Natural language processing (and speech)

The chatbot (section 5.8) is the NLP component. Its pipeline: text preprocessing (lower case, punctuation removed, grid zone names normalised to one word) → **intent classification** with TF-IDF over word 1–2-grams and character 2–4-grams and logistic regression, trained on 733 example phrases across 16 intents → **entity extraction** for crop, zone (grid names or plain directions such as "the north-east corner"), pest or disease (including the dataset's misspellings) and dates ("last week", "3 October") → a structured JSON query → an answer. Character n-grams make it tolerant of spelling mistakes ("sprayin"), and when the classifier is less than 35% sure, the chatbot asks a clarifying question instead of guessing.

**Measured:** 97.3% on 113 held-out phrasings and 88% in 5-fold cross-validation.

**Speech** is the natural next step for farmers who prefer talking to typing: speech recognition in front of the same pipeline (the parsed query already has an `input_mode` field for "voice") and speech synthesis to read answers aloud. It is not in the prototype.

### 5.7 Deep learning

* **Convolutional neural network.** Convolution layers learn local patterns (lesion edges, spots, chewed holes, streaks) and stack them into higher-level features.
* **MobileNetV3-Large** (Howard et al., 2019): built from *depthwise separable convolutions* (a cheap per-channel filter followed by a 1×1 mixing convolution), *inverted residual blocks with linear bottlenecks*, *squeeze-and-excitation* attention that reweights channels, and the *hard-swish* activation. Its architecture was found by neural architecture search for mobile devices, which is why it suits a drone or a laptop.
* **Transfer learning and fine-tuning** from ImageNet weights (section 5.5).
* **Classification head:** global average pooling → dropout (0.3, against overfitting) → fully connected layer with 12 outputs → softmax at inference, which turns the outputs into the confidence scores.
* **Training:** cross-entropy loss with class weights, AdamW optimiser with weight decay, mini-batches of 32, backpropagation; ImageNet normalisation of inputs.
* **Deployment:** the trained model is exported to TorchScript, a self-contained file that the spraying pipeline loads without the training code, ready for an onboard computer.

### 5.8 Other features: chatbot (AgriAssist)

The chatbot is relevant because farmers need answers, not charts: "What's wrong with my maize?", "Which zones need spraying?", "How sure is the AI?". It is set up as follows:

* **16 intents**, from diagnosis, location, affected area, spray map, confidence, explanation, history and trends to recording a farmer's review or flagging a wrong result.
* **Uses the same zones and classes as the rest of the system:** grid cells R0C0–R2C2 and the model's class names, with the latest scan matching the dashboard.
* **Sourced disease information:** plain-language descriptions of all ten class names, each from an extension service or research body (`app/chatbot/disease_info_sources.md`).
* **Safety rules S1–S9, enforced in code:** confidence first; verify before action; never instruct spraying; confidence tiers with warnings; ask when unsure; no products or doses; honest about data gaps (old scans, missed zones, blurry photos); recording is not acting; keep a log.
* **Tested:** 22 automated tests pass.

## 6. AI Solution — practical

Everything runs from the repository root with `pip install -r requirements.txt`:

| Command | What it does |
|---|---|
| `python model/train_deploy_demo.py` | Trains the model on CCMT with the deployed recipe, evaluates it on the test set, exports `leaf_classifier.pt`, and writes metrics, the confusion matrix and the training curves to `model/outputs/`. The submitted runs are in `model/outputs/tuned/` (deployed) and `model/outputs/baseline/` |
| `python model/predict_demo.py --class "Fall armyworm"` | Live prediction on a held-out test image in the farmer-facing format |
| `python app/spraying_pipeline.py` | Loads the trained model, checks and classifies a batch of geotagged images, and writes `spraying_map.json` |
| `python model/time_series_analysis.py` | Time-series analysis and forecast charts |
| `python app/chatbot/agriassist.py --web` | The chatbot in the browser (`--demo` replays sample conversations) |
| `farmer_dashboard.html` | The farmer's dashboard: spray map, zone details, trends and chat |

The spraying demo on the dashboard and poster uses simulated detections and is labelled "Simulated demo run", so its numbers (2 of 9 zones, 77.8% less chemical) stay consistent across the poster and dashboard. The trained model is demonstrated separately through its test-set results (section 5.3) and live predictions on real images. Because there is no drone, the pipeline's placeholder images are random noise; with the trained model loaded, the input check rejects every one and all nine zones are marked "RE-SCAN" with nothing sprayed, which is the correct behaviour for a batch with no usable leaf photos.

## 7. Poster

`project_poster.html` (A1 portrait) gives the overview in eight panels: the problem, the solution pipeline, data and model, how accuracy is measured (with the test-set results), the precision spraying map, the chatbot and dashboard, crop-health trends, and impact and next steps.

## 8. References

* Allmendinger, A., Spaeth, M., Saile, M., Peteinatos, G. G. and Gerhards, R. (2024). Agronomic and technical evaluation of herbicide spot spraying in maize based on high-resolution aerial weed maps: an on-farm trial. *Plants*, 13(15), 2164. <https://doi.org/10.3390/plants13152164>
* CABI (2018). Fall armyworm: impacts and implications for Africa, evidence note update (USD 1–4.6 billion per year across 12 countries), as reported in <https://www.allaboutfeed.net/animal-feed/raw-materials/progress-made-on-controlling-fall-armyworm>
* FAO (2021). Scientific review of the impact of climate change on plant pests (up to 40% of crop production lost annually), as reported by the World Economic Forum: <https://www.weforum.org/stories/2021/06/climate-change-insects-pests-crops-agriculture/>
* Howard, A., Sandler, M., Chu, G. *et al.* (2019). Searching for MobileNetV3. *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 1314–1324.
* Mensah, P. K., Akoto-Adjepong, V., Adu, K. *et al.* (2023). CCMT: Dataset for crop pest and disease detection. *Data in Brief*, 49, 109306. <https://doi.org/10.1016/j.dib.2023.109306>
* Statistics South Africa, Census 2022 agricultural households (2.4 million small-scale farmers, about 12% selling), as reported in <https://farmingportal.co.za/index.php/farming-news/south-africa/12409-south-africa-has-2-4-million-small-scale-farmers-but-only-12-sell-anything>
* Statistics South Africa, Quarterly Labour Force Survey Q4 2024 (agricultural employment about 920,000), as reported in <https://iol.co.za/business-report/opinion/2025-03-04-navigating-employment-fluctuations-in-south-africas-agriculture-sector/>
* University of the Witwatersrand (2021). Pollution of the Vaal River in Emfuleni Local Municipality (thesis). <https://wiredspace.wits.ac.za/handle/10539/31933>
* Disease and pest descriptions: see `app/chatbot/disease_info_sources.md` for the ten sources.
