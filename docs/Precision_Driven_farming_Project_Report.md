# Precision-Driven farming: AI Crop Disease Detection and Precision Spraying


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

Precision-Driven farming is an AI solution for the agriculture industry, the sector the 4IR has so far done least for in developing countries. Cameras on a drone or on fixed poles photograph maize and tomato leaves as the field is scanned. A deep learning model (MobileNetV3, trained on 11,069 expert-validated CCMT images from Ghana) sorts each leaf into one of 12 classes: nine different pests and diseases, or healthy. The system then turns those results into a precision spraying map: a 20 m grid that shows which zones need treatment, how urgently and at what dose, so that only those zones are sprayed. A time-series analysis of repeated scans warns of zones that are heading towards trouble before they get there, and a chatbot lets the farmer ask about the results in plain English. Nothing is sprayed until the farmer has checked and approved each zone. The model is light enough to run on a laptop or a drone's onboard computer, so the solution can reach small and emerging farmers, not only large commercial farms, which is exactly the gap between the promise of the 4IR and people on the ground that the theme describes.

## 2. Problem definition

*(246 words including the three question headings, limit 250; "What exactly is the problem?" 148 words, limit 190)*

Small-scale and commercial farmers have problems identifying crop diseases early and using pesticides in the right way. Crop diseases can spread quickly if they are not noticed early. This can damage crops, reduce the amount of food produced and cause farmers to lose money. Normally, farmers inspect their fields manually, which can take a lot of time, especially on large farms. It can also be difficult to check every plant properly. Farmers sometimes spray pesticides over the whole field even when only a small part of the field is affected. This can waste pesticides, increase farming costs and cause chemicals to enter the soil and nearby water sources. Because of this, farmers need a faster and more effective way to monitor crops and identify diseases early.

AI Benefit
Precision Driven Farming can help by using cameras on drones or fixed poles to take pictures of crops and leaves. Computer vision can then be used to analyse these images and identify possible crop diseases. The system can also create a spraying map that shows which areas may need treatment. This can help farmers use less pesticide, reduce costs and protect their crops. For the municipality and local community, healthier crops can help increase local food production and support farmers. Using less pesticide can also help reduce chemical runoff into nearby water sources and support a cleaner environment.

## 3. Business understanding

### 3.1 Business background

Maize is South Africa's staple food crop and tomato one of its main vegetable crops, and both are grown by large commercial farms and by small-scale farmers. Agriculture employed about 920,000 people in South Africa at the end of 2024 (Stats SA Quarterly Labour Force Survey), and the 2022 census counted about 2.4 million small-scale farmers, of whom only about 12% sell any produce. For these farmers a pest outbreak decides whether there is anything to sell.

Today, crop protection on most farms works like this: someone walks the rows, inspects leaves by eye, and if a problem is found, the whole field is sprayed. Commercial farms can pay agronomists and scouts; small farms usually cannot. Precision agriculture already shows what targeting can save: in a peer-reviewed maize trial, drone-guided spot spraying used 47% less herbicide than blanket spraying with the same weed control (Allmendinger et al., 2024). What has been missing is an affordable way to find *which* plants need treatment early enough to matter, for pests and diseases rather than weeds.

Our target customer is a maize and tomato farm of a few hectares to a few hundred hectares, from emerging farmers supported by an agricultural cooperative or the municipality's agricultural extension services, to commercial growers. The demo field in this project is a 60 m × 60 m maize block (a 3 × 3 grid of 20 m zones) at Vereeniging, in Emfuleni Local Municipality.

1.	Company / Project Background
•	Precision Driven Farming is conceptualized as an AgriTech start-up focused on small to medium commercial farms in the country. The business was founded to solve a simple problem observed on local farms
•	Farmers know they have a disease only when 40% of the crop is already infected.
•	The project leverages the 4th Agricultural Revolution (Agriculture 4.0) which combines AI, drones, and IoT. 
•	Unlike expensive solutions from John Deere or Trimble that cost >R400k and target US mega-farms
•	It is built for the African context: low-cost, works with a smartphone, and uses WhatsApp for alerts which every farmer already uses.

2.	Business Context & Problem Drivers
•	Economic Driver: Fertilizer & pesticide costs are farmer's largest expense at 35% of input costs.
•	Environmental Driver: South Africa's new Agricultural Pest Act pressures farmers to reduce chemical runoff
•	. Technological Driver: 92% of SA farmers now have smartphones and cheap drones (<R5k) are available.
•	Social Driver: Shortage of skilled agronomists - one agronomist serves 100+ farms, cannot visit often.

3.	Current Business Process (AS-IS)
•	Farmer notices yellowing/brown spots
•	Takes photo and sends to agronomist on WhatsApp
•	Waits 2-3 days for response
•	Agronomist visits, confirms Late Blight
•	Farmer sprays entire 100ha field
•	Result: Disease has spread, chemical wasted


4.	Proposed Business Process (TO-BE)
•	Drone / fixed poles cameras captures 100 images per 10ha daily
•	 AI automatically detects Early Blight at 5% severity
•	System generates Spray Map: Zone A=0L/ha, Zone B=50L/ha, Zone C=150L/ha (Critical
•	Farmer receives WhatsApp alert with map: "Spray Zone C within 24h"
•	Tractor with variable-rate sprayer executes map OR manual spot-spraying
•	Result: Infection contained at 5%, 65% chemical save



### 3.2 Business objectives

The main business objective of Precision Driven Farming is to help small to medium commercial farmers in South Africa detect crop diseases early and treat only the affected areas, so that crop losses, chemical costs and environmental damage are reduced. The specific objectives are:
1.	Detect disease early. Identify crop disease at an early stage (about 5% severity) instead of the current situation where farmers only discover disease once about 40% of the crop is infected.
2.	Reduce chemical use and cost. Replace blanket spraying with a zone-based spray map, so that pesticide is applied only where it is needed.
3.	Speed up decision-making. Deliver a disease alert and spray map to the farmer within 24 hours of image capture, instead of the 2 to 3 days currently spent waiting for an agronomist’s response.
4.	Make the solution affordable and accessible. Provide a low-cost system that works with a smartphone, low-cost drones and WhatsApp alerts, in contrast to existing solutions that cost over R400,000.
5.	Support, not replace, the farmer. Keep the farmer in control by showing confidence scores and requiring the farmer to verify results before spraying.
6.	Benefit the community and environment. Protect local food production and reduce chemical runoff into soil and nearby water sources.


### 3.3 Business success criteria

The project will be considered successful if the following measurable targets are met when the prototype is evaluated:
#	Criterion| Target| How it will be measured
1	Disease classification accuracy| At least 90% accuracy on unseen test images|	Held-out test set, confusion matrix
2	Missed detections|	Recall of at least 90% for diseased leaves|	Test set recall per disease class
3	Early detection|	Disease detected at an early stage severity|	Test images labelled by severity
4	Processing time|	Prediction within 10 seconds per image|	Timed runs of the model
5	Alert speed|	Alert and spray map delivered within 24 hours of capture|	End-to-end test of the workflow
6	Chemical reduction|	At least 50% less chemical than blanket spraying|	Comparison of spray-map area against whole-field area on sample fields
7	Affordability	Hardware |cost per farm under R10,000|	Cost breakdown of drone and equipment
8	Usability|	At least 80% of test users can read the result and spray map without help|	Short user test
9	Farmer verification	|Every spray recommendation can be reviewed before action	|Demonstration of the review step
10	Weather forecast|	forecast MAE at least [20]% lower than the seasonal naive baseline|	measured on the held-out final weeks


The model targets are deliberately high, and recall comes first: missing a sick plant costs more than a false alarm, because a missed outbreak spreads, while a false alarm is caught when the farmer reviews the zone. Tuning closed about half of the gap (accuracy from 75.2% to 84.3%). The remaining gap comes from look-alike lesion classes and from training on a laptop CPU, which allowed one tuning experiment of about three hours (section 5.5). Until the targets are met, the farmer-approval gate is what keeps a wrong result from becoming a wrong spray.

### 3.4 Requirements

**Functional requirements**

The system should be able to: 
1. Take pictures of crops using drones or fixed cameras. 
2. Process the images using Python and OpenCV. 
3. Find and analyse leaves in the images. 
4. Identify crop diseases using a PyTorch AI model. 
5. Identify the affected areas of the crop. 
6. Give disease predictions soon after an image is taken. 
7. Record the time and location where a disease is detected. 
8. Use GPS information to connect disease detections to areas in the field. 
9. Create a map showing areas where pesticide treatment may be needed. 
10. Show the results to farmers through a simple interface. 
11. Save previous results so farmers can compare crop health over time. 
12. Give a confidence score or warning when a disease is detected.
13. Allow the farmer to check the results before making a spraying decision. 
14. Forecast likely outbreak weeks



**Non-functional requirements**

Requirement 	Description 
Accuracy    	  The system should identify crop diseases correctly as much as possible. 
Performance 	  The system should process images quickly enough for near-real-time monitoring. 
Reliability 	  The system should work properly during normal crop monitoring. 
Scalability 	  The system should be able to work on both small and larger farms. 
Usability 	    The system should be simple enough for farmers to understand and use. 
Security   	    Farm information, images and user details should be protected from unauthorized access. 
Maintainability The software and AI model should be easy to update and fix when needed 
Compatibility 	The system should work with the chosen cameras, drones, Python, OpenCV and PyTorch. 
Availability    The system should be available when crop monitoring is being carried out. 
Explainability 	The system should show the detected disease and, where possible, how confident the AI is about the result. 

### 3.5 Constraints

1. Limited training data: It may be difficult to find enough good images of different crop diseases to train the AI. 

2. Image quality: Poor lighting, blurry pictures, shadows, wind and the position of leaves can affect the results. 
3. Hardware limitations: Drones and cameras have limits such as battery life, storage and processing power. 

4. Internet connection: Some farming areas may have weak or no internet connection. 

5.	Computing power: Training the AI model may require a powerful computer or GPU. 

6.	Weather: Rain, strong wind, fog and dust can make it difficult to collect good images. 

7.	Different crops: Different crop types and growth stages may make disease detection more difficult. 

8.	Cost: Drones, cameras, maintenance and computer equipment can be expensive. 

9.	Legal requirements: Drone use and pesticide spraying must follow the relevant laws and regulations. 

10.	AI limitations: The AI can make mistakes, especially when it sees images or diseases that were not included in its training data. 

11.	no public time-stamped disease data exists so the disease log will be simulated



### 3.6 Risks and contingencies

Risks and Mitigation |	Possible Impact |	How to Reduce the Risk 
Incorrect disease classification| The farmer could use the wrong treatment. |Train the AI using a large range of good-quality images and allow the farmer to check the result. 

False disease detection |	Pesticide may be used when it is not needed. |Use confidence levels and check uncertain results before spraying. 

Missed disease detection |	A disease could spread to other crops. |	Monitor crops regularly and improve the AI model over time. 

Poor image quality| The AI may give incorrect results. |	Use suitable cameras and check the quality of images. 

Drone failure |	Some areas may not be monitored.| Maintain the drone and have another monitoring method available. 

Battery runs out| 	The drone may not finish checking the field.| 	Plan the flight properly and have charged spare batteries. 

Biased training data |	The AI may not work well on some crops or environments.| 	Use different crops, diseases, environments and lighting conditions when training the model. 

Data security breach |	Farm information or images could be accessed by other people. |Use passwords, access controls and secure data storage. 

Bad weather |	Crop images may not be collected properly. |	Monitor weather conditions and only fly when conditions are suitable. 

Over-reliance on AI | Farmers could make a bad decision based only on the AI result. |Use the AI as a support tool and allow human checking before important decisions. 

Incorrect spraying map| Pesticide could be sprayed in the wrong location. |Check the GPS information and allow the farmer to verify the map. 

Regulatory problems | The project may break drone or pesticide regulations. | Make sure the system follows the relevant laws and regulations. 


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

## 8 Summary 

This project uses AI to solve a real problem in the farming industry. The system uses computer vision and deep learning to help farmers find crop diseases and identify areas that may need treatment. It does not replace the farmer but helps the farmer make better decisions. By using pesticides more carefully, farmers can reduce waste and costs while protecting their crops. The project can also benefit the local community by supporting food production, reducing chemical pollution and helping the local farming economy.


## 9. References

* Allmendinger, A., Spaeth, M., Saile, M., Peteinatos, G. G. and Gerhards, R. (2024). Agronomic and technical evaluation of herbicide spot spraying in maize based on high-resolution aerial weed maps: an on-farm trial. *Plants*, 13(15), 2164. <https://doi.org/10.3390/plants13152164>
* CABI (2018). Fall armyworm: impacts and implications for Africa, evidence note update (USD 1–4.6 billion per year across 12 countries), as reported in <https://www.allaboutfeed.net/animal-feed/raw-materials/progress-made-on-controlling-fall-armyworm>
* FAO (2021). Scientific review of the impact of climate change on plant pests (up to 40% of crop production lost annually), as reported by the World Economic Forum: <https://www.weforum.org/stories/2021/06/climate-change-insects-pests-crops-agriculture/>
* Howard, A., Sandler, M., Chu, G. *et al.* (2019). Searching for MobileNetV3. *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 1314–1324.
* Mensah, P. K., Akoto-Adjepong, V., Adu, K. *et al.* (2023). CCMT: Dataset for crop pest and disease detection. *Data in Brief*, 49, 109306. <https://doi.org/10.1016/j.dib.2023.109306>
* Statistics South Africa, Census 2022 agricultural households (2.4 million small-scale farmers, about 12% selling), as reported in <https://farmingportal.co.za/index.php/farming-news/south-africa/12409-south-africa-has-2-4-million-small-scale-farmers-but-only-12-sell-anything>
* Statistics South Africa, Quarterly Labour Force Survey Q4 2024 (agricultural employment about 920,000), as reported in <https://iol.co.za/business-report/opinion/2025-03-04-navigating-employment-fluctuations-in-south-africas-agriculture-sector/>
* University of the Witwatersrand (2021). Pollution of the Vaal River in Emfuleni Local Municipality (thesis). <https://wiredspace.wits.ac.za/handle/10539/31933>
* Disease and pest descriptions: see `app/chatbot/disease_info_sources.md` for the ten sources.
