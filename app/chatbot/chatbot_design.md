# Role 8 — NLP and Farmer Chatbot Design

This document describes the chatbot as implemented in `agriassist.py`. Section numbers are the
ones the README's mapping table refers to.

## 1. Purpose

The Precision-Driven-Farming project uses an NLP-based farmer chatbot to provide a simple interface between the farmer and the AI agricultural analysis system.

The chatbot allows farmers to ask questions about crop health, detected pests and diseases, AI confidence, affected locations, historical results, trends and the precision spraying map.

The chatbot is designed as a decision-support tool. It does not authorise or trigger pesticide spraying.

## 2. Objectives

The objectives of the NLP and chatbot component are:

* Allow farmers to communicate with the system using natural language.
* Classify farmer questions into predefined intents.
* Provide understandable responses about AI crop analysis.
* Display information about detected pests and diseases and affected areas.
* Provide AI confidence information.
* Provide historical crop health information.
* Support access to the precision spraying map.
* Explain why an area was flagged by the AI system.
* Require farmer verification before important spraying decisions.

## 3. NLP Approach

The chatbot uses a lightweight machine-learning approach for intents and simple rules for details, so it runs on a laptop with no internet connection.

### 3.1 Pipeline

Farmer question
→ text preprocessing (lower case, punctuation removed, every grid cell name replaced by the word "zone")
→ **intent classification** (3.2)
→ **entity extraction** (3.3)
→ clarification if the question is unclear or a required detail is missing
→ structured parsed query (section 4.2)
→ data lookup (scan results, spraying map, disease information)
→ reply template
→ **safety layer** (section 6) applied to every reply.

### 3.2 ML intent classifier

TF-IDF features over word 1–2-grams and character 2–4-grams (the character features cope with spelling mistakes such as "sprayin"), followed by Logistic Regression. It is trained on 733 hand-written example phrases (`TRAINING` in the code) and retrains each time the program starts. The classifier returns an intent and a probability; that probability is the chatbot's **NLP confidence** (section 7).

Zone names are replaced by one shared word before classification, because which zone a farmer names says nothing about what they want to know.

### 3.3 Rule-based entity extraction

Small fixed vocabularies, applied with regular expressions:

| Entity | Values | Examples |
|---|---|---|
| crop | maize, tomato | "corn", "mealies", "tomatoes" |
| zone | the spraying-map grid cells R0C0 to R2C2, or a direction | "R2C2", "the north side" (row 2), "the east side" (column 2), "the north-east corner" (R2C2), "the centre" (R1C1) |
| disease | the 10 farmer-facing names of the model's 12 classes | "fall armyworm", "septoria", "grasshoper" (dataset spelling), "early blight" (→ Leaf blight) |
| date_range | absolute and relative dates | "yesterday", "last week", "3 October", "the last 14 days" |

The zones are the same 20 m × 20 m grid cells the spraying pipeline and the dashboard use, counted northwards from the field origin: row 0 is the south edge and row 2 the north edge, which the dashboard draws at the top. Farmers can use the grid name or plain directions.

The disease list is ordered from most to least specific, because the first match wins: "septoria leaf spot" is checked before "leaf spot", and "leaf blight" before "blight".

## 4. Chatbot Intents and Parsed Query

### 4.1 Intents

The chatbot supports 16 intents. The right-hand column gives the name used in the first version of this document, where one existed.

| Intent | Purpose | Earlier name |
|---|---|---|
| identify_disease | Which pest or disease the AI found | disease_information |
| field_health_summary | General health of the field or a crop | crop_health_status |
| disease_location | Where a problem was detected | disease_location |
| get_affected_area | How much of a zone is affected | (new) |
| show_spray_map | Which zones are flagged on the spraying map | show_spray_map |
| request_treatment_advice | Whether or what to spray (answered with the safety response) | verify_spray_decision |
| get_confidence | How confident the AI is | show_confidence |
| explain_result | Why a zone was flagged, and on what evidence | explain_alert |
| mark_reviewed | Farmer records that they checked a zone | (new) |
| flag_incorrect | Farmer reports a result as wrong | (new) |
| show_history | Results for a past period | historical_results |
| compare_over_time | Whether things are getting better or worse | health_trend |
| get_disease_info | What a pest or disease is (sourced descriptions) | (new) |
| scan_status | When the field was last scanned and how much was covered | (new) |
| smalltalk_help | Greetings and "what can you do?" | (new) |
| out_of_scope | Anything outside crop scan results | (new) |

### 4.2 Structured parsed query

Every message is turned into a JSON object before any data is looked up. For "How confident is the AI about R2C1?":

```json
{
  "query_id": "q-0000",
  "timestamp": "2026-10-10T08:15:02",
  "raw_text": "How confident is the AI about R2C1?",
  "input_mode": "text",
  "intent": "get_confidence",
  "intent_confidence": 0.97,
  "entities": {"crop": null, "zone": "R2C1", "disease": null, "date_range": null},
  "context_ref": null,
  "needs_clarification": false,
  "clarification_prompt": null,
  "safety_flags": ["requires_confidence_check"],
  "requires_farmer_verification": true,
  "route_to": "results_db"
}
```

`context_ref` is filled when the farmer says "this result" while a zone is selected on the dashboard. `route_to` names the data source the question needs.

## 5. Example Conversations

`python agriassist.py --demo` replays six conversations: diagnosis, spray map and review, confidence (with a zone selected on the dashboard), history, pressure to skip verification, and an unclear question. A short example:

Farmer: "Which zones need spraying?"

Chatbot: "2 zones are flagged as possibly needing treatment in the latest scan (today, 07:42): R2C1: Leaf spot, 82% confidence (medium), pending your review. R2C2: Fall armyworm, 89% confidence (high), pending your review. The spray map stays marked Pending your review until you confirm each zone. Nothing is acted on automatically." This is followed by the safety notes from section 6.

The scan data in the prototype is mock data matching the dashboard demo (`default_scans()`).

## 6. Safety and Human Verification

The chatbot must not replace the farmer's judgement. These rules are applied in code, mostly in `apply_safety_layer()`, to every reply:

| Rule | What the chatbot does |
|---|---|
| S1 Confidence first | Every reply about a result states the AI's confidence for each zone. |
| S2 Verify before action | Every reply about a result asks the farmer to check the photos and the plants before deciding. |
| S3 Never instruct spraying | Asked whether or what to spray, the chatbot says it can't decide that and shows the results instead. Any message with a spray-related word gets this response, even if the classifier misread it. |
| S4 Confidence tiers | High (85% or more), medium (60–85%) and low (under 60%). Medium and low zones get an explicit warning to inspect them in the field. |
| S5 Ask when unsure | If the NLP confidence is under 35%, or a needed detail is missing, the chatbot asks a clarifying question. Recording a review or a flag needs at least 55%. |
| S6 No products or doses | The chatbot never names a product, amount or mixing rate, and points to the product label, local regulations or an agronomist. |
| S7 Honest about gaps | It says when the latest scan is old, when part of the field was not scanned, and when photos were blurry. |
| S8 Recording is not acting | "I checked R2C2" only records the farmer's check; flags only go to a review list for the team. Nothing triggers spraying. |
| S9 Keep a record | With `--log`, every turn and its parsed query are written to a log file. |

For example:

Farmer: "Should I spray this area?"

Chatbot: "I can't tell you whether or what to spray. That decision needs your own check of the AI results." It then lists the detections with their confidence, asks the farmer to review the spray map and inspect medium- or low-confidence zones, and adds the product-label note.

## 7. Confidence Scores

Two different confidence values are used.

### NLP confidence

How confident the chatbot is that it understood the farmer's question correctly (the classifier probability, `intent_confidence`). It decides whether to ask a clarifying question (S5).

### Computer vision confidence

How confident the disease detection model is in its prediction for a zone. It is what the farmer sees in replies (S1) and what sets the tier warnings (S4).

These two confidence values must not be confused.

## 8. Low-Confidence Handling

If the computer vision confidence is medium or low, the chatbot does not encourage the farmer to act. It asks them to inspect the zone in the field or ask an agronomist (S4). If the NLP confidence is low, it asks what the farmer meant rather than guessing (S5).

This reduces the risk of over-reliance on the AI system.

## 9. Integration

The intended system flow is:

Drone / fixed camera
→ image processing
→ computer vision model (`model/train_deploy_demo.py`, exported as `leaf_classifier.pt`)
→ pest/disease + confidence per image, with GPS and time
→ spraying pipeline (`app/spraying_pipeline.py`): severity per 20 m grid cell, written to `spraying_map.json`
→ results storage
→ farmer chatbot: intent classification and entity extraction
→ relevant result, map or historical data
→ farmer.

In the prototype, `default_scans()` stands in for results storage. To connect the real system, replace it with reads from the results database or `spraying_map.json`; the zone names already match.

## 10. Testing

`python test_agriassist.py` runs 22 automated tests covering the safety rules, entity extraction (including every model class name), date handling, clarification, state changes and intent accuracy on 113 held-out phrasings that were not used for training. The questions from the first version of this document classify as follows:

| Farmer question | Intent |
|---|---|
| How confident is the AI? | get_confidence |
| Show me last week's results | show_history |
| Which zones need spraying? | show_spray_map |
| Where was the disease detected? | disease_location |
| Is the disease getting worse? | compare_over_time |
| Why was this area flagged? | explain_result |
| Should I spray now? | request_treatment_advice |

Measured accuracy is in the README.

## 11. Limitations

The chatbot depends on the quality and quantity of the training examples, which were written by the team, not by farmers.

It may not correctly understand questions that are very different from the training data.

The scan data is mock data until the chatbot is connected to the results database, and the dashboard photo viewer is not connected.

The chatbot also depends on the availability of results from the computer vision component of the overall system.

The chatbot therefore acts as a support interface and does not independently diagnose crops or authorise pesticide application.

## 12. Conclusion

The NLP and chatbot component provides a farmer-friendly interface for accessing the Precision-Driven-Farming AI system.

The use of intent classification allows natural-language farmer questions to be mapped to relevant system functions, and entity extraction ties them to the same zones and classes the rest of the system uses.

The chatbot also supports responsible AI use by requiring human verification for important spraying decisions.
