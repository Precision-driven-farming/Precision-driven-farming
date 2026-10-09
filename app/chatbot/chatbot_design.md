# Role 8 — NLP and Farmer Chatbot Design

## 1. Purpose

The Precision-Driven-Farming project uses an NLP-based farmer chatbot to provide a simple interface between the farmer and the AI agricultural analysis system.

The chatbot allows farmers to ask questions about crop health, detected diseases, AI confidence, affected locations, historical results, disease trends and precision spraying maps.

The chatbot is designed as a decision-support tool. It does not automatically authorise pesticide spraying.

## 2. Objectives

The objectives of the NLP and chatbot component are:

* Allow farmers to communicate with the system using natural language.
* Classify farmer questions into predefined intents.
* Provide understandable responses about AI crop analysis.
* Display information about detected diseases and affected areas.
* Provide AI confidence information.
* Provide historical crop health information.
* Support access to precision spraying maps.
* Explain why an area was flagged by the AI system.
* Require farmer verification before important spraying decisions.

## 3. NLP Approach

The chatbot uses a lightweight machine-learning NLP approach.

The NLP pipeline is:

Farmer question

↓

Text preprocessing

↓

TF-IDF vectorisation

↓

Logistic Regression intent classification

↓

Intent and confidence score

↓

Chatbot response

The system uses TF-IDF to convert text into numerical features. Logistic Regression is then used to classify the farmer's question into one of the supported intents.

## 4. Chatbot Intents

The chatbot supports the following intents:

| Intent                | Purpose                                                 |
| --------------------- | ------------------------------------------------------- |
| crop_health_status    | Ask about the general health of a crop or field         |
| disease_information   | Ask which disease was detected                          |
| show_confidence       | Ask how confident the AI prediction is                  |
| show_spray_map        | Request the precision spraying map                      |
| historical_results    | Ask about previous results                              |
| health_trend          | Ask whether disease levels are increasing or decreasing |
| disease_location      | Ask where the disease was detected                      |
| explain_alert         | Ask why an area was flagged                             |
| verify_spray_decision | Ask whether spraying should take place                  |

## 5. Example

Farmer:

"How confident is the AI?"

NLP result:

```text
Intent: show_confidence
```

The chatbot then explains that the confidence score represents how confident the computer vision model is in its disease prediction.

## 6. Safety and Human Verification

The chatbot must not replace the farmer's judgement.

For spraying-related questions, the chatbot informs the farmer that the AI result must be reviewed before spraying.

For example:

Farmer:

"Should I spray this area?"

Chatbot:

"The AI recommendation must be reviewed and verified by the farmer before spraying. The chatbot does not automatically authorise spraying."

This supports the project requirement that the farmer must check AI results before spraying.

## 7. Confidence Scores

Two different confidence values are considered.

### NLP confidence

NLP confidence indicates how confident the chatbot is that it understood the farmer's question correctly.

### Computer vision confidence

Computer vision confidence indicates how confident the disease detection model is in its prediction.

These two confidence values must not be confused.

## 8. Low-Confidence Handling

If the computer vision confidence is low, the chatbot should not encourage the farmer to act automatically.

The system can instead indicate that the result requires additional review.

This reduces the risk of over-reliance on the AI system.

## 9. Integration

The intended system flow is:

Drone / Fixed Camera

↓

Image Processing

↓

Computer Vision Disease Model

↓

Disease + Confidence + GPS + Time

↓

Results Storage

↓

Farmer Chatbot

↓

NLP Intent Classification

↓

Relevant Result / Map / Historical Data

↓

Farmer

The chatbot therefore acts as an interface to the outputs produced by the wider Precision-Driven-Farming AI system.

## 10. Testing

The following questions are used to test the chatbot:

| Farmer Question                 | Expected Intent       |
| ------------------------------- | --------------------- |
| How confident is the AI?        | show_confidence       |
| Show me last week's results     | historical_results    |
| Which zones need spraying?      | show_spray_map        |
| Where was the disease detected? | disease_location      |
| Is the disease getting worse?   | health_trend          |
| Why was this area flagged?      | explain_alert         |
| Should I spray now?             | verify_spray_decision |

## 11. Limitations

The chatbot depends on the quality and quantity of the training examples.

It may not correctly understand questions that are very different from the training data.

The chatbot also depends on the availability of results from the computer vision component of the overall system.

The chatbot therefore acts as a support interface and does not independently diagnose crops or authorise pesticide application.

## 12. Conclusion

The NLP and chatbot component provides a farmer-friendly interface for accessing the Precision-Driven-Farming AI system.

The use of intent classification allows natural-language farmer questions to be mapped to relevant system functions.

The chatbot also supports responsible AI use by requiring human verification for important spraying decisions.
