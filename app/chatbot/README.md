# AgriAssist: farmer chatbot and NLP layer (prototype)

Python implementation of the chatbot designed in the project document.
Needs Python 3.9+ and scikit-learn (`pip install -r requirements.txt`).

## Run it
    python agriassist.py             # chat in the terminal
    python agriassist.py --web       # chat in the browser at http://127.0.0.1:8000
    python agriassist.py --demo      # replay the sample conversations
    python agriassist.py --json      # also show the parsed query (intent + entities) for every message
    python agriassist.py --eval      # cross-validate the intent classifier
    python agriassist.py --log f.jsonl   # keep a log of every turn
    python test_agriassist.py        # run the tests

Terminal commands: `/json` toggle parsed query, `/focus East C` simulate a dashboard selection
(so "this result" has something to refer to), `/reset`, `/quit`.

## Things to try
    What's wrong with my maize field?        Which zones need spraying?
    How confident is the AI about this result?   I checked North A and B
    Show me last week's results              Has the rust spread since the last scan?
    Just tell me to spray the whole field.   What is gray leaf spot?   north?

## How it maps to the design document
| Document section | Code |
|---|---|
| 3.5 ML intent classifier (TF-IDF + logistic regression) | `build_pipeline()`, `TRAINING` |
| 3.6 Rule-based entity extraction | `extract_zone/crop/disease`, `parse_dates` |
| 4 Structured parsed query | `AgriAssist.parse()` returns the JSON object |
| 5 Conversations | `--demo`, handlers `h_*` |
| 6 Safety rules S1-S9 | `parse()` (spray override), `apply_safety_layer()`, `--log` |

## Important limits
* Scan data is MOCK (`default_scans()`). Replace it with calls to the real results database.
* The classifier is trained on ~400 hand-written phrases. Add real farmer wording to `TRAINING` and retrain
  (it retrains each time the program starts).
* Confidence tiers (85% / 60%) and the clarification thresholds are starting values to tune.
* The chatbot never triggers spraying. `mark_reviewed` only records the farmer's own check.
* The photo viewer is simulated. Chat logs contain farm information, so protect them.
