# AgriAssist: farmer chatbot and NLP layer (prototype)

Python implementation of the chatbot designed in the project document.
Needs Python 3.9+ and scikit-learn (`pip install -r requirements.txt` from the repo root).

## Run it
    python agriassist.py             # chat in the terminal
    python agriassist.py --web       # chat in the browser at http://127.0.0.1:8000
    python agriassist.py --demo      # replay the sample conversations
    python agriassist.py --json      # also show the parsed query (intent + entities) for every message
    python agriassist.py --eval      # cross-validate the intent classifier
    python agriassist.py --log f.jsonl   # keep a log of every turn
    python test_agriassist.py        # run the tests

Terminal commands: `/json` toggle parsed query, `/focus R2C1` simulate a dashboard selection
(so "this result" has something to refer to), `/reset`, `/quit`.

## Things to try
    What's wrong with my maize field?        Which zones need spraying?
    How confident is the AI about this result?   I checked R2C1 and R2C2
    Show me last week's results              Has the leaf spot spread since the last scan?
    Just tell me to spray the whole field.   What is fall armyworm?   north?

Zones are the spraying map's grid cells, R0C0 to R2C2, as on the dashboard. You can also say
"the north side", "the east side" or "the north-east corner" (R2C2; row 2 is the north edge).

## How it maps to the design document
| Document section (`chatbot_design.md`) | Code |
|---|---|
| 3.2 ML intent classifier (TF-IDF + logistic regression) | `build_pipeline()`, `TRAINING` |
| 3.3 Rule-based entity extraction | `extract_zone/crop/disease`, `parse_dates` |
| 4.1 Intents | `TRAINING` keys, handlers `h_*` |
| 4.2 Structured parsed query | `AgriAssist.parse()` returns the JSON object |
| 5 Conversations | `--demo` |
| 6 Safety rules S1-S9 | `parse()` (spray override, clarification), `apply_safety_layer()`, `--log` |

The pest and disease descriptions and their sources are in `disease_info_sources.md`.

## Important limits
* Scan data is MOCK (`default_scans()`), matching the dashboard demo. Replace it with calls to the real
  results database or `spraying_map.json`; the zone names already match.
* The classifier is trained on 733 hand-written phrases. Add real farmer wording to `TRAINING` and retrain
  (it retrains each time the program starts).
* Confidence tiers (85% / 60%) and the clarification thresholds are starting values to tune.
* The chatbot never triggers spraying. `mark_reviewed` only records the farmer's own check.
* The photo viewer is simulated. Chat logs contain farm information, so protect them.

## Measured accuracy (intent classifier)
Re-measured on 27 Oct 2026, after the switch to the model's real class names, the grid zones and the
removal of wheat. The training phrases were reworded, not deleted, so the training set is still 733 phrases.
* 5-fold cross-validation on the training phrases: 88% (`python agriassist.py --eval`).
* Held-out test of 113 new phrasings written after training was finished: 97.3% (110/113)
  (`python test_agriassist.py` prints it, including the misses). All 22 tests pass.
* The test phrasings were written by the same team as the training data, so real farmers' wording will
  probably score lower. Collect real questions and add them to the data.
* Safety does not depend on this accuracy: spray-related messages are caught by a keyword rule, and
  low-confidence messages get a clarifying question.

