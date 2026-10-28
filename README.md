# Precision-driven-farming
An AI driven system that identifies diseases/infections on certain leaves and creates a spray map according to the severity of the infected plants to alert farmers as to which plants deserve immediate attention 


## Project Overview

**Theme:** AI Solution for Industries (Agriculture)

**Problem:** Farmers currently detect crop disease manually and late — by the time
symptoms are visible to the human eye, damage has already spread, remedial treatment
costs more (or fails entirely), and healthy crops nearby are often sprayed unnecessarily,
wasting chemicals and money.

**Solution:** An AI-powered vision system mounted on agricultural drones or fixed poles
that monitors crop health in real time. The system analyses leaf imagery, classifies crop
diseases as they emerge, and generates a precision spraying map that indicates exactly
which crops need immediate attention — reducing crop loss and cutting unnecessary
spending on remedies.

**Core capabilities:**
- Real-time leaf image capture via drone-mounted or fixed-pole cameras
- Deep learning-based disease classification
- Precision spraying map generation for targeted resource allocation
- Dashboard for monitoring crop health and alerts
- Chatbot/softbot for farmer queries on flagged crops and recommended treatment

## Repository Structure

```
/docs      → Project report (LeafSight_Project_Report.md, in rubric order), architecture, spraying-map design
/data      → Dataset documentation, EDA, sample scan history for the time-series analysis
/model     → Deep learning model: training, evaluation, export, live prediction demo, time-series analysis
/app       → Spraying pipeline, chatbot (AgriAssist)
farmer_dashboard.html, project_poster.html → dashboard and A1 poster (open in a browser)
```

## How to Run

Install once: `pip install -r requirements.txt`. Put the CCMT raw data in `data/Raw Data/Maize` and
`data/Raw Data/Tomato` (see `data/dataset-description.md`). Run everything from the repository root.

| Command | What it does |
|---|---|
| `python model/train_deploy_demo.py` | Train and evaluate the classifier; export `leaf_classifier.pt`; metrics, confusion matrix and curves in `model/outputs/`. Submitted runs: `model/outputs/tuned/` (deployed, 84.3% test accuracy) and `model/outputs/baseline/` (75.2%) |
| `python model/predict_demo.py --class "Fall armyworm"` | Live prediction on a held-out test image |
| `python app/spraying_pipeline.py` | Classify a batch of geotagged images and write `spraying_map.json` |
| `python model/time_series_analysis.py` | Crop-health time series per zone: trend, forecast, early warning, treatment check |
| `python app/chatbot/agriassist.py --web` | Farmer chatbot in the browser (`python app/chatbot/test_agriassist.py` runs its tests) |

## Team Roster

| Team | Members | Responsibilities |
|---   |---      |---               |
| **Leader** | **NhlanhlaRikhotso** | Coordination, GitHub Project board, monitoring, risk log, presentation |
| **Team 1 – Foundation** | **malokampho**, **Buhle-03-coder**, **J_Sizwe** | Business objectives, requirements/constraints/risks, problem definition, dataset documentation |
| **Team 2 – AI Model** | **thamahalehlohonolo7**, **Kamogelo1932**, **Himothy1ofWun** | Computer vision model, deep learning training pipeline, model evaluation & time-series analysis |
| **Team 3 – Application** | **Pfumiy24**, **Sanel32004**, **waltonmoichela** | Chatbot/NLP, precision spraying map logic, dashboard UI, poster, documentation QA |


## Project Timeline

| Phase | Dates | Milestone |
|---    |---    |---        |
| 1. Kickoff | Aug 22–31 | Repo & board setup, initial drafts |
| 2. Foundations | Sep 1–14 | Data exploration, first model prototype, requirements doc |
| 3. Build | Sep 15–30 | Model training, spraying map algorithm, chatbot skeleton |
| 4. Integrate | Oct 1–15 | Full pipeline integration, evaluation metrics |
| 5. Polish | Oct 16–25 | Testing, poster, Grammarly pass, formatting |
| 6. Finalize | Oct 26–Nov 2 | Rehearsal, final review, submission |

**Due date:** 02 November 2026, 23:59
**Presentations:** 09–13 November 2026

## Board Setup

The GitHub Project board uses:
- **Status:** Backlog → In Progress → In Review → Done
- **Team:** Team 1 – Foundation / Team 2 – AI Model / Team 3 – Application / Leader
- **Labels:** `team-1-foundation`, `team-2-model`, `team-3-application`, `blocked`, `needs-review`
