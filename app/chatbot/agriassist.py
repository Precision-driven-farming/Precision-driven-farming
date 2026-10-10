#!/usr/bin/env python3
"""
AgriAssist: farmer chatbot / NLP layer for the crop disease monitoring system.

Pipeline (matches the design document):
  text -> preprocessing -> ML intent classifier (TF-IDF + logistic regression)
       -> rule-based entity extraction (crop, zone, disease, date_range)
       -> slot filling / clarification -> structured query (JSON)
       -> data retrieval (MOCK data here) -> reply templates
       -> rule-based safety layer (confidence first, verify before action)

The scan data below is MOCK data. In the full system, replace the
`default_scans()` function with calls to the results database / spraying-map
module. Nothing in this file triggers spraying: the chatbot only reports the
AI results and records the farmer's own review.

Usage:
  python agriassist.py            interactive chat in the terminal
  python agriassist.py --demo     replay the sample conversations
  python agriassist.py --eval     cross-validate the intent classifier
  python agriassist.py --web      small browser chat on http://127.0.0.1:8000
  python agriassist.py --json     also print the parsed query for every message
  python agriassist.py --log f    append a JSON-lines log of every turn to f
"""
import argparse
import json
import re
import sys
import textwrap
from datetime import date, datetime, timedelta

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

# --------------------------------------------------------------------------
# Settings (tune on validation data)
# --------------------------------------------------------------------------
CLARIFY_THRESHOLD = 0.35   # below this intent probability, ask a question
STATE_THRESHOLD = 0.55     # intents that record something (review, flag) need more certainty
TIER_HIGH = 0.85           # confidence tiers (Table 7 in the design document)
TIER_MEDIUM = 0.60

# --------------------------------------------------------------------------
# Training data for the intent classifier (hand-written examples)
# --------------------------------------------------------------------------
TRAINING = {
    "identify_disease": [
        "what's wrong with my maize field", "what is wrong with my crops",
        "is there any disease in the north field", "what disease did the AI find",
        "what did the AI find on my tomatoes", "are my plants sick",
        "do my crops have a disease", "which disease is affecting my maize",
        "what is the problem with my wheat", "is something wrong with the leaves",
        "any disease detected", "what did the scan find", "is my maize infected",
        "what is attacking my tomatoes", "tell me what's wrong with the crop",
        "disease check on my field", "what illness do my plants have",
        "whats wrong with the maize", "is there rust in my field",
        "did the system detect any disease", "diagnose my field",
        "what is wrong with the wheat",
    ],
    "field_health_summary": [
        "how is my field doing", "give me an overview of crop health",
        "is everything okay with the wheat", "how are my crops",
        "crop health summary", "how healthy is my farm", "overall status of the fields",
        "how is the maize looking", "are my crops healthy", "give me a summary",
        "field status", "how are things on the farm", "is the north field ok",
        "health report please", "how did the field look in the latest scan",
        "overview", "is my tomato healthy", "quick summary of all zones",
        "how's the farm", "what is the general condition of my crops",
        "how is the east zone doing", "north", "north?", "east zone", "what about east c",
        "and the south", "west field", "north a", "centre", "what about the north field",
        "how is everything looking", "status of all my crops", "give me the big picture", "how are the tomatoes doing", "is the wheat okay", "what is the state of the farm", "overall crop condition", "farm summary please",
    ],
    "disease_location": [
        "where exactly is the disease", "which part of the field has rust",
        "show me where the problem is", "where was the disease found",
        "which zones are infected", "where are the sick plants",
        "location of the infected plants", "where did the AI detect disease",
        "which area has gray leaf spot", "where is the blight",
        "point me to the diseased area", "which zones have disease",
        "where are the affected plants", "in which part of the field is the problem",
        "where was it detected", "show infected zones",
        "map of the infected spots", "where is the rust",
        "what part of the field is sick", "where do I find the disease",
        "which blocks have disease", "where are the infected zones", "locate the disease", "which areas are sick", "where did you find the rust", "tell me where the infection is", "which zones show disease", "where are the problem spots",
    ],
    "get_affected_area": [
        "how much of the north field is affected", "what percentage is infected",
        "how big is the infected area", "how many hectares are affected",
        "how much of my crop is sick", "what share of the field is diseased",
        "how bad is it", "how severe is the infection", "how widespread is the disease",
        "what area is affected", "percentage of the field with disease",
        "how much of the maize is infected", "how large is the affected part",
        "extent of the damage", "how much of zone b is sick",
        "is it a small area or a big one", "how many plants are affected",
        "size of the infected area", "how much damage is there",
        "what fraction of the field is affected",
        "how much is infected", "what portion of the field is diseased", "how many hectares have disease", "how far gone is the crop", "what is the infected percentage", "how large is the problem area",
    ],
    "show_spray_map": [
        "which zones need spraying", "show me the spray map",
        "where might I need to treat", "spray map please",
        "which areas need treatment", "show the treatment map",
        "what areas should be sprayed", "map of areas to spray",
        "zones flagged for spraying", "where do I need to spray",
        "open the spraying map", "which parts of the field need pesticide",
        "spray zones", "spray map north", "show spraying zones",
        "where is spraying needed", "treatment zones",
        "which blocks need to be sprayed", "give me the spray plan",
        "areas needing treatment", "show me the spraying map for the east zone",
    ],
    "request_treatment_advice": [
        "what should I spray", "should I spray the whole field",
        "just tell me to spray", "what pesticide should I use",
        "which chemical do I use", "how much pesticide do I need",
        "what dose should I apply", "tell me what to spray",
        "should I spray today", "do I need to spray", "can I spray now",
        "is it time to spray", "what fungicide works",
        "how do I treat this disease", "what is the treatment",
        "how should I treat the maize", "which product should I buy",
        "just tell me what to do", "should I spray everything",
        "what is the best spray", "how much should I spray",
        "is it ok to spray", "is it safe to use chemicals", "tell me to spray",
    ],
    "get_confidence": [
        "how confident is the AI about this result", "how sure are you",
        "can I trust this", "how accurate is this result",
        "what is the confidence level", "how certain is the AI",
        "is the AI sure", "confidence score please", "how reliable is this",
        "how likely is it that this is correct", "are you sure about this",
        "what's the confidence", "how much can I trust the detection",
        "is this result reliable", "how confident is the system",
        "what percentage sure is the AI", "confidence for the north zone",
        "is the AI certain about rust", "can I rely on this result",
        "how good is this detection",
        "how certain are you about east c", "what is the confidence for east c", "is that result trustworthy", "how sure is the system about this", "how accurate is the AI here", "give me the confidence",
    ],
    "explain_result": [
        "why does the AI say it is rust", "show me the photo you used",
        "why was this zone flagged", "what did the AI look at",
        "show me the leaf pictures", "explain this result",
        "how did the AI decide", "show me the images", "show the photos",
        "what does the AI see in the photo", "why do you think it is blight",
        "which photo shows the disease", "let me see the leaf images",
        "what evidence is there", "why is east c flagged", "open the photos",
        "show me the evidence", "what made the AI think that",
        "see the pictures of the sick leaves", "explain why",
        "why is north a flagged", "show me what the AI saw", "why did you say gray leaf spot", "show the leaf pictures for east c",
    ],
    "mark_reviewed": [
        "I've checked North A and it looks right", "mark the east zone as reviewed",
        "I reviewed north b", "I have checked the map", "confirm I checked zone a",
        "I checked it, it looks correct", "reviewed east c",
        "mark north a as checked", "I verified the north zones",
        "the map looks right to me", "approve the north zone",
        "I've inspected east c in the field", "I confirm the results for north a",
        "set north b to reviewed", "done checking the photos",
        "I checked all zones", "I looked at the photos and they match",
        "I walked the field and it matches", "checked north a and b",
        "the detection in east c is right",
        "I have inspected north a", "reviewed and confirmed", "I checked the photos for east c", "north b is confirmed by me", "I double checked north a and it is right", "tick off north b as checked", "I have verified east c", "record that I reviewed the north zone",
    ],
    "flag_incorrect": [
        "that leaf is healthy", "this result is wrong", "the map is in the wrong place",
        "the AI is wrong", "flag this result", "that's not rust",
        "this is incorrect", "report a mistake", "there is no disease there",
        "the detection in north a is wrong", "that is a false alarm",
        "the AI got it wrong", "flag as incorrect", "this isn't right",
        "wrong disease", "the plants there are fine", "I disagree with this result",
        "mark this as a mistake", "the zone is healthy", "that diagnosis is not right",
        "the location is off",
        "the AI made a mistake in north b", "that detection is a mistake", "no that is wrong", "the result for east c is incorrect", "it is not rust it is healthy", "please flag the north a result", "this detection is false", "the AI missed the mark here",
    ],
    "show_history": [
        "show me last week's results", "what did the AI find yesterday",
        "results from 3 october", "show me previous results", "past results please",
        "scan results from last month", "what were the results two days ago",
        "history of scans", "show me results for this week", "old results",
        "results of the last 14 days", "what did the scan show last week",
        "show yesterday's scan", "earlier results", "previous scans",
        "show results from 1 october", "what was found last week",
        "show me the last month's detections", "scan history",
        "results from the past week",
    ],
    "compare_over_time": [
        "is the north field better than last month", "has the rust spread since the last scan",
        "is it getting worse", "compare this scan with the previous one",
        "is the disease spreading", "has it improved since last week",
        "how does it compare with last week", "trend over time",
        "is the situation improving", "compare to last month",
        "has anything changed since the last scan", "is the maize better than before",
        "is it better or worse than last week", "change since last scan",
        "did the disease get worse", "how has crop health changed",
        "is it spreading to other zones", "progress over the last weeks",
        "compare now with last month", "has the infection grown",
        "is it better than last week", "has it got worse since the last scan", "how has it changed", "compare with the previous scan", "is the infection spreading or shrinking",
    ],
    "get_disease_info": [
        "what is gray leaf spot", "how does common rust spread",
        "tell me about early blight", "what causes rust in maize",
        "explain gray leaf spot", "what does blight look like",
        "how does the disease spread", "what are the symptoms of rust",
        "what conditions does gray leaf spot like", "info on early blight",
        "what is common rust", "is gray leaf spot dangerous",
        "tell me more about the disease", "how does blight affect tomatoes",
        "what are the signs of rust", "what is a fungal leaf disease",
        "learn about gray leaf spot", "what is blight", "describe rust",
    ],
    "scan_status": [
        "when was the field last scanned", "did the drone cover the whole field",
        "when was the last scan", "did the scan finish", "was the whole field checked",
        "when will the next scan happen", "is the drone flying today",
        "how much of the field was scanned", "did the drone finish",
        "last scan time", "was anything missed in the scan", "scan coverage",
        "when did the drone last fly", "how many photos were taken",
        "was the scan complete", "did the drone run out of battery",
        "latest scan date", "is the data up to date", "how old is this data",
        "did the cameras work",
    ],
    "smalltalk_help": [
        "hi", "hello", "hey", "good morning", "help", "what can you do",
        "how do I use this", "who are you", "what can I ask you", "thanks",
        "thank you", "ok thanks", "bye", "what are your features",
        "can you help me", "hello there", "show me what you can do",
        "i need help", "menu", "start",
        "good afternoon", "good evening", "what do you do", "hi there", "howzit", "thanks a lot",
    ],
    "out_of_scope": [
        "what is the maize price today", "will it rain tomorrow",
        "order me more pesticide", "tell me a joke", "who won the soccer game",
        "what is the capital of france", "book me a flight", "how do I cook maize",
        "what's the weather", "buy a new drone", "what time is it in london",
        "play some music", "what is the meaning of life", "write me a poem",
        "how do I fix my tractor", "what is the exchange rate",
        "recommend a movie", "how tall is mount everest",
        "send an email to my supplier", "how do I get a loan for seeds",
        "when should I plant maize", "how much does a tractor cost", "what fertiliser is best for wheat", "tell me the news", "how do I register my farm", "where can I buy seed", "what is the price of wheat", "translate this to zulu", "what is five plus five", "who is the president", "what is the date today", "i am hungry", "my phone is broken", "give me a recipe", "how many people live in africa", "can you call my husband", "what is a good name for a dog", "is it going to be hot this week", "how do I irrigate better", "tell me about the stock market",
    ],
}

VERIFY_INTENTS = {
    "identify_disease", "field_health_summary", "disease_location", "get_affected_area",
    "show_spray_map", "request_treatment_advice", "get_confidence", "explain_result",
    "show_history", "compare_over_time",
}
ROUTES = {
    "identify_disease": "results_db", "field_health_summary": "dashboard",
    "disease_location": "spray_map", "get_affected_area": "results_db",
    "show_spray_map": "spray_map", "request_treatment_advice": "spray_map",
    "get_confidence": "results_db", "explain_result": "dashboard",
    "mark_reviewed": "spray_map", "flag_incorrect": "feedback_log",
    "show_history": "results_db", "compare_over_time": "results_db",
    "get_disease_info": "knowledge_base", "scan_status": "scan_log",
    "smalltalk_help": "chatbot", "out_of_scope": "chatbot",
}

# --------------------------------------------------------------------------
# Intent classifier
# --------------------------------------------------------------------------
def preprocess(text):
    t = text.lower().replace("\u2019", "'")
    t = re.sub(r"[^a-z0-9'\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def build_pipeline():
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)),
    ])
    return Pipeline([("tfidf", features),
                     ("clf", LogisticRegression(C=20, max_iter=3000))])


def training_arrays():
    X, y = [], []
    for intent, examples in TRAINING.items():
        for ex in examples:
            X.append(preprocess(ex))
            y.append(intent)
    return X, y


def train_classifier():
    X, y = training_arrays()
    return build_pipeline().fit(X, y)

# --------------------------------------------------------------------------
# Farm configuration and MOCK scan data
# --------------------------------------------------------------------------
ZONES = {
    "North A": {"crop": "maize", "ha": 4.0},
    "North B": {"crop": "maize", "ha": 3.5},
    "East C": {"crop": "maize", "ha": 3.0},
    "South A": {"crop": "maize", "ha": 2.5},
    "West A": {"crop": "wheat", "ha": 4.5},
    "South B": {"crop": "wheat", "ha": 5.0},
    "West B": {"crop": "tomato", "ha": 1.5},
    "Centre": {"crop": "tomato", "ha": 2.0},
}
CROPS = {"maize": r"maize|corn|mealies?", "wheat": r"wheat", "tomato": r"tomato(?:es)?"}
DISEASES = [
    ("gray leaf spot", r"gr[ae]y leaf spot"),
    ("common rust", r"common rust|\brust\b"),
    ("early blight", r"early blight|\bblight\b"),
]
DISEASE_INFO = {
    "gray leaf spot": "Gray leaf spot is a fungal disease of maize. It typically shows as narrow, "
                      "rectangular tan to gray lesions between the leaf veins. It is favoured by warm, "
                      "humid, wet conditions and by maize residue left in the field.",
    "common rust": "Common rust is a fungal disease of maize. It appears as small, powdery, "
                   "reddish-brown pustules on both leaf surfaces and is favoured by cool, humid "
                   "conditions. Its spores are carried by wind, so it can spread between fields.",
    "early blight": "Early blight is a fungal disease of tomato and potato. It usually starts on older "
                    "leaves as dark brown spots with concentric rings, often with a yellow halo, and "
                    "can move upward through the plant. It is favoured by warm, humid weather.",
}


def default_scans(today):
    """MOCK scan history, newest first. Replace with real results-database calls."""
    def det(disease, conf, pct, second=None, second_conf=0.0, blurry=0):
        return {"disease": disease, "conf": conf, "affected_pct": pct,
                "second": second, "second_conf": second_conf, "photos": 9, "blurry": blurry}
    g = "gray leaf spot"
    # Dates are anchored to last week's Monday so "last week" always shows the same story.
    last_mon = today - timedelta(days=today.weekday()) - timedelta(days=7)
    off = lambda d: (today - d).days
    spec = [
        (0, "07:42", [], None, {
            "North A": det(g, .93, 18, "common rust", .04),
            "North B": det(g, .88, 12, "common rust", .06),
            "East C": det(g, .64, 7, "common rust", .22, blurry=2)}),
        (off(last_mon + timedelta(days=5)), "07:51", [], None, {          # Saturday
            "North A": det(g, .90, 12), "North B": det(g, .84, 8), "East C": det(g, .58, 3)}),
        (off(last_mon + timedelta(days=3)), "07:20", ["West A"], "the drone battery ran low", {   # Thursday
            "North A": det(g, .85, 6), "North B": det(g, .72, 2)}),
        (off(last_mon + timedelta(days=1)), "07:44", [], None, {"North A": det(g, .81, 3)}),  # Tuesday
        (off(last_mon - timedelta(days=4)), "07:40", [], None, {}),
        (off(last_mon - timedelta(days=11)), "07:36", [], None, {}),
    ]
    return [{"date": today - timedelta(days=off), "time": tm, "uncovered": unc,
             "reason": why, "detections": dets} for off, tm, unc, why, dets in spec]

# --------------------------------------------------------------------------
# Entity extraction (rule-based: small fixed vocabularies)
# --------------------------------------------------------------------------
MONTH_RE = (r"(january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|august|aug|"
            r"september|sept|sep|october|oct|november|nov|december|dec)")
MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
NUMWORDS = {"one": 1, "a": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}
DIR_RE = re.compile(
    r"\b(north|south|east|west|centre|center)\b(?:\s+(?:zone|field|block|section|area|part))?"
    r"(?:\s+([abc])\b(?:\s*(?:,|and|&)\s*([abc])\b)?)?", re.I)


def extract_zone(text):
    found = []
    for m in DIR_RE.finditer(text):
        d = m.group(1).lower()
        d = "Centre" if d in ("centre", "center") else d.title()
        letters = [x for x in (m.group(2), m.group(3)) if x]
        if d == "Centre":
            found.append("Centre")
        elif letters:
            for L in letters:
                name = f"{d} {L.upper()}"
                found.append(name if name in ZONES else f"{d.lower()} field")
        else:
            found.append(f"{d.lower()} field")
    found = list(dict.fromkeys(found))
    return None if not found else (found[0] if len(found) == 1 else found)


def resolve_zones(entity):
    """Turn a zone entity (name, 'north field' label, or list) into real zone names."""
    if entity is None:
        return []
    items = [entity] if isinstance(entity, str) else list(entity)
    out = []
    for it in items:
        if it in ZONES:
            out.append(it)
        elif it.endswith(" field"):
            d = it.split()[0].title()
            out += [z for z in ZONES if z.startswith(d)]
    return list(dict.fromkeys(out))


def extract_crop(text):
    found = [c for c, pat in CROPS.items() if re.search(rf"\b(?:{pat})\b", text, re.I)]
    return None if not found else (found[0] if len(found) == 1 else found)


def extract_disease(text):
    for name, pat in DISEASES:
        if re.search(pat, text, re.I):
            return name
    return None


def parse_dates(text, today):
    t = text.lower().replace("\u2019", "'")
    wk = today - timedelta(days=today.weekday())

    def rng(label, s, e):
        return {"label": label, "start": s.isoformat(), "end": e.isoformat()}

    def one(label, d):
        return rng(label, d, d)

    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", t)
    if m:
        try:
            return one("date", date(int(m[1]), int(m[2]), int(m[3])))
        except ValueError:
            pass
    for pat, dgrp, mgrp in ((rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?{MONTH_RE}\b", 1, 2),
                            (rf"\b{MONTH_RE}\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", 2, 1)):
        m = re.search(pat, t)
        if m:
            try:
                d = date(today.year, MONTHS[m.group(mgrp)[:3]], int(m.group(dgrp)))
                if d > today:
                    d = d.replace(year=today.year - 1)
                return one("date", d)
            except ValueError:
                pass
    if "day before yesterday" in t:
        return one("day_before_yesterday", today - timedelta(days=2))
    if "yesterday" in t:
        return one("yesterday", today - timedelta(days=1))
    if re.search(r"\btoday\b", t):
        return one("today", today)
    m = re.search(r"\b(\d+|one|two|three|four|five|six|seven)\s+days?\s+ago\b", t)
    if m:
        n = int(m[1]) if m[1].isdigit() else NUMWORDS[m[1]]
        return one(f"{n}_days_ago", today - timedelta(days=n))
    m = re.search(r"\b(?:last|past|previous)\s+(\d+)\s+days\b", t)
    if m:
        return rng(f"last_{m[1]}_days", today - timedelta(days=int(m[1])), today)
    m = re.search(r"\b(?:last|past|previous)\s+(\d+)\s+weeks\b", t)
    if m:
        return rng(f"last_{m[1]}_weeks", today - timedelta(weeks=int(m[1])), today)
    if re.search(r"\bpast week\b", t):
        return rng("last_7_days", today - timedelta(days=7), today)
    if re.search(r"\blast week\b|\bprevious week\b", t):
        return rng("last_week", wk - timedelta(days=7), wk - timedelta(days=1))
    if re.search(r"\bthis week\b", t):
        return rng("this_week", wk, today)
    if re.search(r"\blast month\b|\bprevious month\b|\bpast month\b", t):
        first = today.replace(day=1)
        end = first - timedelta(days=1)
        return rng("last_month", end.replace(day=1), end)
    if re.search(r"\bthis month\b", t):
        return rng("this_month", today.replace(day=1), today)
    return None


def extract_entities(text, today):
    return {"crop": extract_crop(text), "zone": extract_zone(text),
            "disease": extract_disease(text), "date_range": parse_dates(text, today)}

# --------------------------------------------------------------------------
# Safety layer constants
# --------------------------------------------------------------------------
SPRAY_WORDS = re.compile(
    r"\b(spray\w*|treat\w*|pesticid\w*|fungicid\w*|insecticid\w*|herbicid\w*|chemical\w*|dos(?:e|age)|apply)\b",
    re.I)
VERIFY_SHORT = ("These are AI suggestions, not confirmed diagnoses. Please review the leaf photos and "
                "check a few plants in the flagged zones yourself before deciding on any treatment.")
VERIFY_SPRAY = ("Please review the leaf photos and the spray map first, and inspect any medium- or "
                "low-confidence zone in the field yourself before you decide anything. The AI supports "
                "your decision; it does not replace it.")
LABEL_NOTE = ("I can't recommend products, amounts or mixing rates. Please follow the product label and "
              "local regulations, or ask an agronomist.")
HELP_TEXT = ("Hello, I'm AgriAssist. I report what the AI found in your fields: possible diseases, zones "
             "flagged for possible treatment, confidence levels and past results. I can't tell you what to "
             "spray, and the decision is always yours after you have checked.\n"
             "You can ask, for example: \"What's wrong with my maize field?\", \"Which zones need spraying?\", "
             "\"How confident is the AI?\", \"Show me last week's results\", or \"When was the last scan?\"")


def tier(conf):
    return "high" if conf >= TIER_HIGH else "medium" if conf >= TIER_MEDIUM else "low"


def join_and(items):
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def zone_phrase(z):
    return z if z in ZONES else f"the {z}"


def fmt_d(d):
    return f"{d.day} {d.strftime('%b')}"

# --------------------------------------------------------------------------
# The chatbot
# --------------------------------------------------------------------------
class AgriAssist:
    def __init__(self, today=None, scans=None, model=None, log_path=None):
        self.today = today or date.today()
        self.scans = scans if scans is not None else default_scans(self.today)
        self.model = model or train_classifier()
        self.review = {}            # zone -> "reviewed" (latest scan only)
        self.flags = []             # farmer-flagged results (feedback log)
        self.ctx = {"zones": [], "disease": None}
        self.n = 0
        self.log_path = log_path

    # ---- parsing -------------------------------------------------------
    def classify(self, text):
        probs = self.model.predict_proba([preprocess(text)])[0]
        i = int(probs.argmax())
        return str(self.model.classes_[i]), float(probs[i])

    def parse(self, text, input_mode="text"):
        intent, conf = self.classify(text)
        ent = extract_entities(text, self.today)
        flags = []
        spray = bool(SPRAY_WORDS.search(text))
        if spray:
            flags.append("spray_related")
        # Safety override: a spray-related question must never fall into a topic
        # that skips the safety response, whatever the classifier predicted.
        if spray and (conf < CLARIFY_THRESHOLD or intent in
                      ("out_of_scope", "smalltalk_help", "get_disease_info", "scan_status")):
            intent = "request_treatment_advice"
        verify = intent in VERIFY_INTENTS
        if verify:
            flags.append("requires_confidence_check")

        context_ref = None
        needs, prompt = False, None
        leftover = [w for w in DIR_RE.sub(" ", preprocess(text)).split()
                    if w not in ("the", "a", "and", "about", "what", "how", "field", "zone", "ok")]
        zone_only = ent["zone"] is not None and not leftover and not spray
        if zone_only:
            needs = True
            prompt = (f"What would you like to know about {zone_phrase(ent['zone'])}?"
                      if isinstance(ent["zone"], str)
                      else f"What would you like to know about {join_and(ent['zone'])}?")
        elif conf < CLARIFY_THRESHOLD and intent != "request_treatment_advice":
            needs = True
            prompt = (f"What would you like to know about {zone_phrase(ent['zone'])}?"
                      if isinstance(ent["zone"], str)
                      else "I'm not sure I understood. You can ask about disease results, the spray map, "
                           "confidence, past results or scan status. What would you like to know?")
        elif intent in ("mark_reviewed", "flag_incorrect") and conf < STATE_THRESHOLD:
            needs = True
            prompt = ("Do you want me to record a zone as reviewed? If so, tell me which one, for example "
                      "\"I checked North A\"." if intent == "mark_reviewed" else
                      "Do you want me to flag a result as wrong? If so, tell me which zone, for example "
                      "\"the North A result is wrong\".")
        elif intent in ("get_confidence", "explain_result", "mark_reviewed", "flag_incorrect"):
            if ent["zone"] is None and self.ctx["zones"]:
                z = self.ctx["zones"]
                context_ref = z[0] if len(z) == 1 else list(z)
            elif ent["zone"] is None and intent in ("mark_reviewed", "flag_incorrect"):
                needs = True
                prompt = ("Which zone did you check?" if intent == "mark_reviewed"
                          else "Which result would you like to flag? Please tell me the zone.")
        elif intent == "show_history" and ent["date_range"] is None:
            needs, prompt = True, "Which period would you like to see: yesterday, last week or last month?"
        elif intent == "get_disease_info" and ent["disease"] is None and not self.ctx["disease"]:
            needs, prompt = True, "Which disease would you like to know about?"

        return {
            "query_id": f"q-{self.n:04d}", "timestamp": datetime.now().isoformat(timespec="seconds"),
            "raw_text": text, "input_mode": input_mode, "intent": intent,
            "intent_confidence": round(conf, 2), "entities": ent, "context_ref": context_ref,
            "needs_clarification": needs, "clarification_prompt": prompt,
            "safety_flags": flags, "requires_farmer_verification": verify,
            "route_to": "chatbot" if needs else ROUTES[intent],
        }

    # ---- data helpers ----------------------------------------------------
    @property
    def latest(self):
        return self.scans[0]

    def when(self, scan):
        d = (self.today - scan["date"]).days
        if d == 0:
            return f"today, {scan['time']}"
        if d == 1:
            return f"yesterday, {scan['time']}"
        return fmt_d(scan["date"])

    def coverage(self, scan):
        total = sum(z["ha"] for z in ZONES.values())
        missed = sum(ZONES[z]["ha"] for z in scan["uncovered"])
        return round(100 * (1 - missed / total))

    def select(self, ent):
        names = list(ZONES)
        if ent["crop"]:
            crops = [ent["crop"]] if isinstance(ent["crop"], str) else ent["crop"]
            names = [z for z in names if ZONES[z]["crop"] in crops]
        if ent["zone"]:
            wanted = resolve_zones(ent["zone"])
            names = [z for z in names if z in wanted]
        return names

    def dets(self, zones, ent=None, scan=None):
        scan = scan or self.latest
        out = [(z, scan["detections"][z]) for z in zones if z in scan["detections"]]
        if ent and ent.get("disease"):
            out = [(z, d) for z, d in out if d["disease"] == ent["disease"]]
        return out

    @staticmethod
    def scope(ent):
        if ent["zone"] and isinstance(ent["zone"], str):
            return f"the {ent['zone']}" if ent["zone"].endswith("field") else ent["zone"]
        if ent["zone"]:
            return join_and(ent["zone"])
        if ent["crop"] and isinstance(ent["crop"], str):
            return f"your {ent['crop']} fields"
        return "your fields"

    @staticmethod
    def det_line(z, d):
        return f"{z}: {d['disease']}, {d['conf']:.0%} confidence ({tier(d['conf'])})"

    def zones_from(self, parsed):
        ent = parsed["entities"]
        if ent["zone"]:
            return resolve_zones(ent["zone"])
        ref = parsed["context_ref"]
        if ref:
            return [ref] if isinstance(ref, str) else list(ref)
        return [z for z in ZONES if z in self.latest["detections"]]

    # ---- intent handlers (return reply text and meta) -----------------------
    def dispatch(self, parsed):
        fn = getattr(self, "h_" + parsed["intent"])
        return fn(parsed)

    def h_identify_disease(self, p):
        ent, sc = p["entities"], self.latest
        sel = self.select(ent)
        if not sel:
            return "I can't find a zone that matches that. Your zones are: " + ", ".join(ZONES) + ".", {}
        dets = self.dets(sel, ent)
        scope = self.scope(ent)
        if not dets:
            return (f"In the latest scan ({self.when(sc)}) the AI did not detect any disease in {scope}. "
                    "The AI can miss early disease, so walking the crop regularly is still worthwhile."), \
                   {"no_detections": True, "next": "Would you like a summary of all your zones?"}
        names = sorted({d["disease"] for _, d in dets})
        what = f"one possible disease: {names[0]}" if len(names) == 1 else "possible diseases: " + join_and(names)
        lines = [f"In the latest scan ({self.when(sc)}) the AI detected {what}, in {len(dets)} of "
                 f"{len(sel)} zones in {scope} ({join_and([z for z, _ in dets])}).",
                 "Confidence: " + join_and([f"{d['conf']:.0%} in {z}" for z, d in dets]) + "."]
        return "\n".join(lines), {"dets": [(z, d["conf"]) for z, d in dets], "zones": [z for z, _ in dets],
                                  "next": "Would you like to see the photos, the affected area or the spray map?"}

    def h_field_health_summary(self, p):
        ent, sc = p["entities"], self.latest
        sel = self.select(ent)
        dets = self.dets(sel, ent)
        flagged = {z for z, _ in dets}
        clear = [z for z in sel if z not in flagged and z not in sc["uncovered"]]
        lines = [f"Summary for {self.scope(ent)} from the latest scan ({self.when(sc)}): "
                 f"{len(dets)} of {len(sel)} zones show a possible disease."]
        lines += [self.det_line(z, d) for z, d in dets]
        if clear:
            lines.append("No detection in: " + ", ".join(clear) + ". (No detection does not prove a zone is "
                         "healthy; the AI can miss early disease.)")
        return "\n".join(lines), {"dets": [(z, d["conf"]) for z, d in dets], "zones": list(flagged),
                                  "no_detections": not dets,
                                  "next": "You can ask for the spray map, the affected area or the photos."}

    def h_disease_location(self, p):
        ent, sc = p["entities"], self.latest
        dets = self.dets(self.select(ent), ent)
        if not dets:
            return ("The latest scan has no detections in the zones you asked about. The AI can miss "
                    "early disease."), {"no_detections": True}
        lines = [f"Detections in the latest scan ({self.when(sc)}):"]
        lines += [f"{z}: {d['disease']}, {d['conf']:.0%} confidence ({tier(d['conf'])}), recorded "
                  f"{fmt_d(sc['date'])} at {sc['time']}" for z, d in dets]
        return "\n".join(lines), {"dets": [(z, d["conf"]) for z, d in dets], "zones": [z for z, _ in dets],
                                  "next": "Would you like the spray map or the affected area?"}

    def h_get_affected_area(self, p):
        ent = p["entities"]
        sel = self.select(ent)
        dets = self.dets(sel, ent)
        if not dets:
            return "No affected area was detected in those zones in the latest scan.", {"no_detections": True}
        lines = ["Estimated affected area from the latest scan:"]
        hit = 0.0
        for z, d in dets:
            ha = ZONES[z]["ha"] * d["affected_pct"] / 100
            hit += ha
            lines.append(f"{z}: about {d['affected_pct']}% of the zone ({ha:.1f} of {ZONES[z]['ha']} ha), "
                         f"{d['disease']}, {d['conf']:.0%} confidence")
        total = sum(ZONES[z]["ha"] for z in sel)
        lines.append(f"Overall about {100 * hit / total:.1f}% of {self.scope(ent)} ({hit:.1f} of {total:.1f} ha). "
                     "These are estimates from the images.")
        return "\n".join(lines), {"dets": [(z, d["conf"]) for z, d in dets], "zones": [z for z, _ in dets]}

    def _status(self, z):
        return "reviewed by you" if self.review.get(z) else "pending your review"

    def h_show_spray_map(self, p):
        ent, sc = p["entities"], self.latest
        dets = self.dets(self.select(ent), ent)
        if not dets:
            return (f"No zones are flagged as possibly needing treatment in the latest scan ({self.when(sc)}). "
                    "The AI can miss early disease, so keep checking your crop."), {"no_detections": True}
        lines = [f"{len(dets)} zone{'s are' if len(dets) > 1 else ' is'} flagged as possibly needing "
                 f"treatment in the latest scan ({self.when(sc)}):"]
        lines += [f"{self.det_line(z, d)}, {self._status(z)}" for z, d in dets]
        lines.append("The spray map stays marked Pending your review until you confirm each zone. "
                     "Nothing is acted on automatically.")
        return "\n".join(lines), {"dets": [(z, d["conf"]) for z, d in dets], "zones": [z for z, _ in dets],
                                  "next": "You can tell me which zones you have checked, or ask how "
                                          "confident the AI is."}

    def h_request_treatment_advice(self, p):
        sc = self.latest
        dets = self.dets(self.select(p["entities"]), p["entities"])
        lines = ["I can't tell you whether or what to spray. That decision needs your own check of the "
                 "AI results."]
        if dets:
            lines.append(f"Here is what the AI found in the latest scan ({self.when(sc)}):")
            lines += [self.det_line(z, d) for z, d in dets]
            rest = len(ZONES) - len(dets)
            lines.append(f"The other {rest} zones had no detection, although the AI can miss early disease.")
        else:
            lines.append("The latest scan shows no detections in the zones you asked about, although the "
                         "AI can miss early disease.")
        return "\n".join(lines), {"dets": [(z, d["conf"]) for z, d in dets], "zones": [z for z, _ in dets],
                                  "no_detections": not dets,
                                  "next": "Tell me which zones you have reviewed and I will record them."}

    def h_get_confidence(self, p):
        zones = self.zones_from(p)
        dets = self.dets(zones, p["entities"])
        if not dets:
            return "I have no AI detection to report a confidence for in that zone.", {"no_detections": True}
        lines = []
        for z, d in dets:
            s = f"{z}: the AI is {d['conf']:.0%} confident this is {d['disease']} ({tier(d['conf'])} confidence)."
            if d["second"] and d["second_conf"] >= 0.10:
                s += f" Its second most likely result is {d['second']} ({d['second_conf']:.0%})."
            lines.append(s)
        others = [(z, d) for z, d in self.latest["detections"].items() if z not in {z for z, _ in dets}]
        if others and len(dets) == 1:
            lines.append("For comparison: " + join_and([f"{z} {d['conf']:.0%}" for z, d in others]) + ".")
        return "\n".join(lines), {"dets": [(z, d["conf"]) for z, d in dets], "zones": [z for z, _ in dets],
                                  "next": "If you think a result is wrong, say \"flag this result\" and I will "
                                          "record it for the team."}

    def h_explain_result(self, p):
        zones = self.zones_from(p)
        dets = sorted(self.dets(zones, p["entities"]), key=lambda x: x[1]["conf"])
        if not dets:
            return "There is no AI detection to explain for that zone.", {"no_detections": True}
        lines = ["Here is what the AI based each result on (lowest confidence first):"]
        for z, d in dets:
            s = (f"{z}: top result {d['disease']} ({d['conf']:.0%}); {d['photos']} leaf photos analysed")
            if d["blurry"]:
                s += f", {d['blurry']} marked slightly blurry"
            if d["second"] and d["second_conf"] >= 0.10:
                s += f"; second most likely {d['second']} ({d['second_conf']:.0%})"
            lines.append(s + ".")
        lines.append("(Prototype note: the dashboard photo viewer is not connected, so photos are not "
                     "displayed here.)")
        return "\n".join(lines), {"dets": [(z, d["conf"]) for z, d in dets], "zones": [z for z, _ in dets],
                                  "next": "Compare the photos with what you see on the plants, then tell me "
                                          "what you find."}

    def h_mark_reviewed(self, p):
        zones = self.zones_from(p)
        flagged = [z for z in zones if z in self.latest["detections"]]
        if not flagged:
            return "There is no flagged detection in that zone to mark as reviewed.", {}
        for z in flagged:
            self.review[z] = "reviewed"
        pending = [z for z in self.latest["detections"] if not self.review.get(z)]
        text = f"Thank you. I have recorded {join_and(flagged)} as reviewed by you."
        text += (f" {join_and(pending)} {'is' if len(pending) == 1 else 'are'} still pending. Tell me once you "
                 "have checked them." if pending else " All flagged zones are now reviewed.")
        text += " This only records your check; it does not trigger any action."
        return text, {"zones": flagged}

    def h_flag_incorrect(self, p):
        zones = self.zones_from(p)
        for z in zones:
            self.flags.append({"zone": z, "scan": self.latest["date"].isoformat(),
                               "text": p["raw_text"], "at": datetime.now().isoformat(timespec="seconds")})
        return (f"Done. I have flagged the {join_and(zones)} result as possibly incorrect and added it to the "
                "review list for the team. Thank you."), {"zones": zones}

    def h_show_history(self, p):
        dr = p["entities"]["date_range"]
        s, e = date.fromisoformat(dr["start"]), date.fromisoformat(dr["end"])
        inr = sorted([x for x in self.scans if s <= x["date"] <= e], key=lambda x: x["date"])
        span = fmt_d(s) if s == e else f"{fmt_d(s)} to {fmt_d(e)}"
        if not inr:
            return f"I have no scans between {span}.", {"no_detections": True, "skip_age": True}
        lines = [f"Results for {dr['label'].replace('_', ' ')} ({span}). The system completed "
                 f"{len(inr)} scan{'s' if len(inr) > 1 else ''}:"]
        for x in inr:
            d = x["detections"]
            if d:
                cs = [v["conf"] for v in d.values()]
                rng = f"{min(cs):.0%}" if min(cs) == max(cs) else f"{min(cs):.0%} to {max(cs):.0%}"
                line = f"{fmt_d(x['date'])}: {len(d)} zone{'s' if len(d) > 1 else ''} flagged (confidence {rng})"
            else:
                line = f"{fmt_d(x['date'])}: no zones flagged"
            if x["uncovered"]:
                why = f" because {x['reason']}" if x["reason"] else ""
                line += (f". This scan covered about {self.coverage(x)}% of the field{why}, so "
                         f"{join_and(x['uncovered'])} was not checked")
            lines.append(line + ".")
        incomplete = any(x["uncovered"] for x in inr)
        if len(inr) > 1:
            a, b = len(inr[0]["detections"]), len(inr[-1]["detections"])
            lines.append(f"The number of flagged zones went from {a} to {b} over this period"
                         + (", but at least one scan was incomplete, so please compare with care." if incomplete
                            else "."))
        newest = inr[-1]
        if newest["detections"]:
            lines.append(f"Most recent scan in this period ({fmt_d(newest['date'])}): " + join_and(
                [f"{z} {d['conf']:.0%}" for z, d in newest["detections"].items()]) + ".")
        return "\n".join(lines), {"dets": [(z, d["conf"]) for z, d in newest["detections"].items()],
                                  "no_detections": not newest["detections"], "skip_age": True,
                                  "scan": newest, "zones": list(newest["detections"]),
                                  "next": "Would you like to compare this with the latest scan?"}

    def h_compare_over_time(self, p):
        ent = p["entities"]
        sel = self.select(ent)
        latest = self.latest
        ref = None
        if ent["date_range"]:
            s, e = date.fromisoformat(ent["date_range"]["start"]), date.fromisoformat(ent["date_range"]["end"])
            cands = [x for x in self.scans[1:] if s <= x["date"] <= e]
            ref = cands[0] if cands else None
        elif len(self.scans) > 1:
            ref = self.scans[1]
        if ref is None:
            return "I don't have an earlier scan to compare with for that period.", {"no_detections": True}
        now, then = dict(self.dets(sel, ent)), dict(self.dets(sel, ent, ref))
        lines = [f"Comparing the latest scan ({self.when(latest)}) with {self.when(ref)}, for {self.scope(ent)}:",
                 f"Zones flagged: {len(then)} then, {len(now)} now."]
        new = [z for z in now if z not in then]
        gone = [z for z in then if z not in now]
        if new:
            lines.append("Newly flagged: " + join_and(new) + ".")
        if gone:
            lines.append("No longer flagged: " + join_and(gone) + ".")
        for z, d in now.items():
            if z in then:
                lines.append(f"{z}: affected area {then[z]['affected_pct']}% then, {d['affected_pct']}% now "
                             f"({d['disease']}, {d['conf']:.0%} confidence now).")
        for z in new:
            lines.append(f"{z}: now {self.det_line(z, now[z]).split(': ', 1)[1]}.")
        if ref["uncovered"] or latest["uncovered"]:
            lines.append("One of these scans was incomplete, so please compare with care.")
        return "\n".join(lines), {"dets": [(z, d["conf"]) for z, d in now.items()], "zones": list(now),
                                  "no_detections": not now}

    def h_get_disease_info(self, p):
        name = p["entities"]["disease"] or self.ctx["disease"]
        if name not in DISEASE_INFO:
            return "I don't have information about that disease yet.", {}
        return (DISEASE_INFO[name] + "\nThis is general information only. For decisions about your crop, "
                "please speak to an agronomist or your local extension officer."), {}

    def h_scan_status(self, p):
        sc = self.latest
        age = (self.today - sc["date"]).days
        lines = [f"The latest scan was {self.when(sc)}. It covered about {self.coverage(sc)}% of the field."]
        if sc["uncovered"]:
            lines.append(f"{join_and(sc['uncovered'])} was not scanned.")
        n7 = len([x for x in self.scans if (self.today - x["date"]).days <= 7])
        lines.append(f"{n7} scans were completed in the last 7 days.")
        if age >= 3:
            lines.append(f"This data is {age} days old, so conditions may have changed.")
        lines.append("I can't see the drone schedule or control the drone.")
        return "\n".join(lines), {}

    def h_smalltalk_help(self, p):
        return HELP_TEXT, {}

    def h_out_of_scope(self, p):
        return ("I can only help with your crop scan results: diseases found, spray-map zones, confidence levels, "
                "past results and scan status."), {}

    # ---- safety layer --------------------------------------------------------
    def data_notes(self, meta, dets):
        sc = meta.get("scan") or self.latest
        notes = []
        age = (self.today - self.latest["date"]).days
        if age >= 3 and not meta.get("skip_age"):
            notes.append(f"The latest scan is {age} days old, so conditions may have changed.")
        if sc["uncovered"] and not meta.get("skip_age"):
            notes.append(f"The latest scan did not cover {join_and(sc['uncovered'])}, so those areas were not checked.")
        blur = [f"{z} ({sc['detections'][z]['blurry']} of {sc['detections'][z]['photos']})"
                for z, _ in dets if z in sc["detections"] and sc["detections"][z]["blurry"]]
        if blur:
            notes.append("The photo-quality check marked some photos as slightly blurry in "
                         + join_and(blur) + ", which can reduce accuracy.")
        return notes

    def apply_safety_layer(self, reply, meta, parsed):
        """Rules S1-S8 from the design document, applied to every reply."""
        flags, intent = parsed["safety_flags"], parsed["intent"]
        spray_ctx = ("spray_related" in flags or intent in ("show_spray_map", "request_treatment_advice")
                     or bool(SPRAY_WORDS.search(reply)))
        parts = [reply]
        if parsed["requires_farmer_verification"]:
            dets = meta.get("dets", [])
            missing = [(z, c) for z, c in dets if f"{c:.0%}" not in reply]        # S1 confidence first
            if missing:
                parts.append("AI confidence: " + ", ".join(f"{z} {c:.0%} ({tier(c)})" for z, c in missing) + ".")
            if not dets and not meta.get("no_detections"):
                parts.append("I don't have confidence values for this result, so please don't rely on it "
                             "until you have checked it yourself.")
            low = [z for z, c in dets if tier(c) == "low"]                         # S4 tiers
            med = [z for z, c in dets if tier(c) == "medium"]
            if low:
                parts.append(f"The AI is not confident about {join_and(low)}. Please do not rely on "
                             f"{'that result' if len(low) == 1 else 'those results'}; inspect the area "
                             "yourself or ask an agronomist.")
            if med:
                parts.append(f"The AI is only moderately confident about {join_and(med)}. Please inspect "
                             f"{'it' if len(med) == 1 else 'them'} in the field before deciding.")
            parts += self.data_notes(meta, dets)                                   # S7 honest about gaps
            parts.append(VERIFY_SPRAY if spray_ctx else VERIFY_SHORT)              # S2 verify before action
        if spray_ctx:
            parts.append(LABEL_NOTE)                                               # S6 no products or doses
        if meta.get("next"):
            parts.append(meta["next"])
        return "\n\n".join(parts)

    # ---- main entry point ----------------------------------------------------------
    def ask(self, text, input_mode="text"):
        self.n += 1
        parsed = self.parse(text, input_mode)
        if parsed["needs_clarification"]:
            reply = parsed["clarification_prompt"]
        else:
            body, meta = self.dispatch(parsed)
            reply = self.apply_safety_layer(body, meta, parsed)
            if meta.get("zones"):
                self.ctx["zones"] = list(meta["zones"])
                ds = [self.latest["detections"][z]["disease"] for z in meta["zones"]
                      if z in self.latest["detections"]]
                if ds:
                    self.ctx["disease"] = max(set(ds), key=ds.count)
        if self.log_path:                                                          # S9 keep a record
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"parsed": parsed, "reply": reply}) + "\n")
        return reply, parsed

    def focus(self, zone_text):
        zones = resolve_zones(extract_zone(zone_text))
        self.ctx["zones"] = zones
        return zones

# --------------------------------------------------------------------------
# Interfaces
# --------------------------------------------------------------------------
def show(reply, width=92):
    out = []
    for para in reply.split("\n\n"):
        out.append("\n".join(textwrap.fill(line, width, subsequent_indent="  ") for line in para.split("\n")))
    print("\nAgriAssist> " + "\n\n".join(out) + "\n")


def chat(bot, show_json=False):
    print("AgriAssist (prototype, mock scan data). Type /help for commands, /quit to exit.")
    print("Commands: /json toggles the parsed query, /focus East C simulates a dashboard selection, /reset.\n")
    while True:
        try:
            text = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text in ("/quit", "/exit"):
            break
        if text == "/json":
            show_json = not show_json
            print(f"(parsed query display {'on' if show_json else 'off'})")
            continue
        if text == "/reset":
            bot.review.clear(); bot.ctx = {"zones": [], "disease": None}
            print("(conversation state cleared)")
            continue
        if text.startswith("/focus"):
            print(f"(dashboard focus set to: {', '.join(bot.focus(text[6:])) or 'none'})")
            continue
        reply, parsed = bot.ask(text)
        if show_json:
            print(json.dumps(parsed, indent=2))
        show(reply)


DEMO = [
    ("1. Diagnosis", None, ["What's wrong with my maize field?", "Show me the photos."]),
    ("2. Spray map and review", None, ["Which zones need spraying?", "I checked North A and B and they look right."]),
    ("3. Confidence (dashboard focus: East C)", "East C",
     ["How confident is the AI about this result?", "Flag this result."]),
    ("4. History", None, ["Show me last week's results."]),
    ("5. Pressure to skip verification", None, ["Just tell me to spray the whole field."]),
    ("6. Unclear query", None, ["north?"]),
]


def demo(show_json=False):
    for title, focus, msgs in DEMO:
        bot = AgriAssist()
        print("=" * 78 + f"\n{title}\n" + "=" * 78)
        if focus:
            bot.focus(focus)
        for m in msgs:
            print(f"You> {m}")
            reply, parsed = bot.ask(m)
            if show_json:
                print(json.dumps(parsed, indent=2))
            show(reply)


def evaluate():
    from sklearn.metrics import classification_report
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    X, y = training_arrays()
    pred = cross_val_predict(build_pipeline(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
    print(classification_report(y, pred, zero_division=0))
    print("Note: this is cross-validation on the team's own training phrases. It is optimistic. "
          "Use test_agriassist.py (held-out phrasings) and real farmer wording for a fairer estimate.")


PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>AgriAssist</title><style>
body{font-family:Arial,sans-serif;max-width:720px;margin:0 auto;padding:12px;background:#f6f8f6}
h1{font-size:20px;color:#1f4e3d}#log{min-height:320px}
.m{padding:10px 12px;margin:8px 0;border-radius:8px;white-space:pre-wrap;line-height:1.4}
.f{background:#fff6dd;margin-left:15%}.b{background:#eaf3ee;margin-right:5%}
details{font-size:12px;color:#444;margin-top:6px}pre{overflow:auto}
form{display:flex;gap:8px;position:sticky;bottom:0;background:#f6f8f6;padding:8px 0}
input{flex:1;padding:10px;font-size:16px}button{padding:10px 16px;font-size:16px}
.n{font-size:12px;color:#555}</style>
<h1>AgriAssist</h1><div class=n>Prototype using mock scan data. It reports what the AI found; it never tells you to spray.</div>
<div id=log></div><form id=f><input id=q autocomplete=off placeholder="Ask about your fields..." autofocus>
<button>Send</button></form><script>
const log=document.getElementById('log');
function add(cls,t,j){const d=document.createElement('div');d.className='m '+cls;d.textContent=t;
if(j){const e=document.createElement('details');const s=document.createElement('summary');
s.textContent='Parsed query';const p=document.createElement('pre');p.textContent=JSON.stringify(j,null,2);
e.append(s,p);d.append(e)}log.append(d);d.scrollIntoView()}
document.getElementById('f').onsubmit=async e=>{e.preventDefault();const q=document.getElementById('q');
const t=q.value.trim();if(!t)return;q.value='';add('f',t);
const r=await fetch('/ask',{method:'POST',body:JSON.stringify({message:t})});const j=await r.json();
add('b',j.reply,j.parsed)};
</script>"""


def serve(bot, port=8000):
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype):
            data = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._send(200, PAGE, "text/html; charset=utf-8")

        def do_POST(self):
            if self.path != "/ask":
                return self._send(404, "{}", "application/json")
            n = int(self.headers.get("Content-Length", 0))
            msg = json.loads(self.rfile.read(n) or b"{}").get("message", "")
            reply, parsed = bot.ask(msg)
            self._send(200, json.dumps({"reply": reply, "parsed": parsed}), "application/json")

        def log_message(self, *a):
            pass

    print(f"AgriAssist web chat on http://127.0.0.1:{port}  (Ctrl+C to stop)")
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()


def main():
    ap = argparse.ArgumentParser(description="AgriAssist farmer chatbot (prototype)")
    ap.add_argument("--demo", action="store_true", help="replay the sample conversations")
    ap.add_argument("--eval", action="store_true", help="cross-validate the intent classifier")
    ap.add_argument("--web", action="store_true", help="serve a small browser chat UI")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--json", action="store_true", help="print the parsed query for each message")
    ap.add_argument("--log", metavar="FILE", help="append a JSON-lines log of every turn")
    a = ap.parse_args()
    if a.eval:
        return evaluate()
    if a.demo:
        return demo(a.json)
    bot = AgriAssist(log_path=a.log)
    if a.web:
        return serve(bot, a.port)
    chat(bot, a.json)


if __name__ == "__main__":
    main()
