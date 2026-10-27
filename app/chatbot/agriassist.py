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
        "what is the problem with my tomatoes", "is something wrong with the leaves",
        "any disease detected", "what did the scan find", "is my maize infected",
        "what is attacking my tomatoes", "tell me what's wrong with the crop",
        "disease check on my field", "what illness do my plants have",
        "whats wrong with the maize", "is there armyworm in my field",
        "did the system detect any disease", "diagnose my field",
        "what is wrong with the tomatoes",
    ],
    "field_health_summary": [
        "how is my field doing", "give me an overview of crop health",
        "is everything okay with the tomatoes", "how are my crops",
        "crop health summary", "how healthy is my farm", "overall status of the fields",
        "how is the maize looking", "are my crops healthy", "give me a summary",
        "field status", "how are things on the farm", "is the north field ok",
        "health report please", "how did the field look in the latest scan",
        "overview", "is my tomato healthy", "quick summary of all zones",
        "how's the farm", "what is the general condition of my crops",
        "how is the east zone doing", "north", "north?", "east zone", "what about r0c2",
        "and the south", "west field", "r2c2", "centre", "what about the north field",
        "how is everything looking", "status of all my crops", "give me the big picture", "how are the tomatoes doing", "is the maize okay", "what is the state of the farm", "overall crop condition", "farm summary please",
    ],
    "disease_location": [
        "where exactly is the disease", "which part of the field has armyworm",
        "show me where the problem is", "where was the disease found",
        "which zones are infected", "where are the sick plants",
        "location of the infected plants", "where did the AI detect disease",
        "which area has leaf spot", "where is the blight",
        "point me to the diseased area", "which zones have disease",
        "where are the affected plants", "in which part of the field is the problem",
        "where was it detected", "show infected zones",
        "map of the infected spots", "where are the armyworms",
        "what part of the field is sick", "where do I find the disease",
        "which blocks have disease", "where are the infected zones", "locate the disease", "which areas are sick", "where did you find the grasshoppers", "tell me where the infection is", "which zones show disease", "where are the problem spots",
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
        "is the AI certain about leaf spot", "can I rely on this result",
        "how good is this detection",
        "how certain are you about r0c2", "what is the confidence for r0c2", "is that result trustworthy", "how sure is the system about this", "how accurate is the AI here", "give me the confidence",
    ],
    "explain_result": [
        "why does the AI say it is fall armyworm", "show me the photo you used",
        "why was this zone flagged", "what did the AI look at",
        "show me the leaf pictures", "explain this result",
        "how did the AI decide", "show me the images", "show the photos",
        "what does the AI see in the photo", "why do you think it is blight",
        "which photo shows the disease", "let me see the leaf images",
        "what evidence is there", "why is r0c2 flagged", "open the photos",
        "show me the evidence", "what made the AI think that",
        "see the pictures of the sick leaves", "explain why",
        "why is r2c2 flagged", "show me what the AI saw", "why did you say leaf spot", "show the leaf pictures for r0c2",
    ],
    "mark_reviewed": [
        "I've checked r2c2 and it looks right", "mark the east zone as reviewed",
        "I reviewed r2c1", "I have checked the map", "confirm I checked zone a",
        "I checked it, it looks correct", "reviewed r0c2",
        "mark r2c2 as checked", "I verified the north zones",
        "the map looks right to me", "approve the north zone",
        "I've inspected r0c2 in the field", "I confirm the results for r2c2",
        "set r2c1 to reviewed", "done checking the photos",
        "I checked all zones", "I looked at the photos and they match",
        "I walked the field and it matches", "checked r2c1 and r2c2",
        "the detection in r0c2 is right",
        "I have inspected r2c2", "reviewed and confirmed", "I checked the photos for r0c2", "r2c1 is confirmed by me", "I double checked r2c2 and it is right", "tick off r2c1 as checked", "I have verified r0c2", "record that I reviewed the north zone",
    ],
    "flag_incorrect": [
        "that leaf is healthy", "this result is wrong", "the map is in the wrong place",
        "the AI is wrong", "flag this result", "that's not armyworm",
        "this is incorrect", "report a mistake", "there is no disease there",
        "the detection in r2c2 is wrong", "that is a false alarm",
        "the AI got it wrong", "flag as incorrect", "this isn't right",
        "wrong disease", "the plants there are fine", "I disagree with this result",
        "mark this as a mistake", "the zone is healthy", "that diagnosis is not right",
        "the location is off",
        "the AI made a mistake in r2c1", "that detection is a mistake", "no that is wrong", "the result for r0c2 is incorrect", "it is not leaf spot it is healthy", "please flag the r2c2 result", "this detection is false", "the AI missed the mark here",
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
        "is the north field better than last month", "has the leaf spot spread since the last scan",
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
        "what is leaf spot", "how does streak virus spread",
        "tell me about leaf curl", "what causes streak virus in maize",
        "explain leaf spot", "what does blight look like",
        "how does the disease spread", "what are the symptoms of streak virus",
        "what conditions does leaf spot like", "info on septoria leaf spot",
        "what is fall armyworm", "is fall armyworm dangerous",
        "tell me more about the disease", "how does blight affect tomatoes",
        "what are the signs of fall armyworm", "what is a fungal leaf disease",
        "learn about verticillium wilt", "what is blight", "describe fall armyworm",
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
        "when should I plant maize", "how much does a tractor cost", "what fertiliser is best for tomatoes", "tell me the news", "how do I register my farm", "where can I buy seed", "what is the price of maize", "translate this to zulu", "what is five plus five", "who is the president", "what is the date today", "i am hungry", "my phone is broken", "give me a recipe", "how many people live in africa", "can you call my husband", "what is a good name for a dog", "is it going to be hot this week", "how do I irrigate better", "tell me about the stock market",
    ],
}


# Additional training phrases (a mix of hand-written wording and earlier dev/test phrasings).
EXTRA_TRAINING = {
    "identify_disease": [
        "what is making my maize sick",
        "has my tomato crop got a disease",
        "is the corn unhealthy",
        "what has the AI spotted in my crop",
        "name the disease on my tomatoes",
        "what pest or disease is on my plants",
        "is anything infecting the north zone",
        "what sickness did the scan detect",
        "are there signs of disease",
        "check my maize for disease",
        "what's bothering my crops",
        "which disease was found in the east zone",
        "what's the matter with my maize",
        "is my maize diseased",
        "any sickness on the tomato plants",
        "what has the scan picked up on my crops",
        "is a disease attacking the corn",
        "which illness is in the east zone",
        "my maize looks sick, what is it",
        "which disease do my crops have",
        "has the AI picked up any disease on the tomatoes",
    ],
    "field_health_summary": [
        "how's everything on the farm today",
        "is the farm doing alright",
        "condition of my fields",
        "any problems on the farm today",
        "give me the farm status",
        "how are all the zones",
        "what's the health of my tomatoes",
        "is the maize in good condition",
        "report on crop health",
        "are all my zones healthy",
        "run me through how the fields are",
        "how does the farm look",
        "how are my fields looking this morning",
        "give me a quick rundown of the farm",
        "is the whole farm in good shape",
        "crop condition report please",
        "how is the maize looking",
        "what is the overall picture",
        "give me the state of play on the farm",
        "are the tomatoes fine",
        "how do things look today",
    ],
    "disease_location": [
        "where are the problem areas",
        "show the areas with disease",
        "what zones are affected",
        "which zones are showing infection",
        "find the infected spots",
        "where is the sickness in the field",
        "tell me the location of the disease",
        "which zones does the armyworm affect",
        "where has the AI found problems",
        "list the diseased zones",
        "where on the farm is the infection",
        "which parts of the field are diseased",
        "show me the spots where plants are sick",
        "where did the system find leaf spot",
        "tell me which zones have the disease",
        "which area is the blight in",
        "which section of the field has the disease",
        "can you tell me where the infected plants are",
        "point out the zones with disease",
    ],
    "get_affected_area": [
        "how much of the zone is diseased",
        "what is the size of the problem",
        "how many hectares are sick",
        "percentage infected in r0c2",
        "what portion is damaged",
        "how much of the crop has the disease",
        "give me the infected area in hectares",
        "how widespread is it in the north field",
        "how badly affected is the field",
        "what part of the maize is infected in percent",
        "how much of the farm is infected",
        "what percentage of the north zone is sick",
        "how many hectares have the disease",
        "how extensive is the infection",
        "is the diseased area large",
        "what share of my maize is affected",
        "how much of the field is infected",
        "what proportion of my maize is sick",
        "how many hectares are damaged",
    ],
    "show_spray_map": [
        "open the map of zones to treat",
        "which zones should be treated",
        "show me where to spray",
        "spray map for r0c2",
        "what is on the spray map",
        "list the zones flagged for treatment",
        "where is treatment suggested",
        "map of suggested spray areas",
        "the spraying zones please",
        "display the treatment zones",
        "where do I need to spray",
        "show me the zones for treatment",
        "bring up the spraying map",
        "which areas are marked for spraying",
        "give me the spray zones for the east",
        "what needs treating according to the map",
        "show me where treatment is needed",
        "which zones are flagged for treatment",
        "pull up the spray map for the north",
    ],
    "request_treatment_advice": [
        "what should I use to treat this",
        "how do I fix the disease on my maize",
        "which product kills armyworm",
        "what is the recommended pesticide",
        "give me a treatment plan",
        "what spray works on leaf spot",
        "should I treat everything",
        "tell me what chemicals to apply",
        "is it time to apply fungicide",
        "how much to apply per hectare",
        "what mix should I use",
        "do I go ahead and spray the field",
        "should I spray this week",
        "what chemical will fix this",
        "how do I get rid of the disease",
        "which fungicide is best",
        "tell me what to use on the maize",
        "can I just spray everything to be safe",
        "should I go ahead and spray",
        "what product should I put on it",
        "tell me how much to spray",
    ],
    "get_confidence": [
        "how certain is the result",
        "what is the probability that this is right",
        "give me the confidence percentage",
        "how much do you trust this detection",
        "how reliable is the AI on r0c2",
        "what are the chances the AI is correct",
        "how dependable is that result",
        "level of certainty please",
        "how sure is the AI about the maize result",
        "are the results trustworthy",
        "how confident is the AI",
        "is the system certain about r2c2",
        "what are the odds the AI is right",
        "how much should I believe this detection",
        "give me the certainty level",
        "is the result dependable",
        "how sure is the AI about the result",
        "is this result trustworthy",
        "what confidence does the AI have",
    ],
    "explain_result": [
        "what is the basis for this result",
        "how did the AI reach that decision",
        "which part of the leaf did it look at",
        "show me the evidence photos",
        "walk me through why it flagged r2c2",
        "why did the AI pick that disease",
        "what did the AI see that made it say armyworm",
        "can I view the images behind the detection",
        "explain how you decided",
        "show the image for r0c2",
        "why does the system think it is leaf spot",
        "let me look at the photos for r2c1",
        "what in the image made it flag this",
        "show me the pictures behind this",
        "how did you come to that conclusion",
        "can I see the leaf images",
        "why did the AI flag r0c2",
        "can I see the pictures the AI used",
        "what is the AI basing this on",
    ],
    "mark_reviewed": [
        "I am happy with the r2c2 result",
        "I have gone through r2c1 and it is correct",
        "record my check of r0c2",
        "log that I verified r2c2",
        "I confirm r2c1 and r2c2 are right",
        "I looked at the zone myself and it is accurate",
        "mark the zones I checked",
        "I've verified the results",
        "confirm that I reviewed this",
        "r2c2 has been checked by me",
        "I looked at r2c2 and the result is correct",
        "please mark r0c2 as reviewed",
        "I have confirmed the r2c1 detection",
        "I went out and checked r2c2, all good",
        "sign off r2c1 for me",
        "I inspected the plants in r0c2 and they match",
        "I've looked at r2c1 and agree",
        "mark r2c2 as verified",
        "I checked r0c2 in person and it is right",
    ],
    "flag_incorrect": [
        "the AI is wrong about r2c2",
        "that result is not correct",
        "false detection in r0c2",
        "this flag is a mistake",
        "this zone should not be marked",
        "tell the team this is wrong",
        "incorrect result in r2c1",
        "the AI misidentified the disease",
        "the map location is wrong",
        "that is a wrong result",
        "the AI is mistaken about r0c2",
        "those plants are fine, the result is wrong",
        "report this detection as incorrect",
        "r2c1 is not diseased, flag it",
        "the map shows the wrong spot",
        "this is a false positive",
        "the AI got r2c2 wrong",
        "that is not leaf spot",
        "this detection is mistaken",
    ],
    "show_history": [
        "show me the results of the last 3 days",
        "what did the AI find on 5 october",
        "give me the history for this week",
        "scans from two weeks ago",
        "what were the results last month",
        "show me yesterday's detections",
        "how did the field look on 1 october",
        "list my previous scans",
        "show past detections",
        "results from the last 30 days",
        "what were the results on 2 october",
        "show me the scans from the past two weeks",
        "let me see last week's detections",
        "what did you find three days ago",
        "pull up the earlier scans",
        "results for this month",
        "what did the scan find on 3 october",
        "show me results from the last 7 days",
        "show me everything from last month",
    ],
    "compare_over_time": [
        "has the disease got better since last week",
        "is the field improving over time",
        "has it spread compared to the previous scan",
        "how does this scan compare to the last one",
        "is the number of infected zones going up",
        "any change from last month",
        "is r0c2 getting worse",
        "has the area affected grown since last scan",
        "compare this week with last week",
        "is the north field recovering",
        "is the disease worse than last week",
        "has the north field improved",
        "how does today compare with the previous scan",
        "is the infection getting bigger",
        "what has changed since last time",
        "are things getting better or worse",
        "has it spread since last week",
        "is it worse than last month",
        "compare with the last scan",
    ],
    "get_disease_info": [
        "what is streak virus disease",
        "tell me about grasshoppers in maize",
        "what are the symptoms of septoria leaf spot",
        "how does leaf spot spread",
        "what causes leaf blight",
        "describe leaf curl to me",
        "what does leaf curl look like on a leaf",
        "what conditions favour blight",
        "explain septoria leaf spot",
        "what should I know about leaf beetles",
        "what is verticillium wilt",
        "what does leaf spot look like",
        "how does leaf curl spread between plants",
        "explain what streak virus is",
        "what weather does septoria leaf spot like",
        "give me background on leaf beetles",
        "tell me about leaf spot",
        "what does fall armyworm do to maize",
        "how does leaf blight start",
    ],
    "scan_status": [
        "when was the last drone flight",
        "how much of the farm was scanned",
        "were any zones skipped",
        "how fresh is the data",
        "did the drone cover all zones",
        "how many photos were captured",
        "was the last scan complete",
        "when did the last scan run",
        "was the battery enough to finish the scan",
        "which zones were not covered",
        "when did the drone last scan",
        "was every zone covered",
        "how many photos did the last scan take",
        "is the latest data from today",
        "did the scan complete properly",
        "which zones were missed",
        "when did the drone last fly over",
        "was the whole farm covered by the scan",
        "how recent is the latest scan",
    ],
    "smalltalk_help": [
        "hello there",
        "how does this work",
        "what questions can I ask",
        "hey agriassist",
        "thanks for the help",
        "thank you so much",
        "what do you know",
        "can you tell me what you can do",
        "hi, I need some help",
        "good day",
        "hi agriassist",
        "what can I ask you",
        "can you explain how to use this",
        "thanks, that helps",
        "good morning",
        "what are you able to do",
        "good day to you",
        "what can you help me with",
        "thank you very much",
    ],
    "out_of_scope": [
        "what is the price of maize at the market",
        "will it rain this weekend",
        "which tractor should I buy",
        "how do I apply for a loan",
        "what is the best time to plant tomatoes",
        "recommend a seed supplier",
        "who is the minister of agriculture",
        "tell me a story",
        "what is the score of the match",
        "when should I irrigate",
        "how do I prune tomato plants",
        "what fertiliser should I use",
        "how much does a drone cost",
        "can you order seeds for me",
        "what time does the market open",
        "what is the price of tomatoes",
        "will it be sunny on friday",
        "recommend a good tractor",
        "what is the capital of south africa",
        "how do I apply for a farming grant",
        "tell me something funny",
        "who won the rugby",
        "when is the best time to harvest maize",
        "how much is a bag of maize meal",
        "what's the forecast for the weekend",
        "can you book a tractor service",
    ],
}
for _intent, _examples in EXTRA_TRAINING.items():
    TRAINING[_intent].extend(_examples)

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
    # Which zone is named says nothing about what the farmer wants, so every grid
    # cell reads as the same word; otherwise "r0c2" leans towards whichever intent
    # happened to use it most in the training phrases.
    t = GRID_RE.sub(" zone ", t)
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
# The field is the spraying pipeline's grid: 3 x 3 cells of 20 m x 20 m, named
# R<row>C<col> exactly as on the dashboard and in spraying_map.json. The pipeline counts
# rows northwards from the field origin, so row 0 is the south edge and row 2 the north
# edge (the dashboard draws row 2 at the top); column 0 is the west edge. A farmer can
# also say "the north side", "the east side" or "the north-east corner".
GRID_ROWS, GRID_COLS, CELL_M2 = 3, 3, 400
ZONES = {f"R{r}C{c}": {"crop": "maize", "m2": CELL_M2, "row": r, "col": c}
         for r in range(GRID_ROWS) for c in range(GRID_COLS)}
CROPS = {"maize": r"maize|corn|mealies?", "tomato": r"tomato(?:es)?"}
# The model's classes (CLASS_NAMES in model/train_deploy_demo.py), as the farmer-facing
# display names. ORDER MATTERS: extract_disease() returns the first match, so a longer
# name must come before any shorter name it contains ("septoria leaf spot" before
# "leaf spot", "leaf blight" before plain "blight"). The dataset's folder typos
# ("grasshoper", "verticulium") are accepted as input spellings.
DISEASES = [
    ("Septoria leaf spot", r"septoria"),
    ("Verticillium wilt", r"vertic[iu]l+ium|\bwilt(?:s|ed|ing)?\b"),
    ("Fall armyworm", r"\b(?:fall\s+)?army\s*worms?\b|\bfaw\b"),
    ("Streak virus", r"\bstreak(?:\s+virus)?\b|\bmsv\b"),
    ("Leaf curl", r"\bleaf\s+curl\b|\bcurl(?:ed|ing)?\s+leaves\b|\btylcv\b"),
    ("Leaf beetle", r"\b(?:leaf\s+)?beetles?\b"),
    ("Grasshopper", r"\bgrass\s*hop+ers?\b"),
    ("Leaf blight", r"\b(?:leaf\s+)?blight\b"),
    ("Leaf spot", r"\b(?:gr[ae]y\s+)?leaf\s+spots?\b"),
    ("Healthy", r"\bhealthy\b"),
]
# Plain-language descriptions. Sources for every entry: disease_info_sources.md.
DISEASE_INFO = {
    "Fall armyworm": "Fall armyworm is a caterpillar that feeds on young maize. Small caterpillars scrape the "
                     "leaf surface and leave thin, papery \"windows\"; bigger ones chew ragged holes, hide in "
                     "the whorl and can bore into the cob.",
    "Grasshopper": "Grasshoppers chew ragged holes in maize leaves; in bad cases only the midrib is left, and "
                   "they can also clip the silks. Damage usually starts at field edges next to grass or weeds "
                   "and is worse in dry years.",
    "Leaf beetle": "Leaf beetles are small beetles whose adults chew and shred maize leaves, often in patches "
                   "near field edges. In some species the larvae live in the soil and feed on roots, which can "
                   "kill young plants.",
    "Leaf blight": "On maize, leaf blight shows as long grey-green to tan lesions on the leaves, usually "
                   "starting on the lower leaves; northern corn leaf blight is the best-known example, and its "
                   "fungus survives on old maize residue. On tomato, it shows as dark spots, often with rings "
                   "like a target and a yellow edge, starting on the older leaves; early blight is the "
                   "best-known example, favoured by warm, wet weather. The scan does not say which blight it "
                   "is, so ask an extension officer to confirm.",
    "Leaf spot": "Leaf spot is a general name for fungal diseases that leave spots or lesions on maize leaves. "
                 "Gray leaf spot, for example, shows as narrow, rectangular, tan-to-grey lesions between the "
                 "leaf veins, starting on the lower leaves. The scan does not say which fungus, so ask an "
                 "extension officer to confirm.",
    "Streak virus": "Maize streak virus is spread by small leafhoppers. Leaves get narrow white-to-yellow "
                    "streaks along the veins, and early infection can stunt the plant and spoil the cobs. There "
                    "is no chemical cure for the virus, so resistant varieties and keeping leafhopper numbers "
                    "down are the main defences.",
    "Septoria leaf spot": "Septoria leaf spot is a fungal disease of tomato that makes many small, round spots "
                          "with a dark edge and a grey or tan centre, starting on the lowest leaves. It spreads "
                          "in warm, humid weather when water splashes spores up from the soil.",
    "Verticillium wilt": "Verticillium wilt is a fungal disease of tomato that lives in the soil. Leaves turn "
                         "yellow, often in a wedge shape, and wilt, starting with the lower leaves and often on "
                         "one side of the plant; cutting the stem near the soil shows tan streaks inside.",
    "Leaf curl": "Leaf curl in tomato is best known from tomato yellow leaf curl virus, which whiteflies carry: "
                 "young leaves turn yellow and curl, plants become stunted and bushy, and flowers drop before "
                 "fruit sets. Nutrient problems and herbicide damage can look similar, so ask an extension "
                 "officer to confirm.",
    "Healthy": "Healthy means the system found no sign of the pests or diseases it checks for on that leaf. "
               "That is not a guarantee, so keep scouting and look closely at any plant that seems different.",
}


def default_scans(today):
    """MOCK scan history, newest first. Replace with real results-database calls.

    The latest scan matches the dashboard demo (farmer_dashboard.html): R2C2 is the
    high-priority Fall armyworm zone (88.6% confidence, health 21.4%) and R2C1 the
    standard Leaf spot zone (82.4%, health 58.3%). Affected area is 100 minus the
    dashboard's health index, and the earlier scans follow its weekly history.
    """
    def det(disease, conf, pct, photos, second=None, second_conf=0.0, blurry=0):
        return {"disease": disease, "conf": conf, "affected_pct": pct,
                "second": second, "second_conf": second_conf, "photos": photos, "blurry": blurry}
    faw, spot = "Fall armyworm", "Leaf spot"
    # Dates are anchored to last week's Monday so "last week" always shows the same story.
    last_mon = today - timedelta(days=today.weekday()) - timedelta(days=7)
    off = lambda d: (today - d).days
    spec = [
        (0, "07:42", [], None, {
            "R2C2": det(faw, .886, 79, 14, "Leaf beetle", .05),
            "R2C1": det(spot, .824, 42, 12, "Leaf blight", .12, blurry=1)}),
        (off(last_mon + timedelta(days=5)), "07:51", [], None, {          # Saturday
            "R2C2": det(faw, .87, 72, 14), "R2C1": det(spot, .79, 38, 12)}),
        (off(last_mon + timedelta(days=3)), "07:20", ["R0C2"], "the drone battery ran low", {   # Thursday
            "R2C2": det(faw, .85, 64, 13), "R2C1": det(spot, .74, 34, 12)}),
        (off(last_mon + timedelta(days=1)), "07:44", [], None, {"R2C2": det(faw, .82, 55, 13)}),  # Tuesday
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
# A grid cell ("R2C2", "r2 c1", "zone r0c2") or a direction on the grid. A single
# direction is a side of the field ("north side" = row 2); two directions together
# are a corner cell ("north east" = R2C2); "centre" is the middle cell.
GRID_RE = re.compile(r"\br\s*([0-9])\s*c\s*([0-9])\b", re.I)
DIR_RE = re.compile(
    r"\b(?:(north|south|centre|center)(?:[\s-]*(east|west))?|(east|west))\b"
    r"(?:\s+(?:zone|field|block|section|area|part|side|edge|corner|row|column))?", re.I)
SIDES = {"north": ("row", GRID_ROWS - 1), "south": ("row", 0),
         "west": ("col", 0), "east": ("col", GRID_COLS - 1)}


def _direction_zone(ns, ew):
    """Map one or two direction words onto the grid: a cell name or a '<dir> side' label."""
    ns = "centre" if ns == "center" else ns
    if ns == "centre" and not ew:
        return f"R{GRID_ROWS // 2}C{GRID_COLS // 2}"
    if ns and ew:
        r = SIDES[ns][1] if ns in SIDES else GRID_ROWS // 2
        return f"R{r}C{SIDES[ew][1]}"
    return f"{ns or ew} side"


def extract_zone(text):
    found = []
    for m in GRID_RE.finditer(text):
        name = f"R{m.group(1)}C{m.group(2)}"
        found.append(name if name in ZONES else f"unknown {name}")
    for m in DIR_RE.finditer(GRID_RE.sub(" ", text)):
        ns, ew = (m.group(1) or "").lower(), (m.group(2) or m.group(3) or "").lower()
        found.append(_direction_zone(ns, ew))
    found = list(dict.fromkeys(found))
    return None if not found else (found[0] if len(found) == 1 else found)


def resolve_zones(entity):
    """Turn a zone entity (cell name, 'north side' label, or list) into real zone names."""
    if entity is None:
        return []
    items = [entity] if isinstance(entity, str) else list(entity)
    out = []
    for it in items:
        if it in ZONES:
            out.append(it)
        elif it.endswith(" side"):
            axis, idx = SIDES[it.split()[0]]
            out += [z for z, v in ZONES.items() if v[axis] == idx]
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
HELP_TEXT = ("Hello, I'm AgriAssist. I report what the AI found in your fields: possible pests and diseases, zones "
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
        leftover = [w for w in DIR_RE.sub(" ", GRID_RE.sub(" ", preprocess(text))).split()
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
                      "\"I checked R2C2\"." if intent == "mark_reviewed" else
                      "Do you want me to flag a result as wrong? If so, tell me which zone, for example "
                      "\"the R2C2 result is wrong\".")
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
        total = sum(z["m2"] for z in ZONES.values())
        missed = sum(ZONES[z]["m2"] for z in scan["uncovered"])
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
        # "Is it healthy?" asks about every zone, so "Healthy" is never a filter.
        if ent and ent.get("disease") and ent["disease"] != "Healthy":
            out = [(z, d) for z, d in out if d["disease"] == ent["disease"]]
        return out

    @staticmethod
    def scope(ent):
        if ent["zone"] and isinstance(ent["zone"], str):
            return f"the {ent['zone']}" if ent["zone"].endswith("side") else ent["zone"]
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
        ent = parsed["entities"]
        if ent["crop"] and parsed["intent"] in VERIFY_INTENTS and not self.select({**ent, "zone": None}):
            asked = ent["crop"] if isinstance(ent["crop"], str) else join_and(ent["crop"])
            grown = join_and(sorted({v["crop"] for v in ZONES.values()}))
            return (f"The scanned field has no {asked} zones: all {len(ZONES)} zones are {grown}. "
                    "Ask me about those zones, or about a disease by name."), {"no_detections": True}
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
        what = (f"one possible pest or disease: {names[0]}" if len(names) == 1
                else "possible pests or diseases: " + join_and(names))
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
                 f"{len(dets)} of {len(sel)} zones show a possible pest or disease."]
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
            m2 = ZONES[z]["m2"] * d["affected_pct"] / 100
            hit += m2
            lines.append(f"{z}: about {d['affected_pct']}% of the zone ({m2:.0f} of {ZONES[z]['m2']} square "
                         f"metres), {d['disease']}, {d['conf']:.0%} confidence")
        total = sum(ZONES[z]["m2"] for z in sel)
        lines.append(f"Overall about {100 * hit / total:.1f}% of {self.scope(ent)} ({hit:.0f} of {total:.0f} "
                     "square metres). These are estimates from the images.")
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
    print("Commands: /json toggles the parsed query, /focus R2C1 simulates a dashboard selection, /reset.\n")
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
    ("2. Spray map and review", None, ["Which zones need spraying?", "I checked R2C1 and R2C2 and they look right."]),
    ("3. Confidence (dashboard focus: R2C1)", "R2C1",
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
