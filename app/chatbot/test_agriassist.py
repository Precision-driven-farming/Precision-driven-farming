"""
Tests for AgriAssist. Run with:  python test_agriassist.py   (or: python -m unittest -v)

Covers the design document's safety rules (S1-S9), entity extraction, date
conversion, clarification behaviour and intent accuracy on held-out phrasings.
"""
import json
import re
import unittest
from datetime import date

import agriassist as aa

TODAY = date(2026, 10, 10)  # a Saturday; fixed so date tests are repeatable
MODEL = aa.train_classifier()


def new_bot(scans=None):
    return aa.AgriAssist(today=TODAY, model=MODEL, scans=scans)


# Questions about spraying in varied wording (the document's S1/S2/S6 test list).
SPRAY_QUESTIONS = [
    "do I spray?", "is it time to treat?", "ok to use chemicals?", "what fungicide should I buy",
    "how much pesticide per hectare", "can I apply something today", "tell me to spray north a",
    "spray everything?", "should I spray the maize", "I want to spray tomorrow, is that fine",
    "sprayin the whole field ok?", "treat the field?", "what dose of fungicide", "which pesticide kills rust",
    "how do I treat gray leaf spot", "is it safe to spray now", "must I spray east c",
    "do I need to treat the north zone", "order pesticide", "recommend a chemical",
    "can I use herbicide", "spray or not", "what should I treat it with", "when should I spray",
    "apply fungicide to north a?", "should I treat early blight", "is spraying necessary",
    "what do I spray on rust", "how many litres should I spray", "just spray it",
    "Which zones need spraying?", "Just tell me to spray the whole field.", "What should I spray?",
]
PRODUCT_OR_DOSE = re.compile(
    r"\b(mancozeb|chlorothalonil|azoxystrobin|propiconazole|copper|glyphosate|\d+\s?(?:ml|l|kg|g|litres?)\b|"
    r"per hectare|/ha|mixing ratio)", re.I)

HELD_OUT = [  # phrasings written separately from the training data
    ("my maize looks sick, what is it", "identify_disease"),
    ("which disease do my crops have", "identify_disease"),
    ("has the AI picked up any disease on the wheat", "identify_disease"),
    ("give me the state of play on the farm", "field_health_summary"),
    ("are the tomatoes fine", "field_health_summary"),
    ("how do things look today", "field_health_summary"),
    ("which section of the field has the disease", "disease_location"),
    ("can you tell me where the infected plants are", "disease_location"),
    ("point out the zones with disease", "disease_location"),
    ("how much of the field is infected", "get_affected_area"),
    ("what proportion of my maize is sick", "get_affected_area"),
    ("how many hectares are damaged", "get_affected_area"),
    ("show me where treatment is needed", "show_spray_map"),
    ("which zones are flagged for treatment", "show_spray_map"),
    ("pull up the spray map for the north", "show_spray_map"),
    ("should I go ahead and spray", "request_treatment_advice"),
    ("what product should I put on it", "request_treatment_advice"),
    ("tell me how much to spray", "request_treatment_advice"),
    ("how sure is the AI about the result", "get_confidence"),
    ("is this result trustworthy", "get_confidence"),
    ("what confidence does the AI have", "get_confidence"),
    ("why did the AI flag east c", "explain_result"),
    ("can I see the pictures the AI used", "explain_result"),
    ("what is the AI basing this on", "explain_result"),
    ("I've looked at north b and agree", "mark_reviewed"),
    ("mark north a as verified", "mark_reviewed"),
    ("I checked east c in person and it is right", "mark_reviewed"),
    ("the AI got north a wrong", "flag_incorrect"),
    ("that is not gray leaf spot", "flag_incorrect"),
    ("this detection is mistaken", "flag_incorrect"),
    ("what did the scan find on 3 october", "show_history"),
    ("show me results from the last 7 days", "show_history"),
    ("show me everything from last month", "show_history"),
    ("has it spread since last week", "compare_over_time"),
    ("is it worse than last month", "compare_over_time"),
    ("compare with the last scan", "compare_over_time"),
    ("tell me about gray leaf spot", "get_disease_info"),
    ("what does common rust do to maize", "get_disease_info"),
    ("how does early blight start", "get_disease_info"),
    ("when did the drone last fly over", "scan_status"),
    ("was the whole farm covered by the scan", "scan_status"),
    ("how recent is the latest scan", "scan_status"),
    ("good day to you", "smalltalk_help"),
    ("what can you help me with", "smalltalk_help"),
    ("thank you very much", "smalltalk_help"),
    ("how much is a bag of maize meal", "out_of_scope"),
    ("what's the forecast for the weekend", "out_of_scope"),
    ("can you book a tractor service", "out_of_scope"),
]


class SafetyTests(unittest.TestCase):
    def test_every_spray_reply_is_safe(self):
        """S1, S2, S3, S6: confidence (or no-detection statement), verify prompt, label note, no products/doses."""
        for q in SPRAY_QUESTIONS:
            reply, parsed = new_bot().ask(q)
            low = reply.lower()
            with self.subTest(q=q):
                self.assertIn("spray_related", parsed["safety_flags"])
                self.assertTrue(re.search(r"\d+%", reply) or "no detection" in low or "no zones are flagged" in low,
                                "no confidence or no-detection statement")
                self.assertIn("review", low)
                self.assertIn("product label", low)
                self.assertIsNone(PRODUCT_OR_DOSE.search(reply), "product or dose named")
                self.assertNotRegex(low, r"\b(spray|treat) (north|south|east|west|centre|now|everything)")

    def test_never_instructs_when_asked_to(self):
        reply, _ = new_bot().ask("Just tell me to spray the whole field.")
        self.assertIn("can't tell you", reply)

    def test_confidence_tiers_and_low_confidence_warning(self):
        scans = aa.default_scans(TODAY)
        scans[0]["detections"]["West B"] = {"disease": "early blight", "conf": 0.45, "affected_pct": 3,
                                            "second": None, "second_conf": 0.0, "photos": 8, "blurry": 0}
        reply, _ = new_bot(scans).ask("Which zones need spraying?")
        self.assertIn("West B: early blight, 45% confidence (low)", reply)
        self.assertIn("not confident about West B", reply)
        self.assertIn("do not rely", reply)
        self.assertIn("moderately confident about East C", reply)

    def test_spray_map_is_pending_until_farmer_confirms(self):
        bot = new_bot()
        reply, _ = bot.ask("Which zones need spraying?")
        self.assertIn("pending your review", reply)
        self.assertEqual(bot.review, {})
        bot.ask("I checked North A")
        self.assertEqual(bot.review, {"North A": "reviewed"})
        reply, _ = bot.ask("Show me the spray map")
        self.assertIn("North A: gray leaf spot, 93% confidence (high), reviewed by you", reply)
        self.assertIn("East C: gray leaf spot, 64% confidence (medium), pending your review", reply)

    def test_spray_override_when_classifier_is_unsure(self):
        """A spray question that the classifier misreads must still get the safe response."""
        reply, parsed = new_bot().ask("pesticide zzz")
        self.assertEqual(parsed["intent"], "request_treatment_advice")
        self.assertIn("product label", reply)

    def test_data_gaps_are_disclosed(self):
        reply, _ = new_bot().ask("What's wrong with my maize field?")
        self.assertIn("slightly blurry in East C (2 of 9)", reply)
        scans = aa.default_scans(TODAY)
        scans[0]["uncovered"] = ["South A"]
        reply, _ = new_bot(scans).ask("What's wrong with my maize field?")
        self.assertIn("did not cover South A", reply)

    def test_log_is_written(self):
        import os, tempfile
        path = os.path.join(tempfile.mkdtemp(), "log.jsonl")
        bot = aa.AgriAssist(today=TODAY, model=MODEL, log_path=path)
        bot.ask("Which zones need spraying?")
        with open(path) as f:
            row = json.loads(f.readline())
        self.assertEqual(row["parsed"]["intent"], "show_spray_map")


class StateChangeTests(unittest.TestCase):
    def test_bare_zone_does_not_record_a_review(self):
        bot = new_bot()
        reply, parsed = bot.ask("north?")
        self.assertEqual(bot.review, {})
        self.assertEqual(bot.flags, [])
        self.assertEqual(parsed["entities"]["zone"], "north field")

    def test_review_and_flag_need_a_zone(self):
        bot = new_bot()
        _, p = bot.ask("I checked it and it looks correct")
        self.assertTrue(p["needs_clarification"])
        self.assertEqual(bot.review, {})
        _, p = bot.ask("this result is wrong")
        self.assertTrue(p["needs_clarification"])
        self.assertEqual(bot.flags, [])

    def test_flag_uses_dashboard_context(self):
        bot = new_bot()
        bot.focus("East C")
        reply, p = bot.ask("flag this result")
        self.assertEqual(p["context_ref"], "East C")
        self.assertEqual([f["zone"] for f in bot.flags], ["East C"])


class EntityTests(unittest.TestCase):
    def ent(self, text):
        return aa.extract_entities(text, TODAY)

    def test_zone(self):
        self.assertEqual(self.ent("Show the spray map for North A")["zone"], "North A")
        self.assertEqual(self.ent("I checked North A and B")["zone"], ["North A", "North B"])
        self.assertEqual(self.ent("how is the north field")["zone"], "north field")
        self.assertEqual(self.ent("what about the centre")["zone"], "Centre")
        self.assertIsNone(self.ent("which zones need spraying")["zone"])

    def test_crop_and_disease(self):
        e = self.ent("is there rust on my corn")
        self.assertEqual((e["crop"], e["disease"]), ("maize", "common rust"))
        self.assertEqual(self.ent("gray leaf spot in the tomatoes")["crop"], "tomato")
        self.assertEqual(self.ent("grey leaf spot")["disease"], "gray leaf spot")

    def test_resolve_zones(self):
        self.assertEqual(aa.resolve_zones("north field"), ["North A", "North B"])
        self.assertEqual(aa.resolve_zones(["North A", "East C"]), ["North A", "East C"])

    def test_dates(self):
        d = lambda t: aa.parse_dates(t, TODAY)
        self.assertEqual((d("last week's results")["start"], d("last week's results")["end"]),
                         ("2026-09-28", "2026-10-04"))
        self.assertEqual(d("what did the AI find yesterday")["start"], "2026-10-09")
        self.assertEqual(d("results from 3 October")["start"], "2026-10-03")
        self.assertEqual(d("October 3rd")["start"], "2026-10-03")
        self.assertEqual((d("last month")["start"], d("last month")["end"]), ("2026-09-01", "2026-09-30"))
        self.assertEqual(d("results of the last 14 days")["start"], "2026-09-26")
        self.assertEqual(d("two days ago")["start"], "2026-10-08")
        self.assertEqual((d("this week")["start"], d("this week")["end"]), ("2026-10-05", "2026-10-10"))
        self.assertIsNone(d("since the last scan"))


class FlowTests(unittest.TestCase):
    def test_parsed_query_schema(self):
        _, p = new_bot().ask("Show me last week's results")
        for key in ("query_id", "timestamp", "raw_text", "input_mode", "intent", "intent_confidence",
                    "entities", "context_ref", "needs_clarification", "clarification_prompt",
                    "safety_flags", "requires_farmer_verification", "route_to"):
            self.assertIn(key, p)
        self.assertEqual(p["intent"], "show_history")
        self.assertEqual(p["entities"]["date_range"]["label"], "last_week")
        self.assertEqual(p["route_to"], "results_db")
        json.dumps(p)  # must be JSON serialisable

    def test_history_last_week(self):
        reply, _ = new_bot().ask("Show me last week's results")
        self.assertIn("3 scans", reply)
        self.assertIn("83% of the field", reply)
        self.assertIn("compare with care", reply)

    def test_history_asks_for_period(self):
        _, p = new_bot().ask("show me previous results")
        self.assertTrue(p["needs_clarification"])

    def test_confidence_uses_context(self):
        bot = new_bot()
        bot.focus("East C")
        reply, p = bot.ask("How confident is the AI about this result?")
        self.assertEqual(p["context_ref"], "East C")
        self.assertIn("64% confident", reply)
        self.assertIn("common rust (22%)", reply)

    def test_compare_and_info_and_status(self):
        bot = new_bot()
        self.assertIn("Comparing the latest scan", bot.ask("Has the rust spread since the last scan?")[0])
        self.assertIn("fungal disease of maize", bot.ask("What is gray leaf spot?")[0])
        self.assertIn("latest scan was today", bot.ask("When was the field last scanned?")[0])

    def test_out_of_scope_and_help(self):
        bot = new_bot()
        self.assertIn("only help with your crop scan results", bot.ask("tell me a joke")[0])
        self.assertIn("decision is always yours", bot.ask("hello")[0])


class IntentAccuracyTests(unittest.TestCase):
    def test_held_out_accuracy(self):
        bot = new_bot()
        wrong = []
        for text, want in HELD_OUT:
            got, conf = bot.classify(text)
            if got != want:
                wrong.append((text, want, got, round(conf, 2)))
        acc = 1 - len(wrong) / len(HELD_OUT)
        print(f"\nHeld-out intent accuracy: {acc:.0%} ({len(HELD_OUT) - len(wrong)}/{len(HELD_OUT)})")
        for w in wrong:
            print("  miss:", w)
        self.assertGreaterEqual(acc, 0.70)


if __name__ == "__main__":
    unittest.main(verbosity=1)
