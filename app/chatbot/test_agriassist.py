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


def _model_class_names():
    """CLASS_NAMES as written in the training script (read as text, so torch is not needed)."""
    import ast, pathlib
    src = (pathlib.Path(__file__).resolve().parents[2] / "model" / "train_deploy_demo.py").read_text(encoding="utf-8")
    block = src.split("CLASS_NAMES: List[str] = ", 1)[1].split("]", 1)[0] + "]"
    return ast.literal_eval(block)


MODEL_CLASS_NAMES = _model_class_names()


def new_bot(scans=None):
    return aa.AgriAssist(today=TODAY, model=MODEL, scans=scans)


# Questions about spraying in varied wording (the document's S1/S2/S6 test list).
SPRAY_QUESTIONS = [
    "do I spray?", "is it time to treat?", "ok to use chemicals?", "what fungicide should I buy",
    "how much pesticide per hectare", "can I apply something today", "tell me to spray r2c2",
    "spray everything?", "should I spray the maize", "I want to spray tomorrow, is that fine",
    "sprayin the whole field ok?", "treat the field?", "what dose of fungicide", "which pesticide kills armyworm",
    "how do I treat leaf spot", "is it safe to spray now", "must I spray r0c2",
    "do I need to treat the north zone", "order pesticide", "recommend a chemical",
    "can I use herbicide", "spray or not", "what should I treat it with", "when should I spray",
    "apply fungicide to r2c2?", "should I treat leaf blight", "is spraying necessary",
    "what do I spray on grasshoppers", "how many litres should I spray", "just spray it",
    "Which zones need spraying?", "Just tell me to spray the whole field.", "What should I spray?",
]
PRODUCT_OR_DOSE = re.compile(
    r"\b(mancozeb|chlorothalonil|azoxystrobin|propiconazole|copper|glyphosate|\d+\s?(?:ml|l|kg|g|litres?)\b|"
    r"per hectare|/ha|mixing ratio)", re.I)

# Held-out test set. These phrasings were written AFTER the training data was finalised and are
# not in the training data. (Earlier test phrasings were moved into training, so do not re-use them.)
HELD_OUT = [
    ('what exactly is wrong with the plants in my north field', 'identify_disease'),
    ('has the drone found any disease on my maize', 'identify_disease'),
    ('are my tomatoes suffering from something', 'identify_disease'),
    ('what kind of disease is it', 'identify_disease'),
    ('did the AI detect disease in the tomatoes', 'identify_disease'),
    ("what's making the leaves look bad", 'identify_disease'),
    ('is there a disease on r0c2', 'identify_disease'),
    ('tell me how the farm is doing', 'field_health_summary'),
    ("how's the north field today", 'field_health_summary'),
    ('is everything fine with my crops', 'field_health_summary'),
    ('give me a health check of the whole farm', 'field_health_summary'),
    ('how are the tomatoes holding up', 'field_health_summary'),
    ('what is the status of my maize', 'field_health_summary'),
    ('how do my fields look after the latest scan', 'field_health_summary'),
    ('where in the field is the disease', 'disease_location'),
    ('which zones is the leaf spot in', 'disease_location'),
    ('show me the areas that are infected', 'disease_location'),
    ('where did the AI see the problem', 'disease_location'),
    ('can you point out the sick zones', 'disease_location'),
    ('in which zones was armyworm found', 'disease_location'),
    ('where are the infected plants located', 'disease_location'),
    ('how big is the infected part of r2c2', 'get_affected_area'),
    ('what percent of the maize is affected', 'get_affected_area'),
    ('how much of r0c2 is sick', 'get_affected_area'),
    ('how many hectares are infected in total', 'get_affected_area'),
    ('how serious is the spread in the field', 'get_affected_area'),
    ('what is the size of the diseased area', 'get_affected_area'),
    ('how much of the farm has disease', 'get_affected_area'),
    ('which zones need to be sprayed', 'show_spray_map'),
    ('show me the map for spraying', 'show_spray_map'),
    ('where should I treat', 'show_spray_map'),
    ('give me the treatment map', 'show_spray_map'),
    ('what areas are on the spray list', 'show_spray_map'),
    ('open the spray map for the north field', 'show_spray_map'),
    ('zones that need pesticide', 'show_spray_map'),
    ('what should I put on the maize', 'request_treatment_advice'),
    ('should I spray the north field', 'request_treatment_advice'),
    ('which pesticide do I buy', 'request_treatment_advice'),
    ('how do I cure this', 'request_treatment_advice'),
    ('tell me to go ahead and spray', 'request_treatment_advice'),
    ('what is the correct dose to apply', 'request_treatment_advice'),
    ('is it okay to spray now', 'request_treatment_advice'),
    ('how sure is the AI about r0c2', 'get_confidence'),
    ('what is the confidence for this detection', 'get_confidence'),
    ('can I trust the result for r2c2', 'get_confidence'),
    ('how accurate is the AI on this one', 'get_confidence'),
    ('is the AI confident in this', 'get_confidence'),
    ('tell me how reliable this is', 'get_confidence'),
    ('how certain is the AI that it is fall armyworm', 'get_confidence'),
    ('why was r2c2 flagged', 'explain_result'),
    ('show me the photos the AI looked at', 'explain_result'),
    ('what did the AI base this on', 'explain_result'),
    ('can I see the images for r0c2', 'explain_result'),
    ('why does the AI think it is leaf spot', 'explain_result'),
    ('show me the evidence for this result', 'explain_result'),
    ('explain the reasoning behind this detection', 'explain_result'),
    ('I have checked r2c2 and it is right', 'mark_reviewed'),
    ('mark r2c1 as reviewed', 'mark_reviewed'),
    ('I reviewed r0c2 and agree with the AI', 'mark_reviewed'),
    ('I confirm I looked at the north zone', 'mark_reviewed'),
    ('I went to r2c2 and the result matches', 'mark_reviewed'),
    ('please record that I checked r0c2', 'mark_reviewed'),
    ('I verified r2c1 in the field', 'mark_reviewed'),
    ('this result is not right', 'flag_incorrect'),
    ('the AI got r0c2 wrong', 'flag_incorrect'),
    ('flag the r2c1 result as wrong', 'flag_incorrect'),
    ('the plants there are healthy, the AI is mistaken', 'flag_incorrect'),
    ('this detection is incorrect', 'flag_incorrect'),
    ('the zone on the map is in the wrong place', 'flag_incorrect'),
    ('I disagree, that is not leaf spot', 'flag_incorrect'),
    ('show me the results from last week', 'show_history'),
    ('what did the AI find on 29 september', 'show_history'),
    ('let me see scans from the last two weeks', 'show_history'),
    ('results from yesterday please', 'show_history'),
    ('show me what was detected last month', 'show_history'),
    ('what were the findings three days ago', 'show_history'),
    ('pull up the previous results', 'show_history'),
    ('is the disease spreading', 'compare_over_time'),
    ('is the field better than last week', 'compare_over_time'),
    ('has it got worse since the previous scan', 'compare_over_time'),
    ('compare the latest scan with last month', 'compare_over_time'),
    ('is r2c2 improving', 'compare_over_time'),
    ('how has it changed since last time', 'compare_over_time'),
    ('are more zones infected than before', 'compare_over_time'),
    ('what is leaf spot', 'get_disease_info'),
    ('tell me about streak virus', 'get_disease_info'),
    ('how does septoria leaf spot affect tomatoes', 'get_disease_info'),
    ('what are the signs of leaf spot', 'get_disease_info'),
    ('how does fall armyworm spread', 'get_disease_info'),
    ('explain what blight is', 'get_disease_info'),
    ('what causes leaf curl', 'get_disease_info'),
    ('when was the last scan done', 'scan_status'),
    ('did the drone finish scanning the whole farm', 'scan_status'),
    ('how many photos were taken in the last scan', 'scan_status'),
    ('was any zone missed', 'scan_status'),
    ('how up to date is this data', 'scan_status'),
    ('did the scan cover everything', 'scan_status'),
    ('when did the drone last fly', 'scan_status'),
    ('hello', 'smalltalk_help'),
    ('what can you do for me', 'smalltalk_help'),
    ('help me please', 'smalltalk_help'),
    ('thank you', 'smalltalk_help'),
    ('how do I use you', 'smalltalk_help'),
    ('hi there agriassist', 'smalltalk_help'),
    ('what questions can you answer', 'smalltalk_help'),
    ('what is the price of maize today', 'out_of_scope'),
    ('is it going to rain tomorrow', 'out_of_scope'),
    ('can you recommend a good restaurant', 'out_of_scope'),
    ('how do I plant tomatoes', 'out_of_scope'),
    ('what is the best fertilizer for maize', 'out_of_scope'),
    ('tell me a joke', 'out_of_scope'),
    ('who is the president', 'out_of_scope'),
    ('how do I register for a government subsidy', 'out_of_scope'),
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
        scans[0]["detections"]["R1C1"] = {"disease": "Leaf blight", "conf": 0.45, "affected_pct": 3,
                                          "second": None, "second_conf": 0.0, "photos": 8, "blurry": 0}
        reply, _ = new_bot(scans).ask("Which zones need spraying?")
        self.assertIn("R1C1: Leaf blight, 45% confidence (low)", reply)
        self.assertIn("not confident about R1C1", reply)
        self.assertIn("do not rely", reply)
        self.assertIn("moderately confident about R2C1", reply)

    def test_spray_map_is_pending_until_farmer_confirms(self):
        bot = new_bot()
        reply, _ = bot.ask("Which zones need spraying?")
        self.assertIn("pending your review", reply)
        self.assertEqual(bot.review, {})
        bot.ask("I checked R2C2")
        self.assertEqual(bot.review, {"R2C2": "reviewed"})
        reply, _ = bot.ask("Show me the spray map")
        self.assertIn("R2C2: Fall armyworm, 89% confidence (high), reviewed by you", reply)
        self.assertIn("R2C1: Leaf spot, 82% confidence (medium), pending your review", reply)

    def test_spray_override_when_classifier_is_unsure(self):
        """A spray question that the classifier misreads must still get the safe response."""
        reply, parsed = new_bot().ask("pesticide zzz")
        self.assertEqual(parsed["intent"], "request_treatment_advice")
        self.assertIn("product label", reply)

    def test_data_gaps_are_disclosed(self):
        reply, _ = new_bot().ask("What's wrong with my maize field?")
        self.assertIn("slightly blurry in R2C1 (1 of 12)", reply)
        scans = aa.default_scans(TODAY)
        scans[0]["uncovered"] = ["R1C1"]
        reply, _ = new_bot(scans).ask("What's wrong with my maize field?")
        self.assertIn("did not cover R1C1", reply)

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
        self.assertEqual(parsed["entities"]["zone"], "north side")

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
        bot.focus("R2C1")
        reply, p = bot.ask("flag this result")
        self.assertEqual(p["context_ref"], "R2C1")
        self.assertEqual([f["zone"] for f in bot.flags], ["R2C1"])


class EntityTests(unittest.TestCase):
    def ent(self, text):
        return aa.extract_entities(text, TODAY)

    def test_zone(self):
        self.assertEqual(self.ent("Show the spray map for R2C2")["zone"], "R2C2")
        self.assertEqual(self.ent("I checked r2c1 and R2C2")["zone"], ["R2C1", "R2C2"])
        self.assertEqual(self.ent("how is the north side")["zone"], "north side")
        self.assertEqual(self.ent("what about the north-east corner")["zone"], "R2C2")
        self.assertEqual(self.ent("the south west corner")["zone"], "R0C0")
        self.assertEqual(self.ent("what about the centre")["zone"], "R1C1")
        self.assertIsNone(self.ent("which zones need spraying")["zone"])
        self.assertIsNone(self.ent("in the middle of the season")["zone"])

    def test_crop_and_disease(self):
        e = self.ent("is there armyworm on my corn")
        self.assertEqual((e["crop"], e["disease"]), ("maize", "Fall armyworm"))
        self.assertEqual(self.ent("leaf spot in the tomatoes")["crop"], "tomato")
        self.assertEqual(self.ent("grey leaf spot")["disease"], "Leaf spot")
        self.assertEqual(self.ent("grasshoper damage")["disease"], "Grasshopper")        # dataset typo
        self.assertEqual(self.ent("verticulium wilt")["disease"], "Verticillium wilt")  # dataset typo

    def test_every_model_class_extracts_to_itself(self):
        """Longer names must not be shadowed by shorter ones (septoria leaf spot vs leaf spot)."""
        for cls in MODEL_CLASS_NAMES:
            with self.subTest(cls=cls):
                self.assertEqual(aa.extract_disease(cls), cls.split(" (")[0])
                self.assertIn(cls.split(" (")[0], aa.DISEASE_INFO)

    def test_resolve_zones(self):
        self.assertEqual(aa.resolve_zones("north side"), ["R2C0", "R2C1", "R2C2"])
        self.assertEqual(aa.resolve_zones("west side"), ["R0C0", "R1C0", "R2C0"])
        self.assertEqual(aa.resolve_zones(["R2C2", "R0C2"]), ["R2C2", "R0C2"])

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
        self.assertIn("89% of the field", reply)
        self.assertIn("compare with care", reply)

    def test_history_asks_for_period(self):
        _, p = new_bot().ask("show me previous results")
        self.assertTrue(p["needs_clarification"])

    def test_confidence_uses_context(self):
        bot = new_bot()
        bot.focus("R2C1")
        reply, p = bot.ask("How confident is the AI about this result?")
        self.assertEqual(p["context_ref"], "R2C1")
        self.assertIn("82% confident", reply)
        self.assertIn("Leaf blight (12%)", reply)

    def test_compare_and_info_and_status(self):
        bot = new_bot()
        self.assertIn("Comparing the latest scan", bot.ask("Has the leaf spot spread since the last scan?")[0])
        self.assertIn("caterpillar", bot.ask("What is fall armyworm?")[0])
        self.assertIn("no tomato zones", bot.ask("What's wrong with my tomatoes?")[0])
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
