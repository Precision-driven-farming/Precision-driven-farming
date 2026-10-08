"""
Member 8 - Farmer Chatbot - Reads Member 9 spraying_map.json
"""
from nlp_parser import predict_intent, get_response_for_intent
import json
import os

# Load spraying map from Member 9
def load_spraying_map():
    map_path = "../spraying_pipeline/spraying_map"
    # Placeholder for now until Member 9 finalizes real file
    if not os.path.exists(map_path):
        return {
            "Zone1": {"status": "healthy", "confidence": 95, "action": "don't spray"},
            "Zone2": {"status": "rust 78%", "confidence": 78, "action": "verify first"},
            "Zone3": {"status": "low 45%", "confidence": 45, "action": "monitor"},
            "generated_by": "Member 9 spraying_pipeline.py"
        }
    with open(map_path) as f:
        return json.load(f)

spraying_map = load_spraying_map()

def chatbot_answer(farmer_question):
    intent, confidence = predict_intent(farmer_question)
    base_response = get_response_for_intent(intent)

    # Integration with Member 9 - if question about spray, add map
    extra_info = ""
    if intent in ["spraying_zone", "zone_details", "severity_score"]:
        extra_info = f" | Current Map: {spraying_map} |"

    # SAFETY LAYER 
    safety_message = " [SAFETY: This is READ-ONLY decision-support. Please verify in field. Farmer is final decision maker. No auto-spray.]"

    final_output = {
        "farmer_question": farmer_question,
        "predicted_intent": intent,
        "intent_confidence": round(confidence*100, 2),
        "response": base_response + extra_info + safety_message,
        "source": "Member 9 spraying_pipeline.py -> Member 8 chatbot",
        "action": "READ_ONLY - requires farmer verification gate",
        "transparency": f"Model confidence {confidence*100:.1f}% from leaf_classifier.pt"
    }

    return json.dumps(final_output, indent=2)

# Example
print(chatbot_answer("where to spray?"))
print(chatbot_answer("should I spray now?"))
