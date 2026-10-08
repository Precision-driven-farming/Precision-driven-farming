"""
Member 8 - Farmer Chatbot 
Integrates Member 9 spraying_pipeline.py
"""
import os
import sys
import json

# Load Member 9 map
def load_member9_map():
    map_path = os.path.join(os.path.dirname(__file__), "..", "spraying_pipeline", "spraying_map.json")
    map_path = os.path.abspath(map_path)
    if not os.path.exists(map_path):
        return {"Zone2": "78% rust - verify first [PLACEHOLDER until Member 9 real]"}
    with open(map_path) as f:
        return json.load(f)

spraying_map = load_member9_map()

class FarmerChatbot:

    def __init__(self):
        self.nlp = FarmerNLP()
        self.map = spraying_map  # Integration with Member 9

    def safety_check(self, intent, cv_confidence=None):
        """
        Member 8 Safety Gate - Prevents automatic spraying decision
        Farmer must verify - READ-ONLY system
        """
        if intent == "verify_spray_decision":
            return True
        if intent == "show_spray_map":
            return True  # Even showing map needs verification before action
        if cv_confidence is not None and cv_confidence < 0.70:
            return True
        return True  # Default: Always require verification for safety

    def respond(self, message, cv_confidence=None):

        result = self.nlp.classify(message)
        intent = result["intent"]
        nlp_confidence = result["confidence"]

        requires_verification = self.safety_check(intent, cv_confidence)

        if intent == "crop_health_status":
            response = (
                f"I can show the latest crop health results from Member 9 map: {self.map}. "
                f"Detected diseases, affected areas and confidence scores. Current map shows Zone2 78% rust."
            )
        elif intent == "disease_information":
            response = (
                "The system can provide the disease detected "
                "by the computer vision model (Member 6 leaf_classifier.pt) together with "
                f"its confidence score. Map: {self.map}"
            )
        elif intent == "show_confidence":
            response = (
                "The AI confidence score refers to how confident "
                "the computer vision model is in its prediction. I always show it for transparency."
            )
        elif intent == "show_spray_map":
            response = (
                f"I can display the precision spraying map from Member 9: {self.map} "
                f"showing the areas identified for possible treatment. PLEASE VERIFY IN FIELD."
            )
        elif intent == "historical_results":
            response = "I can display previous disease detection results for comparison."
        elif intent == "health_trend":
            response = "I can compare disease detection results over time to show increase/decrease."
        elif intent == "disease_location":
            response = f"I can show GPS locations from Member 9 map: {self.map} where disease was detected."
        elif intent == "explain_alert":
            response = "This area was flagged because CV model detected features of possible crop disease. Confidence shown in map."
        elif intent == "verify_spray_decision":
            response = (
                "The AI recommendation must be reviewed and "
                "verified by the farmer before spraying. "
                "The chatbot does not automatically authorise spraying. READ-ONLY."
            )
        else:
            response = (
                "I could not understand. "
                "Please ask about crop health, disease detection, "
                "confidence, spraying maps or historical results."
            )

        # FINAL SAFETY FOOTER 
        safety_footer = " [SAFETY: READ-ONLY, requires farmer verification, no auto-spray. Source: Member 9 -> Member 8]"

        return {
            "member": "Member 8 - Luyanda",
            "message": message,
            "intent": intent,
            "nlp_confidence": nlp_confidence,
            "response": response + safety_footer,
            "requires_verification": requires_verification,
            "spraying_map_source": "Member 9 spraying_pipeline.py",
            "action": "READ_ONLY"
        }

if __name__ == "__main__":
    chatbot = FarmerChatbot()
    print("Precision-Driven-Farming Farmer Chatbot - Member 8")
    print("Type 'exit' to stop.")
    while True:
        user_message = input("Farmer: ")
        if user_message.lower() == "exit":
            break
        result = chatbot.respond(user_message, cv_confidence=0.78)
        print("Chatbot:", result["response"])
        print("Intent:", result["intent"])
        print("NLP confidence:", result["nlp_confidence"])
        print("Requires verification:", result["requires_verification"])
