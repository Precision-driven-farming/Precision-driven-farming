from chatbot import FarmerChatbot

def test_safety():
    bot = FarmerChatbot()
    r = bot.respond("should I spray now?")
    assert r["requires_verification"] == True

def test_map_loaded():
    bot = FarmerChatbot()
    assert bot.map is not None

def test_intent():
    bot = FarmerChatbot()
    r = bot.respond("where to spray?")
    assert r["intent"] == "show_spray_map"

if __name__ == "__main__":
    test_safety()
    test_map_loaded()
    test_intent()
    print("All tests passed")
