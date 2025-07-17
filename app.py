from flask import Flask, render_template, request, jsonify, session
import json
import random
import pickle
import numpy as np
from datetime import timedelta

# Load data and models
with open("intents.json", "r") as file:
    intents = json.load(file)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Extract tags and responses
tags = [intent['tag'] for intent in intents['intents']]
responses = {intent['tag']: intent['responses'] for intent in intents['intents']}

# Keyword shortcuts (manual override)
keyword_to_tag = {
    "solar": "solar_info",
    "solar panel": "solar_info",
    "calculator": "solar_calculator",
    "ujala": "ujala_scheme",
    "scheme": "government_schemes",
    "schemes": "government_schemes",
    "subsidy": "government_schemes",
    "appliances": "appliances",
    "fridge": "appliances",
    "ac": "appliances",
    "energy saving": "energy_saving",
    "saving energy": "energy_saving",
    "save electricity": "energy_saving",
    "wind": "wind_energy",
    "geothermal": "geothermal_energy",
    "bio": "bio_energy",
    "hydro": "hydro_energy",
    "environment": "environment",
    "types": "types_of_energy",
    "hi": "greeting",
    "hello": "greeting",
    "good morning": "greeting",
    "good evening": "greeting",
    "bye": "goodbye",
    "exit": "goodbye",
    "quit": "goodbye"
}

# Flask app
app = Flask(__name__)
app.secret_key = "11e5ee78ab58128899d2b8861df33498"
app.permanent_session_lifetime = timedelta(minutes=10)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    session.permanent = True
    user_input = request.json.get("message", "").lower().strip()

    # Solar calculator step
    if session.get("awaiting_bill"):
        try:
            bill = int(user_input)
            session.pop("awaiting_bill")
            if bill >= 3000:
                return jsonify({"response": "You can save a lot by installing a rooftop solar panel! A 3–5 kW system might suit your usage. Government subsidies may also apply."})
            else:
                return jsonify({"response": "Your bill is relatively low, but you can still benefit from solar water heaters or smaller systems!"})
        except ValueError:
            return jsonify({"response": "Please enter a valid number for your electricity bill in ₹."})

    if "solar calculator" in user_input or "calculate" in user_input:
        session["awaiting_bill"] = True
        return jsonify({"response": "Sure! Please tell me your average monthly electricity bill in ₹."})

    # First check keyword-to-tag mapping
    for keyword, tag in keyword_to_tag.items():
        if keyword in user_input:
            return jsonify({"response": random.choice(responses[tag])})

    # Else use model prediction with confidence threshold
    X = vectorizer.transform([user_input])
    proba = model.predict_proba(X)[0]
    max_proba = np.max(proba)
    predicted_tag = model.predict(X)[0]

    if max_proba >= 0.55 and predicted_tag in responses:
        reply = random.choice(responses[predicted_tag])
    else:
        reply = "Sorry, I didn’t quite get that. Try asking about solar, wind, government schemes, or saving energy."

    return jsonify({"response": reply})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

