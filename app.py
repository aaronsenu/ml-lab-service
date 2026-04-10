from flask import Flask, request, jsonify
from transformers import pipeline
from dotenv import load_dotenv
import os

# Load secrets from .env file
load_dotenv()

app = Flask(__name__)

# Secret loaded from environment (not hardcoded)
API_KEY = os.getenv("API_KEY", "")

# Sentiment model from Lab 1
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)


def require_api_key(f):
    """Simple API key guard using injected secret."""
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get("X-API-Key", "")
        if API_KEY and key != API_KEY:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
@require_api_key
def predict():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400
    prediction = classifier(data["text"])
    return jsonify({
        "input": data["text"],
        "label": prediction[0]["label"],
        "score": prediction[0]["score"]
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
