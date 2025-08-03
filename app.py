"""
AI Speech to Speech Therapist

This application provides therapeutic responses to user responses
through speech or text. It uses sentiment analysis and a chat history routed to a llm 
to generate empathetic responses
"""

import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from utils import get_sentiments_for_history, build_gemini_prompt, query_gemini, fallback_response

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = Flask(__name__)

#render home page
@app.route("/")
def home():
    return render_template('index.html')

#handle user responses
@app.route("/speech-to-speech", methods=["POST"])
def speech_to_speech():
    
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    history = data.get("history") #chat history 
    if not history or not isinstance(history, list) or not all(isinstance(x, str) for x in history):
        return jsonify({"error": "No valid history provided"}), 400
    
    sentiments = get_sentiments_for_history(history)
    prompt = build_gemini_prompt(history, sentiments)
    response_text, error = query_gemini(prompt)
    if response_text:
        if response_text.startswith("Therapist:"):
            response_text = response_text[10:]  # Remove exactly "Therapist:"
        return jsonify({"response": response_text})
    else:
        # Fallback to sentiment-based response
        fallback = fallback_response(history)
        return jsonify({"response": fallback, "error": error})

if __name__ == "__main__":
    app.run(debug=True)