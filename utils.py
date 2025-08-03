"""
Utility functions for AI Speech Therapist app.
"""

import os
from dotenv import load_dotenv

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import requests
import time

#retrieve API key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

#roberta model trained on sentiment from tweets 
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, use_safetensors=True)
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

#for every chat tracked in history, a sentiment is given for each of the texts
#Label_0 = "negative", label_1 = "neutral", label_2 = "positive"
def get_sentiments_for_history(history):
    sentiments = []
    for msg in history:
        result = classifier(msg)[0]
        label = result['label']
        sentiments.append(label)
    return sentiments


def build_gemini_prompt(history, sentiments):
    #builds the prompt and add conversation history
    LABEL_MAP = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive"
    }

    prompt = ("""You are a compassionate, nonjudgmental AI therapist designed to support users through thoughtful, empathetic conversation.

You will be given the user's most recent conversation history, with each message labeled by its sentiment.

Your Core Responsibilities:
- Acknowledge the user's emotions sincerely and validate their experience
- Use a warm, gentle tone to encourage deeper reflection or sharing
- After getting the user to share provide thoughtful support and advice, but not in a forceful or prescriptive way
- Occasionally use light, appropriate humor to put the user at ease — but only when the tone of the conversation safely allows it
-Vary the language used to avoid repetitive language
CRITICAL SAFETY PROTOCOLS:

If the user expresses any of the following, respond immediately with appropriate crisis intervention:

1. Thoughts of self-harm, suicide, or violence toward others
2. Severe emotional crisis
3. Delusional or highly disoriented thinking

RESPONSE PROTOCOL FOR CRISIS SITUATIONS:
- Respond calmly and empathetically, acknowledging their distress
- Encourage them to seek immediate help from a licensed professional, trusted person, or crisis line
- Do NOT attempt to diagnose, solve, or engage in detailed reasoning about harmful thoughts
- You may say: "It sounds like you're going through something incredibly difficult right now. You're not alone — please consider reaching out to a licensed therapist or a crisis line for real-time support. There are people who care about you and want to help."

For bizarre, confusing, or potentially delusional statements:
- Remain grounded, calm, and gently refocus the conversation
- Acknowledge them without reinforcing false beliefs

IMPORTANT LIMITATIONS:
- Never pretend to be a human
- Never give medical or psychiatric diagnoses
- Always prioritize emotional safety and compassion
- This is for educational/demonstrative purposes only, not professional mental health care
""")

    #track the message and its sentiment at the same time
    #append the conversation history and sentiment to the prompt
    for msg, sent in zip(history, sentiments):
        sentiment_text = LABEL_MAP.get(sent,sent) 
        prompt += f"User: {msg}\n(Sentiment: {sentiment_text})\n"
    prompt += "Therapist:"
    return prompt

def query_gemini(prompt):
    #define gemini endpoint
    url = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": 150}
    }
    params = {"key": GEMINI_API_KEY}

    max_retries = 3 
    # we allow 3 retries to endpoint because gemini api may not always be available
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, params=params, json=data, timeout=10)
            if response.status_code == 200: # success, and attempt to return the response
                result = response.json() 
                try:
                    return result["candidates"][0]["content"]["parts"][0]["text"], None
                except Exception as e: #parsing failed 
                    print("Gemini API response parsing error:", e)
                    return None, "Sorry, I couldn't generate a response."
            elif response.status_code == 503: # if api is overloaded we wait a few seconds an retry
                print(f"Gemini API 503 error (attempt {attempt+1}/{max_retries}). Retrying after pause...")
                print("Response text:", response.text)
                time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                continue
            else: # catches other errors
                print(f"Gemini API error: {response.status_code}")
                print("Response text:", response.text)
                return None, f"Gemini API error: {response.status_code}"
        except Exception as e: # fail to connect to gemini 
            print("Exception during Gemini API call:", e)
            return None, "Exception during Gemini API call."
    # If we reach here, all retries failed
    return None, "Gemini API error: 503 (model overloaded after retries)"
#if gemini api fails we have some very basic logic for responses
def fallback_response(history):
    # Use the last message for fallback sentiment
    text = history[-1] if history else ""
    sentiment = classifier(text)[0] # outputs: label -> sentiment and score-> confidence
    confidence = sentiment['score']
    label = sentiment['label']
    if confidence < 0.5:
        return "I'm not sure exactly how you're feeling, could you elaborate?"
    elif label == 'LABEL_0':  # negative sentiment
        if confidence > 0.8:
            return "I'm really sorry to hear you are struggling. Would you like to talk about it?"
        else:
            return "It sounds like something is bothering you? Want to talk about it?"
    elif label == 'LABEL_1':  # neutral sentiment
        return "Thank you for sharing. I'm here to listen."
    else:  # positive sentiment
        if confidence > 0.8:
            return "That's great to hear!"
        else:
            return "It seems you are feeling pretty good. Tell me more!"