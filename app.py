from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import faiss
import json
import os
import numpy as np
import ijson
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient

# ====================== App Initialization ======================
app = Flask(__name__)
CORS(app)

base_path = r"C:\\Users\\madhu\\Documents\\zenbot"
sentiment_model_path = os.path.join(base_path, "distilroberta-emotion")
faiss_index_path = os.path.join(base_path, "faiss_index.bin")
faiss_texts_path = os.path.join(base_path, "faiss_texts.json")
sentence_model_path = os.path.join(base_path, "all-MiniLM-L6-v2")

# ====================== Sentiment Model ======================
print("🔄 Loading Sentiment Model...")
sentiment_model_name = "distilbert-base-uncased-finetuned-sst-2-english"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
id2label = {0: "negative", 1: "positive"}
print("✅ Sentiment Model Loaded")

def analyze_sentiment(text):
    print(f"🧠 Analyzing sentiment for: {text}")
    inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
        sentiment_id = torch.argmax(probs).item()
        sentiment = id2label.get(sentiment_id, "neutral")
        print(f"🧾 Sentiment Detected: {sentiment}")
        return sentiment

# ====================== FAISS Index & Embedding ======================
print("📦 Loading FAISS Index...")
faiss_index = faiss.read_index(faiss_index_path)
print("✅ FAISS Index Loaded")

print("📖 Loading FAISS Texts...")
faiss_texts = []
try:
    with open(faiss_texts_path, "r", encoding="utf-8") as f:
        for item in ijson.items(f, "item"):
            faiss_texts.append(item)
    print(f"✅ Loaded {len(faiss_texts)} texts")
except Exception as e:
    print(f"[FAISS Texts Error]: {e}")
    faiss_texts = []

print("🧠 Loading Sentence Embedding Model...")
embed_model = SentenceTransformer(sentence_model_path)
print("✅ Sentence Transformer Loaded")

def retrieve_relevant_knowledge(user_input, k=2):
    print(f"🔍 Retrieving knowledge for: {user_input}")
    try:
        query_embedding = np.array(embed_model.encode([user_input]), dtype=np.float32)
        distances, indices = faiss_index.search(query_embedding, k)
        retrieved_texts = [faiss_texts[i] for i in indices[0] if i < len(faiss_texts)]
        print(f"📚 Retrieved {len(retrieved_texts)} pieces of knowledge")
        return " ".join(retrieved_texts) if retrieved_texts else ""
    except Exception as e:
        print(f"[FAISS Retrieval Error]: {e}")
        return ""

# ====================== distilgpt2 Model ======================
print("🤖 Loading distilgpt2 Model...")
gpt2_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
gpt2_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
generator = pipeline("text-generation", model=gpt2_model, tokenizer=gpt2_tokenizer, device="cpu")
print("✅ distilgpt2 Model Loaded")

# ====================== MongoDB Setup ======================
print("🔌 Connecting to MongoDB...")
try:
    client = MongoClient("mongodb://localhost:27017/")
    db = client["zenbot_db"]
    collection = db["chat_history"]
    print("✅ MongoDB Connected")
except Exception as e:
    print(f"[MongoDB Connection Error]: {e}")

def store_chat_history(user, user_input, bot_response):
    try:
        collection.insert_one({"user": user, "query": user_input, "response": bot_response})
        print(f"💾 Stored chat for user: {user}")
    except Exception as e:
        print(f"[MongoDB Storage Error]: {e}")

def get_context_aware_history(user, last_n=2):
    try:
        history = collection.find({"user": user}).sort("_id", -1).limit(last_n)
        context = " ".join([doc["query"] + " " + doc["response"] for doc in history])
        print(f"📜 Fetched {last_n} previous chats")
        return context
    except Exception as e:
        print(f"[MongoDB Retrieval Error]: {e}")
        return ""

# ====================== Response Generation ======================
def generate_response(user, user_input):
    sentiment = analyze_sentiment(user_input)
    retrieved_knowledge = retrieve_relevant_knowledge(user_input)
    chat_history = get_context_aware_history(user, last_n=2)

    prompt = f"""The following is a conversation between a user and an AI mental wellness assistant named ZenBot.

The assistant is empathetic, supportive, and context-aware.

Previous Conversation:
{chat_history}

User's Message:
{user_input}

User's Sentiment: {sentiment}

Relevant Knowledge: {retrieved_knowledge}

ZenBot:"""

    print(f"🧠 Generating response for: {user_input}")
    response = generator(prompt, max_new_tokens=100, do_sample=True, temperature=0.6)[0]["generated_text"]
    response_text = response.split("ZenBot:")[-1].strip().split("User:")[0].strip()

    print(f"🤖 Generated Response: {response_text}")
    store_chat_history(user, user_input, response_text)
    return response_text

# ====================== Flask Routes ======================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat_ui")
def chat_ui():
    return render_template("chat.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("text", "").strip()
    user = data.get("user", "default_user")

    print(f"📨 Received input from {user}: {user_input}")

    if not user_input:
        return jsonify({"response": "⚠️ Empty message received."})

    try:
        response = generate_response(user, user_input)
        return jsonify({"response": response})
    except Exception as e:
        print(f"[Response Generation Error]: {e}")
        return jsonify({"response": "⚠️ An error occurred while generating response."})

# ====================== App Runner ======================
if __name__ == "__main__":
    print("🚀 Starting ZenBot Flask server...")
    app.run(host="127.0.0.1", port=5000, debug=True)
