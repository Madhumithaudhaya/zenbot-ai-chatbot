# ZenBot - AI Mental Healthcare Chatbot

ZenBot is an AI-powered chatbot designed to offer mental health support using Retrieval-Augmented Generation (RAG) architecture.

It integrates:

- **DistilRoBERTa** for emotion and sentiment analysis  
- **FAISS** with Sentence Transformers for semantic search  
- **TinyLlama** as the response generator  
- **MongoDB** to store chat history  
- **Flask** as the backend API server  

> ⚠️ This is a research and academic project and is **not intended for real-world clinical use**.

---

## 🔧 Features

- Detects user sentiment using DistilRoBERTa
- Retrieves relevant past responses using FAISS
- Generates context-aware responses with LLM (TinyLlama)
- Stores all user-bot conversations in MongoDB
- Easy to deploy using Python & Flask

---

## 🛠️ Installation

```bash
git clone https://github.com/YOUR_USERNAME/zenbot-ai-chatbot.git
cd zenbot-ai-chatbot
pip install -r requirements.txt
