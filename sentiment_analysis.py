
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Load tokenizer and model
model_name = "distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained("distilroberta-base-finetuned-sst-2-english")

# Labels for classification
labels = ['NEGATIVE', 'POSITIVE']

def analyze_sentiment(text):
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Convert logits to probabilities
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=1)
    confidence, predicted_class = torch.max(probs, dim=1)
    
    result = {
        "label": labels[predicted_class.item()],
        "confidence": float(confidence.item())
    }
    return result

# Example usage
if __name__ == "__main__":
    test_input = "I feel really low and hopeless."
    result = analyze_sentiment(test_input)
    print(f"Sentiment: {result['label']} (Confidence: {result['confidence']:.2f})")
