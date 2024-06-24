from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"

tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)


def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=-1)

    pred = torch.argmax(probs, dim=1).item()
    prob = torch.max(probs).item()
    return pred, prob


texts = [
    "I love this product! It's absolutely amazing.",
    "This is the worst experience I've ever had.",
    "The movie was okay, not great but not terrible either."
]

for text in texts:
    pred, prob = predict_sentiment(text)

    sentiment = "positive" if pred > 2 else "negative" if pred < 2 else "neutral"

    print(f"Text: {text}\nSentiment: {sentiment} (confidence: {prob:.2f})\n")

