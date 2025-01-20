import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

model_path = "./sentiment_model"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

sentiments = ["negative", "neutral", "positive"]

def prepare_input(text: str):
    return tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")

def score(text: str):
    input = prepare_input(text)
    with torch.no_grad():
        output = model(**input)
        logits = output.logits
        idx = torch.argmax(logits, dim=1).item()
        return sentiments[idx], torch.max(logits, dim=1).values.item()
