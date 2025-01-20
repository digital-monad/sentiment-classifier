import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, TrainingArguments, Trainer
import nltk

# nltk.download('punkt_tab')

data = pd.read_json("./data/data.jsonl", lines=True)

data["text"] = data["text"].apply(nltk.sent_tokenize)

data = data.explode("text")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

train_dataset = Dataset.from_pandas(train_data)
test_dataset = Dataset.from_pandas(test_data)

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased", num_labels=3)

tokenized_train = train_dataset.map(tokenize_function, batched=True)

tokenized_test = test_dataset.map(tokenize_function, batched=True)

def encode_labels(examples):
    examples['label'] = [2 if rating > 3 else 1 if rating == 3 else 0 for rating in examples['rating']]
    return examples

tokenized_train = tokenized_train.map(encode_labels, batched=True)
tokenized_test = tokenized_test.map(encode_labels, batched=True)

tokenized_train = tokenized_train.remove_columns(["rating", "title", "images", "asin", "parent_asin", "user_id", "timestamp", "helpful_vote", "verified_purchase"])
tokenized_test = tokenized_test.remove_columns(["rating", "title", "images", "asin", "parent_asin", "user_id", "timestamp", "helpful_vote", "verified_purchase"])

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)


training_args = TrainingArguments(
    output_dir = "./results",
    eval_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test
)
trainer.train()


model.save_pretrained('./sentiment_model')
tokenizer.save_pretrained('./sentiment_model')
