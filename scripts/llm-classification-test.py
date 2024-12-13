import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from sklearn.metrics import classification_report

# Define Function to Convert Rows into Structured Text
def convert_to_text(row, exclude_column="Physical Activity Status"):
    """Convert a row into structured text."""
    row_text = []
    for col in row.index:
        if col != exclude_column:  # Exclude the label column from the text
            row_text.append(f"{col}: {row[col]}")
    return " | ".join(row_text)

# Load Test Data
test_data = pd.read_csv("/Users/923673423/lime/submission/data/universal_test_data.csv")

# Convert Rows into Text
test_data["input_text"] = test_data.apply(
    lambda row: convert_to_text(row),
    axis=1,
)

# Set Input and Label Columns
X_test = test_data["input_text"].tolist()
y_test = test_data["Physical Activity Status"].astype("category").cat.codes.tolist()

# Custom Dataset Class
class ClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# Load Tokenizer and Fine-Tuned Model
model_path = "/Users/923673423/lime/submission/models/deberta-classification-finetuned"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Create Dataset and DataLoader
test_dataset = ClassificationDataset(X_test, y_test, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=16)

# Evaluate the Model
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Classification Report
print(classification_report(all_labels, all_preds))