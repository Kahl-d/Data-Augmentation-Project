import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch

# Define Function to Convert Rows into Structured Text
def convert_to_text(row, exclude_column="Physical Activity Status"):
    """Convert a row into structured text."""
    row_text = []
    for col in row.index:
        if col != exclude_column:  # Exclude the label column from the text
            row_text.append(f"{col}: {row[col]}")
    return " | ".join(row_text)  # Join columns into structured text

# Load Data
data = pd.read_csv("/Users/923673423/lime/submission/data/universal_train_data.csv")  # Ensure your CSV contains the required columns

# Convert Rows into Text
data["input_text"] = data.apply(
    lambda row: convert_to_text(row),
    axis=1,
)

# Set Input and Label Columns
X = data["input_text"].tolist()
y = data["Physical Activity Status"].astype("category").cat.codes.tolist()  # Convert labels to numeric

# Split Data into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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


# Load Tokenizer and Model
model_name = "bert-base-uncased"  # You can replace with another pre-trained model
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(set(y)))

# Create Dataset Objects
train_dataset = ClassificationDataset(X_train, y_train, tokenizer)
test_dataset = ClassificationDataset(X_test, y_test, tokenizer)

# Define Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    logging_dir="./logs",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    save_steps=500,
    save_total_limit=2,
    learning_rate=2e-5,
    weight_decay=0.01,
)

# Trainer Instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Fine-Tune the Model
trainer.train()

# Save Fine-Tuned Model
trainer.save_model("/Users/923673423/lime/submission/models/deberta-classification-finetuned")
tokenizer.save_pretrained("/Users/923673423/lime/submission/models/deberta-classification-finetuned")