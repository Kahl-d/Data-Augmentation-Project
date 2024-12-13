import pandas as pd
import random
import torch
from transformers import DebertaV2Tokenizer, DebertaV2ForMaskedLM, DataCollatorForLanguageModeling
from datasets import Dataset

# Load data
data = pd.read_csv("/Users/923673423/lime/submission/data/total_data_class.csv")

# Convert a row into structured text format with randomly masked columns
def convert_to_text_with_random_masks(row, mask_columns):
    row_text = []
    for col in row.index:
        if col in mask_columns:
            row_text.append(f"{col}: [MASK]")  # Use BERT-compatible [MASK] token
        else:
            row_text.append(f"{col}: {row[col]}")
    return " | ".join(row_text)  # Use delimiter for clarity

# Prepare training data: Randomly mask a subset of columns for each row
training_data = []
for idx, row in data.iterrows():
    num_masks = random.randint(1, max(1, len(data.columns) // 2))  # Mask fewer columns for better signal
    mask_columns = random.sample(list(data.columns), num_masks)  # Randomly select columns to mask
    prompt = convert_to_text_with_random_masks(row, mask_columns)
    training_data.append({"text": prompt})

# Example of prepared data
print("Sample Training Data:")
for item in training_data[:3]:
    print(item["text"])

# Create a Hugging Face Dataset
hf_dataset = Dataset.from_pandas(pd.DataFrame(training_data))

# Load tokenizer and model for DeBERTa
tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")
model = DebertaV2ForMaskedLM.from_pretrained("microsoft/deberta-v3-base")

# Tokenize the dataset
def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

tokenized_dataset = hf_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Set up data collator for MLM
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,  # Enable MLM for DeBERTa
    mlm_probability=0.3  # Mask 15% of tokens randomly
)

# Training arguments with better hyperparameters
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./deberta-mlm-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=5,  # Increased epochs for better learning
    per_device_train_batch_size=16,  # Larger batch size
    save_steps=500,
    save_total_limit=2,
    learning_rate=2e-5,  # Default learning rate
    warmup_steps=100,  # Add learning rate warm-up
    weight_decay=0.01,
    logging_dir="/Users/923673423/lime/submission/logs",
    logging_steps=100
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("/Users/923673423/lime/submission/models/deberta-mlm-finetuned")
tokenizer.save_pretrained("/Users/923673423/lime/submission/models/deberta-mlm-finetuned")

