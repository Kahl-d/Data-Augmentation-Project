import pandas as pd
import random
import torch
from transformers import DebertaV2Tokenizer, DebertaV2ForMaskedLM, DataCollatorForLanguageModeling
from datasets import Dataset


# Load model and tokenizer
model_path = "/Users/923673423/lime/submission/models/deberta-mlm-finetuned"
tokenizer = DebertaV2Tokenizer.from_pretrained(model_path)
model = DebertaV2ForMaskedLM.from_pretrained(model_path)


# Load data
data = pd.read_csv("/Users/923673423/lime/submission/data/universal_train_data.csv")


# Convert a row into structured text format with randomly masked columns
def convert_to_text_with_random_masks(row, mask_columns):
    row_text = []
    for col in row.index:
        if col in mask_columns:
            row_text.append(f"{col}: [MASK]")  # Use BERT-compatible [MASK] token
        else:
            row_text.append(f"{col}: {row[col]}")
    return " | ".join(row_text)  # Use delimiter for clarity
    
# Function to predict masked tokens with DeBERTa
def predict_with_deberta(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt", max_length=128, truncation=True)
    inputs = {key: val.to(model.device) for key, val in inputs.items()}  # Move inputs to the correct device
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits
    masked_index = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

    for index in masked_index:
        predicted_token_id = predictions[0, index].argmax(dim=-1).item()
        predicted_token = tokenizer.decode([predicted_token_id])
        prompt = prompt.replace("[MASK]", predicted_token, 1)

    return prompt

# Test the model with multiple perturbations from a single data point
def test_with_multiple_perturbations(row, num_perturbations=5):
    print(f"Original Row: {row.to_dict()}")
    predictions = []
    for _ in range(num_perturbations):
        num_masks = random.randint(1, max(1, len(row.index) // 2))  # Mask fewer columns
        mask_columns = random.sample(list(row.index), num_masks)  # Randomly select columns to mask
        prompt = convert_to_text_with_random_masks(row, mask_columns)
        prediction = predict_with_deberta(prompt, model, tokenizer)
        print(f"Prediction: {prediction}")
        predictions.append((prompt, prediction))
    return predictions

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Select a random row from the dataset for testing
test_row = data.iloc[random.randint(0, len(data) - 1)]
test_with_multiple_perturbations(test_row)