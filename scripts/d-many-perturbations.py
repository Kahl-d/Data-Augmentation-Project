import pandas as pd
import random
import torch
from transformers import DebertaV2Tokenizer, DebertaV2ForMaskedLM

# Load data
data = pd.read_csv("/Users/923673423/lime/submission/data/total_data_class.csv")

# Function to structure a row with randomly masked columns, excluding the specified column
def convert_to_text_with_random_masks(row, mask_columns, exclude_column="Physical Activity Status"):
    """Convert a row into structured text with randomly masked columns, excluding one."""
    row_text = []
    for col in row.index:
        if col in mask_columns and col != exclude_column:  # Mask only eligible columns
            row_text.append(f"{col}: [MASK]")  # Mask selected columns
        else:
            row_text.append(f"{col}: {row[col]}")
    return " | ".join(row_text)  # Join columns into structured text

# Function to predict masked tokens using DeBERTa
def predict_with_deberta(prompt, model, tokenizer):
    """Predict masked tokens using DeBERTa."""
    inputs = tokenizer(prompt, return_tensors="pt", max_length=128, truncation=True)
    inputs = {key: val.to(model.device) for key, val in inputs.items()}  # Move inputs to the correct device
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits

    # Find masked token indices
    masked_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

    predicted_values = {}
    for index in masked_indices:
        predicted_token_id = predictions[0, index].argmax(dim=-1).item()
        predicted_token = tokenizer.decode([predicted_token_id])
        predicted_values[index.item()] = predicted_token  # Map index to predicted value

    return predicted_values

# Generate perturbations for 60% of the rows
def generate_perturbations(data, model, tokenizer, output_path, exclude_column="Physical Activity Status"):
    """Generate perturbations for 60% of the rows in the dataset."""
    perturbations = []
    sampled_data = data.sample(frac=0.6, random_state=42)  # Randomly select 60% of the rows
    num_perturbations = int(0.6 * len(data))  # Create 0.6N perturbations

    for _ in range(num_perturbations):
        # Select a random row from the sampled data
        row = sampled_data.sample(n=1).iloc[0]

        # Randomly select columns to mask, excluding the specified column
        maskable_columns = [col for col in data.columns if col != exclude_column]
        num_masks = random.randint(1, max(1, len(maskable_columns) // 4))  # Less than half the columns
        mask_columns = random.sample(maskable_columns, num_masks)

        # Prepare the prompt with random masks
        prompt = convert_to_text_with_random_masks(row, mask_columns, exclude_column=exclude_column)
        
        # Predict the masked tokens
        predicted_values = predict_with_deberta(prompt, model, tokenizer)
        
        # Reconstruct the row with the predicted values for masked columns
        reconstructed_row = row.to_dict()  # Convert the original row to a dictionary
        for col, value in zip(mask_columns, predicted_values.values()):
            reconstructed_row[col] = value  # Replace masked column with predicted value
        
        perturbations.append(reconstructed_row)

    # Save the perturbations as a new CSV file
    pd.DataFrame(perturbations).to_csv(output_path, index=False)
    print(f"Perturbations saved to {output_path}")

# Load model and tokenizer
model_path = "/Users/923673423/lime/submission/models/deberta-mlm-finetuned"
tokenizer = DebertaV2Tokenizer.from_pretrained(model_path)
model = DebertaV2ForMaskedLM.from_pretrained(model_path)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Generate perturbations and save to a CSV file
output_csv_path = "/Users/923673423/lime/submission/data/perturbed_data_total.csv"
generate_perturbations(data, model, tokenizer, output_csv_path)