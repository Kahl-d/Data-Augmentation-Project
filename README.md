
# Context-Aware Perturbations for Data Augmentation

## Technical Problem to Solve
Machine learning models often face challenges with limited data or poorly distributed datasets, leading to reduced accuracy and generalization. Traditional random perturbations used for data augmentation often fail to maintain the context and structure of the dataset, resulting in unrealistic variations that degrade model performance. 

This project solves these issues by:
- Generating **context-aware perturbations** using fine-tuned language models (e.g., DeBERTa).
- Creating realistic and domain-aligned variations for data augmentation.
- Improving model fidelity and interpretability by aligning perturbations with the dataset's inherent structure.

---

## How to Run

### 1. Prepare Your Data
- Place your dataset in the `data/` directory.
- Split it into `train.csv` and `test.csv`.

### 2. Navigate to the `scripts/` Folder
The `scripts/` folder contains the essential scripts for running the project:

#### a. Fine-Tune the Model
Run `p-perturb-model-train.py` to fine-tune a language model (LM) on your training data:
```bash
python scripts/p-perturb-model-train.py --data ./data/train.csv --output ./models/fine_tuned_model/
```

![image](https://github.com/user-attachments/assets/f245772a-5139-4d21-97d6-8dd3074cb52e)

- **Customizable**: Modify the script to use a different LM (e.g., BERT, RoBERTa).

#### b. Generate Perturbations
- **Single Perturbation**: Use `d-perturb-test.py` to generate a single perturbation by specifying a base row or freezing certain columns:
```bash
python scripts/d-perturb-test.py --model ./models/fine_tuned_model/ --data ./data/test.csv --output ./results/single_perturbation.csv --freeze_columns "Gender,Age"
```

- **Multiple Perturbations**: Use `d-many-perturbations.py` to create a dataset with multiple perturbations:
```bash
python scripts/d-many-perturbations.py --model ./models/fine_tuned_model/ --data ./data/test.csv --output ./results/perturbed_data.csv --num_perturbations 100
```

### 3. Explore Results
- The `notebooks/` folder contains validation and proofs:
  - Random Forest (RF) model performance testing.
  - Validation using NHANES data.

---

## How to Verify the Output

1. **Verify Perturbations**:
   - Check the generated perturbations in the output files (e.g., `perturbed_data.csv`).
   - Ensure the perturbations are realistic and maintain the context of the data.

2. **Evaluate Model Performance**:
   - Test the model performance using the original dataset and the augmented dataset.
   - Expected improvement: **3-3.5% increase in accuracy**.

3. **Stability Testing**:
   - Run the perturbation scripts with different random seeds to verify consistency of outputs.

---

## Test Cases

### 1. Single Perturbation
- **Input**: A single row from `test.csv`.
- **Output**: A perturbed row with masked and replaced values.
- **Verification**: Check if the replacements are realistic and align with the dataset context.

### 2. Multiple Perturbations
- **Input**: The entire `test.csv` dataset.
- **Output**: A dataset with `N` perturbations saved in `perturbed_data.csv`.
- **Verification**: Ensure the output has `N` realistic perturbations that preserve feature relationships.

### 3. Random Forest Validation
- **Input**: Train and test the Random Forest model on both original and augmented datasets.
- **Output**: Accuracy improvement of 3-3.5% with the augmented dataset.
- **Verification**: Compare performance metrics before and after augmentation.

---

## Directory Structure

```plaintext
📂 Project Root
├── 📂 data/                  # Dataset directory
│   ├── train.csv             # Training data
│   ├── test.csv              # Test data
├── 📂 models/               # Saved fine-tuned models
│   ├── fine_tuned_model/    # Fine-tuned language model
├── 📂 scripts/              # Core scripts for the project
│   ├── p-perturb-model.py        # Fine-tune the language model
│   ├── d-perturb-test.py         # Generate a single perturbation
│   ├── d-many-perturbations.py   # Generate multiple perturbations
├── 📂 results/              # Results directory
│   ├── single_perturbation.csv   # Single perturbation output
│   ├── perturbed_data.csv        # Multiple perturbations dataset
├── 📂 notebooks/            # Validation and proof of performance
│   ├── RF_model_test.ipynb       # Random Forest model testing
│   ├── NHANES_validation.ipynb   # Validation using NHANES data
├── README.md               # Project documentation
```

---

## Future Enhancements
- **Causal Relationships:**: Incorporate causal relationships into the perturbation generation process to enhance dataset quality and interpretability.  
- **Knowledge Graphs**: Leverage knowledge graphs to ensure contextual integrity and generate more meaningful perturbations based on domain-specific relationships.  
- **Beyond Tabular Data**: Extend the approach to handle non-tabular datasets such as images, audio, and time-series data, enabling a broader range of applications.
- **Domain Adaptation**: Focus on robust domain adaptation techniques to fine-tune models for specific use cases, improving data augmentation effectiveness.
- **Scaling**: Adapt the approach for larger models like GPT and LLaMA to generate highly realistic perturbations.
- **Tool Integration**: Incorporate perturbations into tools like LIME and SHAP for enhanced interpretability.


---

## Contact
For any questions or contributions:
- **Maintainer**: Khalid Mehtab Khan  
- **Email**: email.khalidmkhan@gmail.com  
- **GitHub**: [Your GitHub Profile](https://github.com/Kahl-d)
