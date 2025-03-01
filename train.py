# train.py
import os
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# ----------------------------
# 1. Create a Dummy Dataset
# ----------------------------
# In a real scenario, you would load your labeled CTI data from a CSV or JSON file.
# Here we simulate a dataset where each text is associated with a label.
# (Assume labels: 0 = "Initial Access/Execution", 1 = "Lateral Movement/Data Exfiltration", 2 = "Command and Control")
data = {
    "text": [
        "The report describes an attack that used spearphishing for initial access.",
        "This threat report mentions lateral movement and data exfiltration techniques.",
        "Attackers used a command and control server to communicate with compromised systems."
    ],
    "labels": [0, 1, 2]  # Dummy labels for a 3-class classification problem
}

# Create a Hugging Face Dataset object from the dictionary
dataset = Dataset.from_dict(data)

# ----------------------------
# 2. Tokenization
# ----------------------------
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

# Apply tokenization to the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# ----------------------------
# 3. Load the Pre-Trained Model
# ----------------------------
# We use a simple sequence classification model.
num_labels = 3
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# ----------------------------
# 4. Set Up Training Arguments
# ----------------------------
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=5,
)

# ----------------------------
# 5. Create the Trainer
# ----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,  # For demonstration, we use the same data for evaluation.
)

# ----------------------------
# 6. Train the Model
# ----------------------------
trainer.train()

# ----------------------------
# 7. Save the Fine-Tuned Model and Tokenizer
# ----------------------------
save_dir = "./fine_tuned_model"
os.makedirs(save_dir, exist_ok=True)
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print("Model training complete and saved to", save_dir)
