# test_inference.py
from preprocess import clean_text
from model import model_inference

# Sample CTI report text
sample_text = "The report describes an attack that used spearphishing for initial access."

# Clean the text using our preprocessing function
cleaned_text = clean_text(sample_text)
print("Cleaned Text:", cleaned_text)

# Get the model prediction
prediction = model_inference(cleaned_text)
print("Model Prediction:", prediction)
