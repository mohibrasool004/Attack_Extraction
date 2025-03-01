from flask import Flask, request, render_template
from model import model_inference
from preprocess import clean_text
import csv
import os

app = Flask(__name__)

# Function to save extraction results to a text file
def save_result_to_log(report_text, extracted_result):
    with open("extraction_results.txt", "a", encoding="utf-8") as file:
        file.write("CTI Report:\n")
        file.write(report_text + "\n\n")
        file.write("Extraction Result:\n")
        file.write(extracted_result + "\n")
        file.write("=" * 50 + "\n\n")  # Separator for readability

# Function to save extraction results to a CSV file
def save_result_to_csv(report_text, extracted_result):
    file_path = "extraction_results.csv"
    
    # Check if file exists, if not, create with header
    file_exists = os.path.exists(file_path)
    
    with open(file_path, "a", newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        
        # Write header if file is new
        if not file_exists:
            writer.writerow(["CTI Report", "Extraction Result"])
        
        writer.writerow([report_text, extracted_result])

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        cti_report = request.form["cti_report"]
        # Preprocess the input text
        cleaned_text = clean_text(cti_report)
        # Pass the cleaned text to the model inference function
        result = model_inference(cleaned_text)
        
        # Save results to log and CSV
        save_result_to_log(cti_report, result)
        save_result_to_csv(cti_report, result)

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
