# ATT&CK Tactics and Techniques Extraction Project

This project extracts MITRE ATT&CK Tactics and Techniques from Cyber Threat Intelligence (CTI) reports using a large language model. It provides a simple web interface built with Flask.

## Project Structure

attack_extraction_project/ ├── README.md ├── requirements.txt ├── app.py ├── model.py ├── preprocess.py ├── data/ │ └── sample_report.txt └── templates/ └── index.html


## Setup Instructions

1. Create and activate a virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Run the Flask app using `python app.py`.
4. Open your browser and go to `http://127.0.0.1:5000/` to test the interface.

*Note:* The model inference function currently returns dummy output. Replace it with your fine-tuned model code as needed.
