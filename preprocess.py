# preprocess.py
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Download NLTK data (only needed the first time)
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')

def clean_text(text):
    """
    Cleans the input text by:
    - Converting to lowercase
    - Removing HTML tags and special characters
    - Removing stop words
    - Tokenizing the text
    Returns a cleaned string.
    """
    # Remove HTML tags if any
    text = re.sub(r'<[^>]+>', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stop words
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    # Join tokens back into string
    cleaned_text = " ".join(tokens)
    return cleaned_text
