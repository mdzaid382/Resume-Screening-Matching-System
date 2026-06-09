import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('stopwords')



def clean_text(text: str) -> str:

    # Initialize lemmatizer and stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    # Remove email addresses
    text = re.sub(r'\S+@\S+', ' ', text)

    # Remove URLs (linkedin, github, websites)
    text = re.sub(r'http\S+|www\.\S+|linkedin\.com/\S+|github\.com/\S+', ' ', text)

    # Remove phone numbers
    text = re.sub(
        r'(\+\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})',
        ' ',
        text
    )

    # Remove markdown links
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

    # Remove bullet symbols
    text = re.sub(r'[•▪◦●■♦★]', ' ', text)

    # Remove asterisks used as bullets
    text = re.sub(r'\*', ' ', text)

    # Keep only letters, numbers, +, #, /, . and spaces
    text = re.sub(r'[^a-zA-Z0-9+#/. ]', ' ', text)

    # Convert to lowercase
    text = text.lower()

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    #remove stopwords
    text = " ".join([word for word in text.split() if word not in stop_words])

    #lematize the text
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
   

    return text


