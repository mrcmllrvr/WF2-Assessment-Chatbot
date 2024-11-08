# embedding.py

import json
import re
import numpy as np
import hashlib
import logging
import time
from threading import Lock
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import chromadb
from chromadb.utils import embedding_functions
import openai
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# # Ensure necessary NLTK packages are downloaded
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')

# Constants
CHROMA_DATA_PATH = 'chromadb_WF2_chatbot/'
COLLECTION_NAME = "document_embeddings"
JSON_FILE_PATH = 'data/WF2_KB.json'

# Setup ChromaDB client
client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=openai.api_key, model_name="text-embedding-ada-002", dimensions=1536)
collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=openai_ef, metadata={"hnsw:space": "cosine"})

# Define stop words
stop_words = set(stopwords.words('english'))

# Embedding function
def get_embedding(text):
    """Generates an embedding for the input text using OpenAI's API."""
    response = openai.Embedding.create(model="text-embedding-ada-002", input=text)
    return response["data"][0]["embedding"]

# Text preprocessing
def preprocess_text(text, use_lemmatization=True):
    """Preprocesses the input text by tokenizing, removing stop words, and optionally lemmatizing."""
    normalized_text = re.sub(r'\W+', ' ', text.lower())
    tokens = word_tokenize(normalized_text)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer() if use_lemmatization else None
    final_tokens = [lemmatizer.lemmatize(token) if lemmatizer else token for token in filtered_tokens]
    return " ".join(final_tokens)

# Generate unique ID for documents
def generate_unique_id(document):
    """Generates a unique ID for each document based on its content."""
    hash_object = hashlib.sha256(document.encode('utf-8'))
    return hash_object.hexdigest()

# Function to load JSON knowledge base
def load_json_data(file_path):
    """Loads JSON data from the specified file path."""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def create_index(file_path):
    # Delete the existing collection if it exists
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection: {COLLECTION_NAME}")
    except Exception:
        print("No existing collection found. Creating a new one.")

    # Re-create the collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=openai_ef
    )
    print("Created a new collection.")

    # Load JSON data and index it
    data = load_json_data(file_path)
    for entry in data["contents"]:
        title = entry.get("section", entry.get("sub_topic", ""))
        content = entry.get("text", "")
        
        if title and content:
            processed_text = preprocess_text(content)
            embedding = get_embedding(processed_text)
            unique_id = generate_unique_id(title + content)
            
            # Add document to ChromaDB
            try:
                collection.add(ids=[unique_id], embeddings=[embedding], metadatas=[{"title": title, "text": content}])
                logging.info(f"Document '{title}' indexed successfully.")
            except Exception as e:
                logging.error(f"Error indexing document '{title}': {str(e)}")

    print("Data indexing complete.")

# Set up logging
logging.basicConfig(level=logging.INFO)

# Run indexing
if __name__ == '__main__':
    create_index(JSON_FILE_PATH)
