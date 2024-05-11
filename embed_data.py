import os
import argparse
import logging
import sys
import requests
import fitz  # PyMuPDF

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

logging.basicConfig(level=logging.INFO)

def get_or_create_collection(collection_name: str, persist_directory: str, api_key: str):
    """
    Instantiates the Chroma client, and creates a collection, using OpenAI embeddings.
    """
    try:
        client = chromadb.Client(settings=Settings(
            chroma_db_impl="duckdb+parquet", persist_directory=persist_directory))
        embedding_function = OpenAIEmbeddingFunction(api_key=api_key)
        collection = client.get_or_create_collection(
            name=collection_name, embedding_function=embedding_function)
        return collection
    except Exception as e:
        logging.error(f"Failed to create or retrieve the collection: {e}")
        sys.exit(1)

def process_pdf_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        document = fitz.open("pdf", response.content)
        text = ""
        for page in document:
            text += page.get_text()
        document.close()
        return text
    except requests.HTTPError as e:
        logging.error(f"HTTP Error: {e}")
    except Exception as e:
        logging.error(f"Error processing PDF from URL: {e}")
    return None

def process_local_pdf(file_path):
    try:
        document = fitz.open(file_path)
        text = ""
        for page in document:
            text += page.get_text()
        document.close()
        return text
    except Exception as e:
        logging.error(f"Error reading local PDF file {file_path}: {e}")
    return None

def main(data_dir: str, collection_name: str, persist_directory: str, api_key: str):
    if not os.path.isdir(data_dir) and not data_dir.startswith("http"):
        logging.error(f"The specified data directory does not exist or is not a valid URL: {data_dir}")
        sys.exit(1)

    documents = []
    metadatas = []
    if data_dir.startswith("http"):
        text = process_pdf_from_url(data_dir)
        if text:
            documents.append(text)
            metadatas.append({'filename': data_dir})
    else:
        files = os.listdir(data_dir)
        for filename in files:
            file_path = os.path.join(data_dir, filename)
            if filename.lower().endswith('.pdf'):
                text = process_local_pdf(file_path)
                if text:
                    documents.append(text)
                    metadatas.append({'filename': filename})
            else:
                try:
                    with open(file_path, 'r') as file:
                        documents.append(file.read())
                        metadatas.append({'filename': filename})
                        logging.info(f"Processed file: {filename}")
                except Exception as e:
                    logging.warning(f"Failed to read file {filename}: {e}")

    collection = get_or_create_collection(collection_name, persist_directory, api_key)

    count = collection.count()
    logging.info(f'Collection contains {count} documents before addition.')
    ids = [str(i) for i in range(count, count + len(documents))]

    try:
        collection.add(ids=ids, documents=documents, metadatas=metadatas)
        new_count = collection.count()
        logging.info(f'Added {new_count - count} documents to the collection.')
    except Exception as e:
        logging.error(f"Failed to add documents to the collection: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Embed data from a directory or a URL into a Chroma collection')
    parser.add_argument('--data_directory', required=True, type=str, help='The directory where your text files are stored, or a URL to a PDF document')
    parser.add_argument('--persist_directory', required=True, type=str, help='The directory where you want to store the Chroma collection')
    parser.add_argument('--collection_name', required=True, type=str, help='The name of the Chroma collection')

    args = parser.parse_args()

    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    if not OPENAI_API_KEY:
        logging.error("OpenAI API key is not set. Set the OPENAI_API_KEY environment variable.")
        sys.exit(1)

    main(data_dir=args.data_directory, collection_name=args.collection_name, persist_directory=args.persist_directory, api_key=OPENAI_API_KEY)