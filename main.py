import os
import sys
import logging
from typing import List, Dict, Union

import openai
import tiktoken
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import psycopg2

logging.basicConfig(level=logging.INFO)

def check_openai_api_key():
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    if not OPENAI_API_KEY:
        logging.error("Error: You need to set your OpenAI API key as the OPENAI_API_KEY environment variable in Repl.it Secrets. Get your key from https://platform.openai.com/account/api-keys")
        sys.exit(1)
    return OPENAI_API_KEY

def get_user_query():
    while True:
        query = input("Query: ").strip()
        if not query:
            print("Please enter a question. Ctrl+C to Quit.", file=sys.stderr)
        else:
            return query

def build_prompt(query: str, context: List[str]) -> List[Dict[str, str]]:
    """
    Builds a prompt for the LLM.
    """

    system = {
        'role': 'system',
        'content': 'I am going to ask you a question, which I would like you to answer based only on the provided context, and not any other information. If there is not enough information in the context to answer the question, say "I am not sure", then try to make a guess. Break your answer up into nicely readable paragraphs.'
    }
    user = {
        'role': 'user',
        'content': f"""
            The question is {query}. Here is all the context you have: {(' ').join(context)}
            """
    }

    return [system, user]

def get_chatGPT_response(query: str, context: List[str]) -> str:
    """
    Queries the GPT API to get a response to the question.
    """

    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=build_prompt(query, context),
    )

    return response.choices[0].message.content

def filter_results(results: Dict[str, List[Union[Dict[str, str], str]]], max_prompt_length: int = 3900) -> Dict[str, List[Union[Dict[str, str], str]]]:
    """
    Filters the query results so they don't exceed the model's token limit.
    """

    contexts = []
    sources = []

    # We use the tokenizer from the same model to get a token count
    tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo')

    total = 0
    for i, c in enumerate(results['documents'][0]):
        total += len(tokenizer.encode(c))
        if total <= max_prompt_length:
            contexts.append(c)
            sources.append(results['metadatas'][0][i]['page_number'])
        else:
            break
    return contexts, sources

def get_collection(collection_name='russel_norvig', persist_directory='chroma-russel-norvig'):
    """
    Instantiates the Chroma client, and returns the collection.
    """

    # Instantiate the Chroma client. We use persistence to load the already existing index.
    # Learn more at docs.trychroma.com
    client = chromadb.Client(settings=Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_directory))

    # Get the collection. We use the same embedding function we used to create the index.
    OPENAI_API_KEY = check_openai_api_key()
    embedding_function = OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY)
    collection = client.get_collection(name=collection_name, embedding_function=embedding_function)

    return collection

def main():
    # Retrieve PostgreSQL connection details from environment variables
    pg_database = os.getenv('PGDATABASE')
    pg_host = os.getenv('PGHOST')
    pg_port = os.getenv('PGPORT')
    pg_user = os.getenv('PGUSER')
    pg_password = os.getenv('PGPASSWORD')

    # Construct PostgreSQL connection string
    DATABASE_URL = f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_database}"

    # Connect to PostgreSQL database
    try:
        conn = psycopg2.connect(DATABASE_URL)
        logging.info("Connected to PostgreSQL database")
    except psycopg2.Error as e:
        logging.error(f"Unable to connect to PostgreSQL database: {e}")
        sys.exit(1)

    collection = get_collection()

    print("""
    This is a demo demonstrating how to plug knowledge into LLMs using Chroma.

    We've turned the popular AI textbook, "Artificial Intelligence: A Modern Approach" by Stuart Russell and Peter Norvig, into pluggable knowledge - this demo lets you ask questions about the content of the book, and get answers with page references. 

    Input a Query to get answers. Some queries to try:
    - What is an intelligent agent?
    - What are backtracking search algorithms?
    - What can you tell me about machine learning?
    - What is the difference between a computer and a robot?

    """)

    while True:
        query = get_user_query()

        results = collection.query(query_texts=[query], n_results=5, include=['documents', 'metadatas'])

        MAX_PROMPT_LENGTH = 3900
        contexts, sources = filter_results(results, MAX_PROMPT_LENGTH)

        response = get_chatGPT_response(query, contexts)

        print(response)
        print(f"Source pages: {sources}")
        print()

if __name__ == '__main__':
    main()