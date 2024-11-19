import streamlit as st
import zipfile
import os
from sentence_transformers import SentenceTransformer, util
import requests
import json

# Set up OpenAI API
openai_api_key = os.getenv("OPENAI_API_KEY")

# Function to extract zip file
def extract_zip(input_zip):
    with zipfile.ZipFile(input_zip, 'r') as z:
        z.extractall('temp_codebase')

# Function to read all files in a directory and its subdirectories
def read_files_in_dir(dir_path):
    code_snippets = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith(('.py', '.java', '.js', '.cpp', '.c', '.h', '.txt')):
                with open(os.path.join(root, file), 'r') as f:
                    code_snippets.append(f.read())
    return code_snippets

# Function to perform RAG
def rag_search(query, code_snippets):
    # Load the Sentence Transformer model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Encode the query and code snippets
    query_embedding = model.encode(query)
    snippet_embeddings = model.encode(code_snippets)

    # Find the most similar code snippet
    hits = util.semantic_search(query_embedding, snippet_embeddings, 
top_k=3)
    return [code_snippets[hit['corpus_id']] for hit in hits[0]]

# Function to call OpenAI Chat Completions API
def get_chat_completion(query, context):
    url = "http://localhost:11434/v1/chat/completions"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {openai_api_key}'
    }
    data = {
        "model": "hermes3",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query},
            {"role": "assistant", "content": context}
        ]
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response.json()['choices'][0]['message']['content']

# Streamlit App
st.title("Codebase RAG with OpenAI and Sentence Transformers")

# File upload
uploaded_file = st.file_uploader("Upload a .zip file containing the codebase", type="zip")
if uploaded_file is not None:
    # Extract the zip file
    extract_zip(uploaded_file)
    
    # Read all files in the extracted directory
    code_snippets = read_files_in_dir('temp_codebase')
    
    # User input for query
    query = st.text_input("Enter your question about the codebase:")
    if st.button("Search"):
        # Perform RAG search
        relevant_snippets = rag_search(query, code_snippets)
        
        # Combine relevant snippets into context
        context = "\n\n".join(relevant_snippets)
        
        # Get chat completion from OpenAI
        answer = get_chat_completion(query, context)
        
        st.subheader("Answer")
        st.write(answer)

# Clean up temporary files
if os.path.exists('temp_codebase'):
    for root, dirs, files in os.walk('temp_codebase', topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir('temp_codebase')
