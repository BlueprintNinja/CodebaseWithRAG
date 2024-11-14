import zipfile
import os
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import requests
import streamlit as st

# Initialize the embedding model
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Helper functions
def unzip_codebase(zip_path, extract_to="codebase"):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    return extract_to

def read_codebase(directory, max_chunk_size=1000):
    code_chunks = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith((".py", ".js", ".cpp")):  # Adjust extensions as needed
                with open(os.path.join(root, file), 'r') as f:
                    content = f.read()
                    for i in range(0, len(content), max_chunk_size):
                        code_chunks.append(content[i:i + max_chunk_size])
    return code_chunks

def generate_embeddings_with_huggingface(chunks):
    embeddings = embedding_model.encode(chunks)
    return embeddings

def retrieve_relevant_chunks(query, embeddings, code_chunks):
    query_embedding = embedding_model.encode([query])[0]
    similarities = [1 - cosine(query_embedding, emb) for emb in embeddings]
    relevant_chunks = sorted(zip(similarities, code_chunks), key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in relevant_chunks[:5]]

def generate_answer_with_ollama(query, relevant_chunks):
    context = "\n".join(relevant_chunks)
    response = requests.post(
        "http://localhost:11434/v1/chat/completions",
        json={
            "model": "hermes3",
            "messages": [
                {"role": "system", "content": "Answer the following query based on the provided context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ]
        }
    )
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return "Error: Could not retrieve a response."

# Streamlit App
st.title("Codebase Q&A with RAG using Ollama")
st.write("Upload a codebase ZIP file, and enter a query to get insights based on the code.")

# File Upload
uploaded_file = st.file_uploader("Upload your codebase (.zip file)", type="zip")
query = st.text_input("Enter your query", "How does the main authentication process work in this codebase?")

if uploaded_file and query:
    # Unzip the uploaded file
    with open("uploaded_codebase.zip", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    directory = unzip_codebase("uploaded_codebase.zip")
    
    # Process code and generate embeddings
    st.write("Processing codebase and generating embeddings...")
    code_chunks = read_codebase(directory)
    embeddings = generate_embeddings_with_huggingface(code_chunks)
    
    # Retrieve relevant chunks and generate answer
    st.write("Retrieving relevant information and generating answer...")
    relevant_chunks = retrieve_relevant_chunks(query, embeddings, code_chunks)
    answer = generate_answer_with_ollama(query, relevant_chunks)
    
    # Display the answer
    st.subheader("Answer")
    st.write(answer)
else:
    st.write("Please upload a codebase ZIP file and enter a query.")
