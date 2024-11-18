import streamlit as st
import zipfile
from sentence_transformers import SentenceTransformer, util
import openai
import os
import re
import io

# Configure Streamlit page
st.set_page_config(page_title="RAG on Codebase", layout="wide")

# Set up OpenAI API key
openai.api_key = "key"
openai.api_base = "http://localhost:11434/v1"  # Use your local API URL

# Initialize Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_code_files(zip_file):
    """Extract code files from a .zip file and return a list of (file_path, 
content) tuples."""
    with zipfile.ZipFile(zip_file, 'r') as z:
        file_contents = []
        for name in z.namelist():
            if name.endswith(('.py', '.js', '.java', '.cpp', '.c', '.rb', 
'.go', '.rs', '.ts')):
                content = z.read(name).decode('utf-8')
                file_contents.append((name, content))
    return file_contents

def create_embeddings(file_contents):
    """Create embeddings for the code files."""
    texts = [content for _, content in file_contents]
    embeddings = model.encode(texts, convert_to_tensor=True)
    return file_contents, embeddings

def search_code(query, file_contents, embeddings, top_k=3):
    """Search for relevant code snippets using semantic similarity."""
    query_embedding = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, embeddings, 
top_k=top_k)[0]
    
    results = []
    for hit in hits:
        idx = hit['corpus_id']
        file_path, content = file_contents[idx]
        score = hit['score']
        snippet = '\n'.join(content.split('\n')[:10])  # Show the first 10 
        results.append((file_path, snippet, score))
    return results

def generate_response(query, context):
    """Generate a response using OpenAI's Chat Completions API with the 
given context."""
    prompt = f"Question: {query}\nContext:\n{context}\nAnswer:"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": ""}
    ]
    response = openai.ChatCompletion.create(
        model="hermes3",
        messages=messages,
        max_tokens=200
    )
    return response.choices[0].message['content']

# Streamlit UI
st.title("RAG on Codebase")

uploaded_file = st.file_uploader("Upload a .zip file containing your codebase", type=["zip"])

if uploaded_file:
    with st.spinner('Extracting and indexing code files...'):
        file_contents = extract_code_files(io.BytesIO(uploaded_file.getvalue()))
        file_contents, embeddings = create_embeddings(file_contents)
    
    query = st.text_input("Enter your question about the codebase:")
    
    if query:
        with st.spinner('Searching for relevant code snippets...'):
            results = search_code(query, file_contents, embeddings)
        
        context = ""
        for file_path, snippet, score in results:
            st.write(f"**File:** {file_path} (Score: {score:.2f})")
            st.code(snippet, language='python')
            context += f"**File:** {file_path}\n{snippet}\n\n"
        
        with st.spinner('Generating response...'):
            answer = generate_response(query, context)
        
        st.subheader("Answer:")
        st.write(answer)

else:
    st.write("Please upload a .zip file to get started.")
