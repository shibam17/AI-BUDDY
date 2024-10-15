from flask import Flask, request, jsonify
import openai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

app = Flask(__name__)

# Set up OpenAI API (replace with your own key)
openai.api_key = "your-openai-api-key"

# Sentence embedding model
embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# FAISS vector database setup
dimension = 384  # Adjust to match the embedding model
index = faiss.IndexFlatL2(dimension)

# In-memory document store (mapping doc_id to content)
doc_store = []

# Function to store document vectors in FAISS
def store_document(text, doc_id):
    vector = embedder.encode(text).astype('float32')
    index.add(np.array([vector]))  # Add to FAISS index
    doc_store.append((doc_id, text))  # Store the document

# Function to retrieve documents based on a query
def retrieve_relevant_documents(query, top_k=3):
    query_vector = embedder.encode(query).astype('float32').reshape(1, -1)
    distances, indices = index.search(query_vector, top_k)
    return [doc_store[i][1] for i in indices[0]]  # Retrieve top_k relevant documents

# OpenAI text generation (using context)
def generate_answer(question, context):
    input_text = f"Context: {context}\n\nQuestion: {question}"
    response = openai.Completion.create(
        engine="text-davinci-003",  # You can use GPT-4 if available
        prompt=input_text,
        max_tokens=150,
        temperature=0.7
    )
    return response.choices[0].text.strip()

# Flask route for handling user queries
@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data.get('question')
    
    # Retrieve relevant documents using FAISS
    relevant_docs = retrieve_relevant_documents(question)
    
    # Generate an answer based on the retrieved documents
    context = " ".join(relevant_docs)  # Combine documents as context
    answer = generate_answer(question, context)
    
    return jsonify({"answer": answer})

# Example: Add a document to the database (for demo purposes)
store_document("Photosynthesis is the process by which plants make their own food using sunlight.", 1)
store_document("Mitochondria is the powerhouse of the cell.", 2)

if __name__ == '__main__':
    app.run(debug=True)