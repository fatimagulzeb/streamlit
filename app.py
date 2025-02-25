import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# Load precomputed document embeddings (Assuming embeddings.npy and documents.txt exist)
embeddings = np.load("embeddings.npy")
with open("documents.txt", "r", encoding="utf-8") as f:
    documents = f.readlines()
def retrieve_top_k(query_embedding, embeddings, k=10):
    similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings)[0]
    top_k_indices = similarities.argsort()[-k:][::-1]
    return [(documents[i], similarities[i]) for i in top_k_indices]



# Input query
query = st.text_input("Enter your query:")

def get_query_embedding(query):
    return np.random.rand(embeddings.shape[1]) # Replace with actual embedding function
# Initialize results as an empty list before the button is clicked
results = []

if st.button("Search"):
    query_embedding = get_query_embedding(query)
    results = retrieve_top_k(query_embedding, embeddings)

# Display results (even if the button is not clicked)
st.write("### Top 10 Relevant Documents:")
for doc, score in results:
    st.write(f"- **{doc.strip()}** (Score: {score:.4f})")
