# 02_embedding.py

# ------------------------------
# Data handling
# ------------------------------
import pandas as pd
import numpy as np
import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer

# For reproducibility
np.random.seed(42)

# ------------------------------
# Custom text chunking function
# ------------------------------
def chunk_text(text, chunk_size=500, overlap=50):
    """
    Split text into chunks with given size and overlap.
    """
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# ------------------------------
# Load filtered complaints dataset
# ------------------------------
df = pd.read_csv("../data/processed/filtered_complaints.csv")
print(f"Dataset shape: {df.shape}")
print(df.head())

# ------------------------------
# Sample 12,000 complaints stratified by product
# ------------------------------
sample_size = 12000
products = df['Product'].unique()
sampled_df = df.groupby('Product', group_keys=False).apply(
    lambda x: x.sample(frac=min(1, sample_size/len(df)), random_state=42)
)

print("Sampled complaints by product:")
print(sampled_df['Product'].value_counts())
print(f"Sampled dataset shape: {sampled_df.shape}")

# ------------------------------
# Split complaints into chunks and build metadata
# ------------------------------
all_chunks = []
metadata = []

for idx, row in sampled_df.iterrows():
    chunks = chunk_text(row['clean_text'], chunk_size=500, overlap=50)
    all_chunks.extend(chunks)
    metadata.extend([{
        "complaint_id": row['Complaint ID'],
        "product": row['Product']
    }] * len(chunks))

print(f"Total text chunks: {len(all_chunks)}")

# ------------------------------
# Load pre-trained sentence transformer model
# ------------------------------
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embed_model = SentenceTransformer(model_name)

# Generate embeddings for all chunks
embeddings = embed_model.encode(all_chunks, show_progress_bar=True)
print(f"Embeddings shape: {embeddings.shape}")

# ------------------------------
# Create FAISS index
# ------------------------------
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings, dtype='float32'))

# Save index and metadata
os.makedirs("../vector_store", exist_ok=True)
faiss.write_index(index, "../vector_store/faiss_index.index")

with open("../vector_store/metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("FAISS index and metadata saved in vector_store/")

# ------------------------------
# Example retrieval
# ------------------------------
query = "Unauthorized credit card opened"
query_emb = embed_model.encode([query])

D, I = index.search(np.array(query_emb, dtype='float32'), k=3)

print("\nTop 3 similar complaints for the query:")
for i, idx in enumerate(I[0]):
    print(f"\nResult {i+1}:")
    print("Product:", metadata[idx]['product'])
    print("Complaint ID:", metadata[idx]['complaint_id'])
    print("Text chunk:", all_chunks[idx][:300], "...")
