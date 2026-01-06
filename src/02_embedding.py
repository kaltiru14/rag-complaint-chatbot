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
data_path = r"D:\tenx\week 6\rag-complaint-chatbot\data\processed\filtered_complaints.csv"
df = pd.read_csv(data_path)
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

# ------------------------------
# Ensure vector_store directory exists
# ------------------------------
vector_store_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../vector_store"))
os.makedirs(vector_store_path, exist_ok=True)
print(f"Saving vector store files in: {vector_store_path}")

# Save FAISS index
faiss_index_path = os.path.join(vector_store_path, "faiss_index.index")
faiss.write_index(index, faiss_index_path)

# Save metadata
metadata_path = os.path.join(vector_store_path, "metadata.pkl")
with open(metadata_path, "wb") as f:
    pickle.dump(metadata, f)

# Save all chunks
chunks_path = os.path.join(vector_store_path, "chunks.pkl")
with open(chunks_path, "wb") as f:
    pickle.dump(all_chunks, f)

print("FAISS index, metadata, and chunks.pkl saved successfully.")

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
