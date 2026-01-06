# 03_rag_pipeline.py with error handling

import os
import pickle
import faiss

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError("Please install sentence-transformers: pip install sentence-transformers") from e

try:
    from transformers import pipeline
except ImportError as e:
    raise ImportError("Please install transformers: pip install transformers") from e

import numpy as np

# ------------------------------
# Paths
# ------------------------------
vector_store_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../vector_store"))
faiss_index_path = os.path.join(vector_store_path, "faiss_index.index")
metadata_path = os.path.join(vector_store_path, "metadata.pkl")
chunks_path = os.path.join(vector_store_path, "chunks.pkl")

# ------------------------------
# Load FAISS index, metadata, and chunks
# ------------------------------
try:
    print("Loading FAISS index...")
    index = faiss.read_index(faiss_index_path)
except Exception as e:
    raise FileNotFoundError(f"Could not load FAISS index from {faiss_index_path}. Error: {e}")

try:
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
except Exception as e:
    raise FileNotFoundError(f"Could not load metadata from {metadata_path}. Error: {e}")

try:
    with open(chunks_path, "rb") as f:
        all_chunks = pickle.load(f)
except Exception as e:
    raise FileNotFoundError(f"Could not load chunks from {chunks_path}. Error: {e}")

if len(all_chunks) != len(metadata):
    print(f"Warning: Number of chunks ({len(all_chunks)}) does not match metadata entries ({len(metadata)})")

print(f"Loaded {len(all_chunks)} text chunks.")

# ------------------------------
# Load embedding model
# ------------------------------
try:
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embed_model = SentenceTransformer(model_name)
except Exception as e:
    raise RuntimeError(f"Failed to load embedding model '{model_name}'. Error: {e}")

# ------------------------------
# Load LLM generator
# ------------------------------
try:
    generator = pipeline("text-generation", model="gpt2", tokenizer="gpt2", max_new_tokens=200)
except Exception as e:
    raise RuntimeError(f"Failed to load text-generation model. Error: {e}")

# ------------------------------
# Retriever function
# ------------------------------
def retrieve_chunks(query, top_k=5):
    """
    Given a user query, return top-k most similar text chunks and their metadata.
    """
    try:
        query_emb = embed_model.encode([query])
        D, I = index.search(np.array(query_emb, dtype='float32'), top_k)
    except Exception as e:
        raise RuntimeError(f"Error during embedding or FAISS search. Query: '{query}'. Error: {e}")

    results = []
    for idx in I[0]:
        try:
            results.append({
                "text": all_chunks[idx],
                "metadata": metadata[idx]
            })
        except IndexError:
            print(f"Warning: FAISS returned an invalid index {idx}")
    return results

# ------------------------------
# Prompt template
# ------------------------------
def build_prompt(question, retrieved_chunks):
    """
    Build prompt for the LLM using retrieved complaint excerpts.
    """
    context = "\n\n".join([f"- {chunk['text']}" for chunk in retrieved_chunks])
    prompt = f"""
You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints. 
Use the following retrieved complaint excerpts to formulate your answer. If the context doesn't contain the answer, state that you don't have enough information.

Context:
{context}

Question: {question}

Answer:
"""
    return prompt.strip()

# ------------------------------
# RAG function
# ------------------------------
def answer_question(question, top_k=5):
    """
    Complete RAG pipeline: retrieve chunks, build prompt, generate answer.
    """
    try:
        retrieved = retrieve_chunks(question, top_k=top_k)
        prompt = build_prompt(question, retrieved)
    except Exception as e:
        raise RuntimeError(f"Failed to prepare prompt for question '{question}'. Error: {e}")

    try:
        response = generator(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)
        answer_text = response[0]["generated_text"][len(prompt):].strip()  # remove prompt from output
    except Exception as e:
        answer_text = "Error generating answer."
        print(f"Warning: LLM generation failed for question '{question}'. Error: {e}")

    return {
        "question": question,
        "answer": answer_text,
        "retrieved_sources": retrieved
    }

# ------------------------------
# Example usage
# ------------------------------
if __name__ == "__main__":
    test_questions = [
        "Why are customers unhappy with credit cards?",
        "What issues do customers report about personal loans?",
        "Are there complaints regarding money transfer delays?",
        "Which companies have the most billing disputes?"
    ]

    for q in test_questions:
        try:
            result = answer_question(q, top_k=5)
            print("\n" + "="*60)
            print("Question:", result["question"])
            print("Answer:", result["answer"])
            print("Top Retrieved Sources:")
            for s in result["retrieved_sources"][:2]:
                print(f"- [{s['metadata']['product']}] {s['text'][:150]}...")
        except Exception as e:
            print(f"Error handling question '{q}': {e}")
