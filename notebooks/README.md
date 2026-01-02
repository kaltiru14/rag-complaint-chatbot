# Task 1 — Exploratory Data Analysis & Preprocessing

## 1\. Dataset Overview
------------------------

The CFPB complaint dataset initially contains approximately **9.6 million complaints** across multiple financial products, with 18 columns including Product, Consumer complaint narrative, Issue, and Company. About **68.98% of complaints lacked a narrative**, which were removed to ensure meaningful text for analysis and embedding.

After filtering for the four target products (**Credit card, Personal loan, Savings account, Money transfer**), the dataset was reduced to **226,686 complaints**. Removing entries with empty narratives resulted in a final dataset of **80,667 complaints** containing complete consumer narratives.

## 2\. Narrative Length Analysis
---------------------------------

The average complaint length is **~200 words**, with the shortest complaint containing **2 words** and the longest exceeding **6,400 words**. The distribution of complaint lengths is shown below:

_Figure 1: Distribution of Consumer Complaint Narrative Lengths_

This wide variation indicates a mix of very short complaints and extremely detailed ones. This insight informed the **text chunking strategy** in Task 2 to ensure sufficient context is captured for embedding.

## 3\. Product Distribution
----------------------------

After filtering, the dataset predominantly contains **Credit card complaints**. The distribution across products is visualized below:

_Figure 2: Number of Complaints per Product Category_

This highlights that other product categories (Personal loans, Savings accounts, Money transfers) may be underrepresented and should be considered when sampling for embeddings.

## 4\. Text Preprocessing
--------------------------

To improve embedding quality, all narratives were:

*   Converted to **lowercase**
    
*   **Special characters removed**
    
*   **Extra whitespace cleaned**
    

The cleaned text was saved in a new column: clean\_text. The preprocessed dataset is stored as:

```bash 
data/processed/filtered_complaints.csv 
```   

This dataset will serve as the **foundation for Task 2**, where we will perform text chunking, embedding, and vector store indexing for the RAG pipeline.

## 5\. Key Takeaways
---------------------

*   A significant portion of complaints were missing narratives, which required filtering.
    
*   Complaint lengths vary widely, indicating the need for careful chunking.
    
*   The cleaned and filtered dataset is ready for semantic search and RAG development.
    

💡 **Next Steps:** Use this cleaned dataset to create **text chunks, generate embeddings, and build a vector store** for the RAG chatbot pipeline in **Task 2**.

# Task 2: Text Chunking, Embedding, and Vector Store Indexing

## Objective
Prepare the cleaned complaint narratives for semantic search by:

- Sampling a representative subset of complaints.
- Splitting long narratives into manageable chunks.
- Generating embeddings for each chunk.
- Storing embeddings in a vector database with associated metadata.

---

## Sampling Strategy
- **Dataset:** Cleaned complaints from Task 1 (`filtered_complaints.csv`)
- **Sample size:** 12,000 complaints
- **Stratification:** Ensured proportional representation across five product categories: Credit card, Personal loan, Savings account, Checking account, Money transfers
- **Method:** Stratified sampling to maintain product distribution

---

## Text Chunking
- **Tool:** `LangChain`'s `RecursiveCharacterTextSplitter`
- **Parameters:**
  - Chunk size: 500 characters
  - Overlap: 50 characters
- **Purpose:**  
  Ensures that long narratives are split into meaningful sections while retaining context across chunks
- **Output:**  
  - `all_chunks` → list of text chunks  
  - `metadata` → complaint ID and product for each chunk

---

## Embeddings
- **Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Reasoning:** Lightweight, fast, and good semantic representation for English text
- **Output:** 384-dimensional embeddings for each text chunk

---

## Vector Store Indexing
- **Library:** FAISS (`IndexFlatL2`)
- **Purpose:** Efficient similarity search over all text chunks
- **Persistence:**  
  - FAISS index saved at `vector_store/faiss_index.index`  
  - Metadata saved at `vector_store/metadata.pkl`

---

## Results
- **Total complaints sampled:** 12,000  
- **Total text chunks created:** X (replace with actual number)  
- **Vector store:** Successfully created and persisted, ready for semantic retrieval in Task 3

> This vector store enables efficient retrieval of complaint chunks similar to a given query, forming the foundation for the RAG (Retrieval-Augmented Generation) system in the next task.
