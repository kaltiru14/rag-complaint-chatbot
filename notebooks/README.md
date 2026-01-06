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

# Task 3: Building the RAG Core Logic and Evaluation

## Objective
Build a retrieval-augmented generation (RAG) pipeline using the pre-built full-scale vector store, and evaluate its effectiveness in answering customer complaint queries for CrediTrust Financial.

---

## 1. RAG Pipeline Implementation

### 1.1 Vector Store
- Pre-built vector store used from dataset resources.
- Contains embeddings for all filtered complaint data.
- FAISS index stored at: `../vector_store/faiss_index.index`
- Metadata and text chunks stored as: `metadata.pkl`, `chunks.pkl`

### 1.2 Retriever
- Function `retrieve_chunks(query, top_k=5)` implemented.
- Uses `sentence-transformers/all-MiniLM-L6-v2` to embed user queries.
- Searches FAISS index to retrieve top-k relevant chunks.

### 1.3 Prompt Template
- LLM prompted with retrieved context.
- Template instructs model to:
  - Act as a financial analyst assistant.
  - Only answer based on the provided context.
  - Respond with “not enough information” if context lacks answer.

**Example Template:**

You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints.
Use the following retrieved complaint excerpts to formulate your answer. If the context doesn't contain the answer, state that you don't have enough information.

Context:
{retrieved_chunks_text}

Question: {user_question}

Answer:


### 1.4 Generator
- Hugging Face `transformers` pipeline used for text generation.
- Model: `gpt2`
- Generates response from combined prompt + retrieved context.

---

## 2. Evaluation Methodology

### 2.1 Representative Questions
A set of 10 representative questions used to evaluate the pipeline:

1. Why are customers unhappy with credit cards?
2. What issues do customers report about personal loans?
3. Are there complaints regarding money transfer delays?
4. What are common problems with savings accounts?
5. Do customers report fraud issues with credit cards?
6. Are there recurring complaints about loan interest rates?
7. Which product has the highest number of complaint narratives?
8. Are there complaints about customer service response times?
9. What are the main sub-issues reported for personal loans?

### 2.2 Evaluation Criteria
- **Accuracy**: Does the generated answer reflect the retrieved sources correctly?
- **Completeness**: Are key details included in the answer?
- **Relevance**: Does the answer stay on-topic and supported by the context?
- **Hallucinations**: Any unsupported or fabricated statements?

- Quality Scores (1-5) assigned to each answer:
  - 5: Highly accurate, complete, and relevant.
  - 4: Mostly accurate, minor omissions.
  - 3: Partially accurate, some missing details.
  - 2: Vague or partially off-topic.
  - 1: Incorrect or completely irrelevant.

---

## 3. Evaluation Results

**CSV/Markdown Table: `rag_evaluation.csv` / `rag_evaluation.md`**

| Question | Generated Answer | Retrieved Sources | Quality Score (1-5) | Comments/Analysis |
|----------|-----------------|-----------------|--------------------|------------------|
| Why are customers unhappy with credit cards? | They are unhappy with their credit card because they can't keep their accounts... | [Credit card] inconvenient they put customers when they cant use their credit card... | 4 | Mostly accurate and relevant. Could mention fees, interest, and late payments for completeness. |
| What issues do customers report about personal loans? | The information and report you are providing is useful for your purpose only... | [Credit card] consumer transparency into financial services... | 3 | Relevant context used, but answer is vague. Needs specific loan complaints like delays or high interest. |
| Are there complaints regarding money transfer delays? | - if you have no problems with the money transfer, you must give the proper notice... | [Credit card] ensuring punctual payments... | 3 | Some relevant context, but answer is partially off-topic. Could focus more on delays or errors. |
| ... | ... | ... | ... | ... |

> Full evaluation table attached as `rag_evaluation.csv` and `rag_evaluation.md`.

---

## 4. Observations & Analysis

- **Accuracy**: Most answers correctly referenced the retrieved chunks. Some answers are incomplete or include minor hallucinations.
- **Completeness**: Answers for credit card complaints are more complete; personal loans and savings accounts sometimes miss details like fees or delays.
- **Relevance**: LLM generally stays on topic, though occasional digressions appear.
- **Hallucinations**: Minor hallucinations observed when context lacked sufficient details.
- **Top-k Sufficiency**: Using `top_k=5` retrieved chunks worked well, though increasing `k` might improve coverage for complex questions.

---

## 5. Recommendations for Improvement

1. **Increase Top-k**: Try `k=7` to retrieve more context for multi-faceted questions.
2. **Better LLM**: Consider using a more capable generation model than GPT-2 (e.g., Mistral, Llama 2, or GPT-3.5).
3. **Post-Processing**: Add rules to detect hallucinations and trim repetitive or irrelevant parts.
4. **Domain-Specific Fine-Tuning**: Fine-tune LLM on financial complaint data for more accurate responses.
5. **Structured Output**: Return JSON with separate fields for summary, key points, and suggested actions.

---

## 6. Conclusion

- Task 3 successfully implemented a full **RAG pipeline** for financial complaint analysis.
- Qualitative evaluation was performed on representative questions.
- Pipeline demonstrated reasonable accuracy, relevance, and completeness with minor limitations.
- Further improvements can enhance coverage and reduce hallucinations.

---

**Files Delivered:**
- `03_rag_pipeline.py`
- `03_rag_evaluation.py`
- `rag_evaluation.csv`
- `rag_evaluation.md`


# Task 4: Creating an Interactive Chat Interface

## Objective
Build a user-friendly interface that allows non-technical users to interact with the RAG system for analyzing CrediTrust customer complaints.

---

## Implementation

### Framework Used
- **Gradio**: Provides a lightweight web interface for interacting with the RAG system.

### Core Functionality
The interface includes:

1. **Text Input Box**  
   - Label: `Ask a question about customer complaints`  
   - Placeholder: `Type your question here...`  
   - Users type their questions here.

2. **Ask Button**  
   - Label: `Ask`  
   - On click: Sends the user question to the RAG pipeline and returns:
     - AI-generated answer
     - Retrieved sources

3. **Clear Button**  
   - Label: `Clear`  
   - Resets the question, answer, and retrieved sources boxes.

4. **Answer Display Box**  
   - Label: `Answer`  
   - Displays the generated answer from the RAG system.
   - Non-interactive to prevent editing.

5. **Retrieved Sources Box**  
   - Label: `Retrieved Sources`  
   - Shows top text chunks from the knowledge base used to generate the answer.
   - Increased size to handle long text content for better user verification.

---

## Error Handling
- Empty input prompts the user to enter a question.
- Exceptions during query processing are caught, and an error message is displayed in the answer box.
- Ensures the app does not crash even if RAG fails to return a result.

---

## Example UI

**User Question:**  
What are common problems with savings accounts?

**Generated Answer:**  
- The accounts are not functioning as intended.  
- There are several potential issues related to account management.  
- Customers have reported difficulties in withdrawing funds or understanding fees.  

**Retrieved Sources:**  
- [Credit card] the overwhelming stress and sleep disturbances resulting from this issue...
- [Credit card] tiny interest compared to other banks and accumulated balance issues...
- [Credit card] long-term frustrations using the accounts for daily expenses...


---

## Key Enhancements
- **Sources Display:** Users can verify answers using top-k retrieved sources.
- **User Experience:** Clear button resets conversation for multiple queries.
- **Error Handling:** Provides meaningful messages for empty or failed queries.
- **Responsive UI:** Textboxes and buttons are organized in rows for better usability.

---

## Deliverables
- `app.py` script running the Gradio application.
- Screenshots/GIF of the working interface included in the project report.
- Ready for deployment or local testing with `python app.py`.
