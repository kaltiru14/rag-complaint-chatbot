# 03_rag_evaluation.py with error handling

import os

try:
    import pandas as pd
except ImportError as e:
    raise ImportError("Please install pandas: pip install pandas") from e

try:
    from rag_pipeline import answer_question
except ImportError as e:
    raise ImportError("Could not import 'answer_question' from rag_pipeline. Ensure 03_rag_pipeline.py is in the path.") from e

# ------------------------------
# Ensure reports directory exists
# ------------------------------
reports_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../reports"))
os.makedirs(reports_dir, exist_ok=True)

# ------------------------------
# Representative questions
# ------------------------------
test_questions = [
    "Why are customers unhappy with credit cards?",
    "What issues do customers report about personal loans?",
    "Are there complaints regarding money transfer delays?",
    "What are common problems with savings accounts?",
    "Do customers report fraud issues with credit cards?",
    "Are there recurring complaints about loan interest rates?",
    "Which product has the highest number of complaint narratives?",
    "Are there complaints about customer service response times?",
    "What are the main sub-issues reported for personal loans?"
]

# ------------------------------
# Run RAG pipeline for each question
# ------------------------------
results = []

for q in test_questions:
    try:
        result = answer_question(q, top_k=5)
    except Exception as e:
        print(f"Warning: Failed to process question '{q}'. Error: {e}")
        results.append({
            "Question": q,
            "Generated Answer": "Error generating answer",
            "Retrieved Sources": "",
            "Quality Score (1-5)": "",
            "Comments/Analysis": f"Error: {e}"
        })
        continue

    # Collect top 3 retrieved sources with product and text snippet
    sources = []
    for s in result.get("retrieved_sources", [])[:3]:
        try:
            snippet = s['text'].replace("\n", " ").strip()[:150] + "..."
            product = s['metadata'].get('product', 'Unknown')
            sources.append(f"[{product}] {snippet}")
        except Exception as e:
            print(f"Warning: Failed to process retrieved source. Error: {e}")

    results.append({
        "Question": q,
        "Generated Answer": result.get("answer", "No answer"),
        "Retrieved Sources": "\n".join(sources),
        "Quality Score (1-5)": "",   # Fill manually after review
        "Comments/Analysis": ""      # Optional manual notes
    })

# ------------------------------
# Convert to DataFrame
# ------------------------------
try:
    df_results = pd.DataFrame(results)
except Exception as e:
    raise RuntimeError(f"Failed to convert results to DataFrame. Error: {e}")

# ------------------------------
# Save Markdown evaluation table
# ------------------------------
output_md_path = os.path.join(reports_dir, "rag_evaluation.md")
try:
    df_results.to_markdown(output_md_path, index=False)
    print(f"Markdown evaluation table saved to {output_md_path}")
except ImportError:
    print("Warning: 'tabulate' package not installed. Install it via 'pip install tabulate' to save Markdown table.")
except Exception as e:
    print(f"Warning: Failed to save Markdown table. Error: {e}")

# ------------------------------
# Save CSV for easier editing
# ------------------------------
output_csv_path = os.path.join(reports_dir, "rag_evaluation.csv")
try:
    df_results.to_csv(output_csv_path, index=False)
    print(f"CSV version saved to {output_csv_path}")
except Exception as e:
    print(f"Warning: Failed to save CSV file. Error: {e}")

# ------------------------------
# Optional: print summary table
# ------------------------------
try:
    print("\nEvaluation Table Preview:\n")
    print(df_results[["Question", "Generated Answer", "Retrieved Sources"]])
except Exception as e:
    print(f"Warning: Failed to print evaluation table preview. Error: {e}")
