import os
import pandas as pd

# Paths
reports_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../reports"))
csv_path = os.path.join(reports_dir, "rag_evaluation.csv")
md_path = os.path.join(reports_dir, "rag_evaluation.md")

# Read CSV with tolerant encoding
try:
    df = pd.read_csv(csv_path, encoding="latin-1")
except Exception as e:
    raise RuntimeError(f"Failed to read CSV file at {csv_path}. Error: {e}")

# Save Markdown
try:
    df.to_markdown(md_path, index=False)
    print(f"Markdown table regenerated at {md_path}")
except ImportError:
    print("Warning: 'tabulate' not installed. Install via 'pip install tabulate' to save Markdown.")
except Exception as e:
    print(f"Failed to save Markdown. Error: {e}")
