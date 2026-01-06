# app.py

import gradio as gr
from src.rag_pipeline import answer_question

# ------------------------------
# Function to handle user query
# ------------------------------
def chat_with_rag(user_input):
    """
    Takes user question, queries the RAG pipeline, and returns a cleaned answer + sources.
    """
    if not user_input.strip():
        return "Please enter a question.", ""

    try:
        # Get answer from RAG pipeline
        result = answer_question(user_input, top_k=5)
        answer = result.get("answer", "No answer generated.")

        # Deduplicate and trim repeated lines
        lines = answer.split("\n")
        unique_lines = []
        for line in lines:
            line = line.strip()
            if line and line not in unique_lines:
                unique_lines.append(line)
        # Limit to top 5 key points for readability
        answer = "\n".join(unique_lines[:5])

        # Format retrieved sources
        sources = result.get("retrieved_sources", [])
        if sources:
            sources_text = "\n\n".join(
                [f"[{s['metadata'].get('product','Unknown')}] {s['text'][:500]}..." for s in sources]
            )
        else:
            sources_text = "No sources retrieved."

        return answer, sources_text

    except Exception as e:
        # Error handling
        return f"Error: {str(e)}", ""


# ------------------------------
# Gradio Interface
# ------------------------------
with gr.Blocks() as demo:
    gr.Markdown("<h2>CrediTrust Complaint Analysis Chatbot</h2>")

    # User input row
    with gr.Row():
        user_input = gr.Textbox(
            label="Ask a question about customer complaints",
            placeholder="Type your question here..."
        )

    # Buttons row
    with gr.Row():
        ask_btn = gr.Button("Ask")
        clear_btn = gr.Button("Clear")

    # Answer output
    with gr.Row():
        output_answer = gr.Textbox(
            label="Answer",
            interactive=False,
            lines=7  # Increased for readability
        )

    # Retrieved sources output
    with gr.Row():
        output_sources = gr.Textbox(
            label="Retrieved Sources",
            interactive=False,
            lines=20,  # Increased for longer text
            placeholder="Sources used by the AI will appear here..."
        )

    # Button actions
    ask_btn.click(chat_with_rag, inputs=user_input, outputs=[output_answer, output_sources])
    clear_btn.click(lambda: ("", ""), inputs=[], outputs=[output_answer, output_sources])


# ------------------------------
# Launch app
# ------------------------------
if __name__ == "__main__":
    demo.launch(share=True)
