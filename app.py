"""
Gradio query interface.

Run:
    python app.py
then open http://localhost:7860

All domain-specific strings (title, description, examples, placeholder) come from
the active DomainConfig, so the same UI serves any corpus.

Prerequisite: build the index once with `python build_index.py`.
"""

import gradio as gr

from src.config import ACTIVE_CONFIG
from src.pipeline import ask


def handle_query(question):
    if not question or not question.strip():
        return "Please enter a question.", "", ""
    result = ask(question)

    sources = "\n".join(f"• {s}" for s in result["sources"]) or "(none)"
    chunk_view = "\n\n".join(
        f"[{c['label']} | distance={c['distance']}]\n{c['text']}"
        for c in result["chunks"]
    )
    return result["answer"], sources, chunk_view


with gr.Blocks(title=ACTIVE_CONFIG.ui_title) as demo:
    gr.Markdown(f"# {ACTIVE_CONFIG.ui_title}\n{ACTIVE_CONFIG.ui_description}\n\n")
    inp = gr.Textbox(
        label="Your question",
        placeholder=ACTIVE_CONFIG.ui_placeholder,
    )
    btn = gr.Button("Ask", variant="primary")
    answer = gr.Textbox(label="Answer", lines=6)
    sources = gr.Textbox(label="Retrieved from (sources)", lines=4)
    with gr.Accordion("Retrieved chunks (what the answer was built from)", open=False):
        chunks = gr.Textbox(label="", lines=14)

    if ACTIVE_CONFIG.examples:
        gr.Examples(examples=ACTIVE_CONFIG.examples, inputs=inp)

    btn.click(handle_query, inputs=inp, outputs=[answer, sources, chunks])
    inp.submit(handle_query, inputs=inp, outputs=[answer, sources, chunks])


if __name__ == "__main__":
    demo.launch()
