# ✅ Import libraries
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import numpy as np
import gradio as gr
import faiss

# ✅ Load models once (not inside functions)
embedder = SentenceTransformer("all-MiniLM-L6-v2")
generator = pipeline("text2text-generation", model="google/flan-t5-small")

# ✅ Process PDF function
def process_pdf(pdf_file):
    reader = PdfReader(pdf_file.name)
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"
    chunks = [text[i:i+400] for i in range(0, len(text), 350)]
    embeddings = embedder.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return (index, chunks), f"✅ PDF loaded successfully! {len(chunks)} chunks created."

# ✅ Answer generation function
def answer_question(query, state):
    if not state:
        return "⚠️ Please process a PDF first."
    index, chunks = state
    q_emb = embedder.encode([query])
    distances, indices = index.search(np.array(q_emb), k=3)
    context = "\n".join([chunks[i] for i in indices[0]])
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer based only on the context."
    print("🔹 Prompt:", prompt[:300])  # Debugging line
    result = generator(prompt, max_length=250, temperature=0.7)
    return result[0]["generated_text"]

# ✅ Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Mini RAG Chatbot")  # 👈 simplified title

    pdf = gr.File(label="📁 Upload PDF", scale=0.5, height=110)

    status = gr.Textbox(label="Status", interactive=False)
    state = gr.State()

    btn = gr.Button("📘 Process PDF")
    btn.click(process_pdf, inputs=[pdf], outputs=[state, status])

    query = gr.Textbox(label="Ask a question")
    answer = gr.Textbox(label="Answer", lines=7)  # 👈 bigger visible area (no scroll)
    ask = gr.Button("🚀 Generate Answer")
    ask.click(answer_question, inputs=[query, state], outputs=[answer])

demo.launch(share=True)
