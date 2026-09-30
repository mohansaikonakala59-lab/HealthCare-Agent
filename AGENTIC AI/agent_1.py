from langchain.agents import create_agent
from langchain_core.tools import tool
from pypdf import PdfReader
from llm_config import create_llm
from tool_calls import web_search, get_lab_report_data
import gradio as gr

current_doc = {"name": None, "text": None}  # most recently uploaded PDF


@tool
def get_uploaded_pdf_text() -> str:
    """Return the text of the PDF the user most recently uploaded in the chat."""
    if not current_doc["text"]:
        return "No PDF has been uploaded yet."
    return f"File: {current_doc['name']}\n\n{current_doc['text'][:30000]}"


SYSTEM_PROMPT = """You are a Health Insights Assistant. Review healthcare documents
and give general, non-diagnostic lifestyle and follow-up guidance.

Source of data:
- If the user has uploaded a PDF, call `get_uploaded_pdf_text` and use that document.
- Otherwise use `get_lab_report_data` (the default lab report) with simple keywords.
- If the user uploads a new PDF, forget the previous document and use the new one.

FULL REVIEW (when asked to review a report):
1. Retrieve the document.
2. List every test flagged High, Low, or Abnormal with its value and reference range.
3. Optionally use `web_search` for reputable general info on unfamiliar flags.
4. Reply in Markdown:
## 🩺 Report Summary
### ⚠️ Flagged Results
A table: | Test | Result | Reference Range | Flag | (Flag prefixed with 🔴 High, 🔵 Low, 🟠 Abnormal)
### 💡 General Guidance
A `####` subheading per flagged test with 2-3 general lifestyle/dietary bullets.
### 📋 Next Steps
A short bullet list (e.g. re-test timing, discuss with your doctor).

If the document is not a lab report (e.g. discharge summary, prescription), summarize
it in plain language instead. For FOLLOW-UP QUESTIONS, answer concisely and don't
repeat the full format.

Rules: never give a diagnosis, treatment plan, or medication advice. If a field is
missing, write "Not specified". Always end with:
**This is general information, not medical advice — please consult your doctor.**
"""

health_agent = create_agent(
    model=create_llm(),
    tools=[web_search, get_lab_report_data, get_uploaded_pdf_text],
    system_prompt=SYSTEM_PROMPT,
)


def to_text(content):
    """Turn str or a list of content blocks into plain text ('' if no text)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b if isinstance(b, str) else b.get("text", "")
            for b in content
            if isinstance(b, (str, dict))
        )
    return ""


def file_path(f):
    return f if isinstance(f, str) else f.get("path", "")


def load_pdf(path):
    text = "\n".join(p.extract_text() or "" for p in PdfReader(path).pages).strip()
    current_doc.update(name=path.replace("\\", "/").split("/")[-1], text=text)
    return bool(text)


def chat(message, history):
    note = ""
    for f in message.get("files", []) or []:
        path = file_path(f)
        if path.lower().endswith(".pdf"):
            try:
                ok = load_pdf(path)
            except Exception as e:
                return f"⚠️ I couldn't open that PDF: `{e}`"
            if not ok:
                return "⚠️ I couldn't read any text in that PDF (it may be a scanned image)."
            note = f"[The user just uploaded a new PDF: {current_doc['name']}]\n"

    # keep only text from earlier turns (file-only turns become empty and are skipped)
    messages = []
    for m in history:
        text = to_text(m["content"])
        if text.strip():
            messages.append({"role": m["role"], "content": text})

    user_text = message.get("text") or "Review this document."
    messages.append({"role": "user", "content": note + user_text})

    try:
        reply = health_agent.invoke({"messages": messages})["messages"][-1].content
    except Exception as e:
        return f"⚠️ Something went wrong: `{e}`"
    return to_text(reply)


# ---------------------------- UI ----------------------------

HEADER = """
<div class="hero">
  <div class="hero-icon">🩺</div>
  <h1>Health Insights Agent</h1>
  <p>Upload any lab report or healthcare PDF and get clear, general lifestyle guidance in seconds.</p>
  <div class="chips">
    <span>📄 PDF Analysis</span>
    <span>🔬 Flagged Results</span>
    <span>🌐 Web-Backed Research</span>
    <span>💬 Follow-up Chat</span>
  </div>
</div>
"""

FOOTER = """
<div class="disclaimer">
  ⚕️ For general information only. Not a diagnosis or medical advice. Always consult your doctor.
</div>
"""

PLACEHOLDER = """
<div style="text-align:center; padding: 2rem 1rem;">
  <div style="font-size:3rem;">👋</div>
  <h3 style="margin:0.5rem 0;">How can I help with your health report?</h3>
  <p style="opacity:0.7;">Attach a PDF with the 📎 button, or try an example below.</p>
</div>
"""

CSS = """
.gradio-container {max-width: 900px !important; margin: auto !important;}
footer {display: none !important;}

.hero {
    text-align: center;
    padding: 2rem 1.5rem;
    border-radius: 20px;
    background: linear-gradient(135deg, #0d9488 0%, #0891b2 50%, #6366f1 100%);
    color: white;
    box-shadow: 0 10px 30px rgba(13, 148, 136, 0.35);
    margin-bottom: 1rem;
}
.hero-icon {font-size: 3rem; margin-bottom: 0.25rem;}
.hero h1 {margin: 0; font-size: 2.2rem; font-weight: 800; color: white;}
.hero p {margin: 0.5rem auto 1rem; max-width: 560px; opacity: 0.95; font-size: 1.05rem;}

.chips {display: flex; flex-wrap: wrap; gap: 0.5rem; justify-content: center;}
.chips span {
    background: rgba(255, 255, 255, 0.2);
    backdrop-filter: blur(6px);
    border: 1px solid rgba(255, 255, 255, 0.35);
    padding: 0.35rem 0.85rem;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
}

.disclaimer {
    text-align: center;
    font-size: 0.85rem;
    padding: 0.75rem 1rem;
    margin-top: 0.75rem;
    border-radius: 12px;
    background: rgba(245, 158, 11, 0.12);
    border: 1px solid rgba(245, 158, 11, 0.4);
}
"""

theme = gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="cyan",
    neutral_hue="slate",
    radius_size="lg",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
)

with gr.Blocks(title="Health Insights Agent") as agent:
    gr.HTML(HEADER)
    gr.ChatInterface(
        fn=chat,
        multimodal=True,
        chatbot=gr.Chatbot(height=520, placeholder=PLACEHOLDER, show_label=False),
        textbox=gr.MultimodalTextbox(
            file_types=[".pdf"],
            file_count="single",
            placeholder="Ask a question or attach a PDF...",
            show_label=False,
        ),
        examples=[
            {"text": "Review my lab report and give me general health and lifestyle recommendations."},
            {"text": "Which of my results are flagged, and what do they usually mean?"},
            {"text": "What foods could help with my flagged results?"},
        ],
    )
    gr.HTML(FOOTER)

if __name__ == "__main__":
    agent.launch(theme=theme, css=CSS)