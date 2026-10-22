import os

import streamlit as st
from fpdf import FPDF
from groq import Groq, APIStatusError, APIConnectionError

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Multi-Agent Research Assistant", layout="wide")

st.title("🤖 Multi-Agent Research Assistant")
st.caption("Powered by Groq LLM (Fast • Reliable)")

# llama-3.1-8b-instant was shut down on 2026-08-16 (that caused the 404).
# Primary + fallback models. Override with a GROQ_MODEL env var / Streamlit secret.
DEFAULT_MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]


def get_secret(name: str):
    value = os.getenv(name)
    if value:
        return value
    try:
        return st.secrets.get(name, None)
    except Exception:
        return None


API_KEY = get_secret("GROQ_API_KEY")
if not API_KEY:
    st.error("GROQ_API_KEY is missing. Add it as an env var or in Streamlit Secrets.")
    st.stop()

client = Groq(api_key=API_KEY)

custom_model = get_secret("GROQ_MODEL")
MODELS = ([custom_model] if custom_model else []) + DEFAULT_MODELS

SYSTEM_PROMPT = (
    "You are a professional technology research analyst. "
    "Provide clear, structured, and detailed comparisons using headings, "
    "bullet points, pros/cons, and a concise final verdict. "
    "Avoid markdown tables."
)

# ---------------- INPUT ----------------
query = st.text_input(
    "Enter your research question:",
    placeholder="Compare Dell and ASUS laptops",
)


# ---------------- HELPERS ----------------
def clean_text(text: str) -> str:
    """Make text safe for FPDF's built-in latin-1 fonts."""
    replacements = {
        "—": "-", "–": "-", "’": "'", "‘": "'",
        "“": '"', "”": '"', "•": "-", "…": "...", "\u00a0": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = text.replace("**", "").replace("__", "").replace("`", "")
    text = text.replace("#", "")
    return text.encode("latin-1", "ignore").decode("latin-1")


def generate_response(user_query: str) -> str:
    last_error = None
    for model in MODELS:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_query},
                ],
                temperature=0.3,
                # gpt-oss models are reasoning models: reasoning tokens count
                # toward this limit, so leave plenty of room.
                max_completion_tokens=4000,
                reasoning_effort="low",
            )
            content = response.choices[0].message.content
            if content and content.strip():
                return content
            last_error = RuntimeError(f"Model {model} returned an empty response.")
        except APIStatusError as e:
            # 404 = model retired/unknown, 400 = bad param -> try next model
            if e.status_code in (400, 404):
                last_error = e
                continue
            raise
    raise last_error or RuntimeError("No model available.")


def build_pdf(text: str) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)

    for line in clean_text(text).split("\n"):
        line = line.strip()
        if not line:
            pdf.ln(4)
            continue
        pdf.multi_cell(0, 6, line, new_x="LMARGIN", new_y="NEXT")

    return bytes(pdf.output())


# ---------------- RUN BUTTON ----------------
if st.button("Run Research"):
    if not query.strip():
        st.warning("Please enter a research question.")
    else:
        try:
            with st.spinner("Analyzing with Groq LLM..."):
                result = generate_response(query)
        except APIConnectionError:
            st.error("Could not reach the Groq API. Check your internet connection.")
            st.stop()
        except APIStatusError as e:
            st.error(f"Groq API error ({e.status_code}): {e.message}")
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            st.stop()

        st.success("Research Completed!")

        st.subheader("📄 Generated Report")
        st.markdown(result)

        os.makedirs("reports", exist_ok=True)
        with open("reports/output.md", "w", encoding="utf-8") as f:
            f.write(result)

        try:
            pdf_bytes = build_pdf(result)
            st.download_button(
                label="⬇️ Download PDF Report",
                data=pdf_bytes,
                file_name="research_report.pdf",
                mime="application/pdf",
            )
        except Exception as e:
            st.warning(f"PDF could not be generated ({e}). The report above is still available.")
            st.download_button(
                label="⬇️ Download Markdown Report",
                data=result,
                file_name="research_report.md",
                mime="text/markdown",
            )
