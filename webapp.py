# streamlit_app.py
import streamlit as st
from agents.meta_agent import run_meta_agent
import agents.core

st.set_page_config(page_title="Paralegal & Attorney", layout="wide")

with st.sidebar:
    st.header("⚙️ Settings")
    mode = st.radio(
        "Task Mode",
        ["Generic", "Generate Case Study", "Summarise Statutes", "Generate Court Style Document"]
    )
    st.markdown("---")

st.markdown("<h1 style='text-align:center;'>⚖️ Paralegal & Attorney</h1>", unsafe_allow_html=True)
st.caption("Your AI-powered legal research assistant.")

chat_container = st.container()

if "history" not in st.session_state:
    st.session_state.history = []

for role, msg in st.session_state.get("history", []):
    with chat_container.chat_message(role):
        st.write(msg)

query = st.chat_input("Type your question...")

if query:
    with chat_container.chat_message("user"):
        st.write(query)

    st.session_state.history.append(("user", query))

    with chat_container.chat_message("assistant"):
        status = st.status("⏳ Thinking...", expanded=True)

        def update(msg):
            status.write(msg)

        result = run_meta_agent(query, mode, status_callback=update)
        status.update(label="Done", state="complete")

        if result.final_answer:
            st.markdown("### ✅ Final Answer")
            st.markdown(
                f"<div style='background:#eef9ee;padding:15px;border-radius:10px;'>{result.final_answer}</div>",
                unsafe_allow_html=True
            )

        if result.reasoning:
            with st.expander("🧩 Reasoning"):
                st.write(result.reasoning)

        if result.rag_citations:
            st.markdown("### 📚 Citations Used")
            for c in result.rag_citations:
                st.markdown(
                    f"<div style='padding:10px;background:#F7F7F9;border-radius:8px;margin-bottom:8px;'>{c}</div>",
                    unsafe_allow_html=True
                )

        if result.pdf_bytes:
            st.download_button(
                "📄 Download PDF",
                data=result.pdf_bytes,
                file_name="document.pdf"
            )

    st.session_state.history.append(("assistant", result.final_answer or "…"))
