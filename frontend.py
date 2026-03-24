import streamlit as st
import requests

st.set_page_config(page_title="University Support Chatbot", page_icon="🤖")

st.title("University Support Chatbot")
st.write("Ask questions about university documents.")

question = st.text_input("Enter your question:")

if st.button("Ask") and question:
    try:
        with st.spinner("Thinking..."):
            response = requests.post(
                "http://127.0.0.1:8000/ask", json={"question": question}, timeout=60
            )
    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to backend: {e}")
        st.stop()

    if response.status_code == 200:
        data = response.json()

        st.subheader("Answer")
        st.success(data["answer"])

        st.subheader("Sources")
        for src in data["sources"]:
            st.write(f"**{src['source']}** — page {src['page']}")
            with st.expander(f"View excerpt from {src['source']} page {src['page']}"):
                st.write(src["text"])
    else:
        st.error(f"Backend error: {response.status_code}")
