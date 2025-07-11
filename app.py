import streamlit as st
from langgraph_agent import ask
from langgraph_agent import visualize_graph, get_graph_mermaid_png

st.set_page_config(page_title="Product Chatbot", page_icon="🤖")

st.title("🛒 KingArthurBaking Product Chatbot")
st.write("Ask me anything about products, prices, or recipes!")

# Session state to store chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Type your question here..."):
    # Add user message to chat history and display it
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get answer from backend with full chat history
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Pass the current chat history to the backend
            answer, updated_history = ask(prompt, st.session_state["messages"])
            st.markdown(answer)
            
            # Update the session state with the new history from backend
            st.session_state["messages"] = updated_history

with st.expander("Show LangGraph Flow Visualization"):
    png_bytes = get_graph_mermaid_png()
    if png_bytes:
        st.image(png_bytes, caption="LangGraph Flow (Mermaid PNG)")
    else:
        ascii_diagram, mermaid_code = visualize_graph()
        st.subheader("ASCII Diagram")
        st.code(ascii_diagram, language="text")
        st.subheader("Mermaid Diagram Code")
        st.code(mermaid_code, language="mermaid")