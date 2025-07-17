import streamlit as st
from src.agent.langgraph_agent import ask, visualize_graph, get_graph_mermaid_png
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Product Chatbot", page_icon="🤖")

st.title("🛒 KingArthurMix Product Chatbot")
st.write("Ask me anything about products, prices, or recipes!")

# Session state to store chat history and thinking state
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "thinking" not in st.session_state:
    st.session_state["thinking"] = False
if "example_prompt" not in st.session_state:
    st.session_state["example_prompt"] = None

# Sidebar with example questions and clear chat button
def sidebar_disabled():
    return st.session_state.get("thinking", False)

with st.sidebar:
    st.header("💡 Example Questions")
    example_questions = [
        "What is the cheapest cookie product?",
        "How many Pizza Crust Mix left?",
        "Show me gluten-free products.",
        "What is the highest rated gluten-free product?",
        "How much does the Lemon Bar Mix cost?"
    ]
    for q in example_questions:
        if st.button(q, key=f"ex_{q}", disabled=sidebar_disabled()):
            st.session_state["example_prompt"] = q
            st.rerun()
    st.markdown("---")
    st.markdown(
        """
        <style>
        div[data-testid="stSidebar"] button[kind="secondary"] {
            display: block;
            margin-left: auto;
            margin-right: auto;
        }
        /* Overlay to block sidebar when thinking */
        .sidebar-overlay-blocker {
            position: fixed;
            top: 0;
            left: 0;
            width: 350px;
            height: 100vh;
            background: rgba(255,255,255,0.6);
            z-index: 9999;
            pointer-events: all;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    if st.button("🧹 Clear Chat History", key="clear_sidebar", disabled=sidebar_disabled()):
        st.session_state["messages"] = []
        st.session_state["example_prompt"] = None
        st.rerun()
    # Overlay to block sidebar interaction when thinking
    if st.session_state.get("thinking", False):
        st.markdown('<div class="sidebar-overlay-blocker"></div>', unsafe_allow_html=True)
        st.info("🤖 The chatbot is thinking. Please wait...")

# Display chat history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Auto-scroll to bottom using JS (after new message)
st.components.v1.html(
    """
    <script>
    window.scrollTo(0, document.body.scrollHeight);
    </script>
    """,
    height=0,
)

# Determine the prompt source
prompt = None
if st.session_state["example_prompt"]:
    prompt = st.session_state["example_prompt"]
    st.session_state["example_prompt"] = None
else:
    prompt = st.chat_input("Type your question here...", disabled=st.session_state["thinking"])

if prompt and not st.session_state["thinking"]:
    st.session_state["thinking"] = True
    # Add user message to chat history and display it
    with st.chat_message("user"):
        st.markdown(prompt)
    print(f"User: {prompt}")  # Print user message to console

    # Get answer from backend with full chat history
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Pass the current chat history to the backend
            answer, updated_history = ask(prompt, st.session_state["messages"])
            st.markdown(answer)
            print(f"AI: {answer}")  # Print AI message to console
            # Update the session state with the new history from backend
            st.session_state["messages"] = updated_history
    st.session_state["thinking"] = False
    st.rerun()

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