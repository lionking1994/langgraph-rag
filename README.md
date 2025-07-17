---
title: "KingArthurBaking Product Chatbot"
emoji: "🛒"
colorFrom: "indigo"
colorTo: "blue"
sdk: "streamlit"
sdk_version: "1.32.0"
app_file: app.py
pinned: false
---

# 🛒 KingArthurBaking Product Chatbot

A conversational AI assistant for King Arthur Baking products, built with Streamlit, LangGraph, OpenAI, and SQLite. Ask about products, prices, recipes, stock, and more!

---

## 🚀 Features
- Natural language product search and Q&A
- Price, stock, and ingredient queries
- Handles follow-up and context-aware questions
- Visualizes reasoning flow with LangGraph
- Uses OpenAI LLMs for reasoning and SQL generation
- Fast, local vector search with FAISS
- Easy-to-use Streamlit web interface

---

## 🗂️ Project Structure
```
langgraph-rag/
  app.py                  # Streamlit UI
  src/
    agent/
      graph_builder.py    # LangGraph orchestration and graph logic
      langgraph_agent.py  # LangGraph agent and routing
    config.py             # Configuration
    db/
      products.db         # SQLite database
      products.json       # Product data
      setup_db.py         # Database setup script
      sql_service.py      # SQL query logic
      faiss_mix/
        index.faiss       # FAISS vector index
        index.pkl         # FAISS metadata
    models/
      agent_state.py      # Agent state dataclass
    nodes/                # LangGraph node implementations
      data_retrieval_node.py
      general_chat_node.py
      knowledge_search_node.py
      reasoning_node.py
      reasoning_router.py
      response_synthesis_node.py
      __init__.py
    prompts/              # Prompt templates
      ...
    tools/
      query_processor.py  # SQL, DB, and product query logic
    utils/
      prompt_service.py   # Prompt loading utilities
      result_formatter.py # Output formatting
  requirements.txt        # Python dependencies
  README.md               # This file
  Dockerfile              # (Optional) Containerization
```

---

## ⚙️ Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd langgraph-rag
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   # On Unix/macOS:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   - Create a `.env` file with your OpenAI API key:
     ```
     OPENAI_API_KEY=sk-...
     ```

5. **Prepare the database and vector store**
   ```bash
   python src/db/setup_db.py
   ```

6. **Run the app**
   ```bash
   streamlit run app.py
   ```
   - Open your browser to the displayed URL (e.g., http://localhost:8501)

---

## 💬 Example Queries
- What gluten-free products do you have?
- How many different types of cookie products are there?
- Which products cost more than $10?
- How many Pizza Crust Mix left?
- What are the ingredients in the Gluten-Free Confetti Cake Mix?
- Which is more expensive, Gluten-Free Pancake Mix or Gluten-Free Muffin Mix?
- Show me all cake mixes.

---

## 🧪 Testing
- Try a variety of queries (see above) and follow-up questions.
- Use the "Show LangGraph Flow Visualization" expander to see the reasoning flow.
- For database debugging, use `sqlite3 products.db` or a GUI like Navicat.

---

## 🛠️ Troubleshooting
- **No answer shown?** Check the backend logs and ensure the answer is being returned from the agent.
- **LLM errors?** Make sure your OpenAI API key is valid and you have internet access.
- **Database issues?** Re-run `python setup_db.py` to rebuild the database from `products.json`.
- **Dependency errors?** Double-check your `requirements.txt` and Python version.

---

## 📞 Support / Contact
For questions or support, open an issue or contact the project maintainer.

---

Enjoy chatting with your baking products! 🥖🍰🤖 