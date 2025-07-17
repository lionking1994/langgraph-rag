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
chatbot/
  app.py                # Streamlit UI
  langgraph_agent.py    # LangGraph orchestration and routing
  query_processor.py    # SQL, DB, and product query logic
  products.db           # SQLite database
  products.json         # Product data
  faiss_mix/            # FAISS vector store
  requirements.txt      # Python dependencies
  README.md             # This file
```

---

## ⚙️ Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd chatbot
   ```

2. **Create and activate a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
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
   python setup_db.py
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