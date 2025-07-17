from . import vectorstore

def knowledge_search_node(state):
    if state.get("needs_semantic_search", False):
        search_query = state["query"]
        if state.get("is_product_followup") and state.get("last_product_query"):
            reference_words = ["it", "this", "that", "them", "these", "those"]
            if any(word in search_query.lower() for word in reference_words) or len(search_query.split()) <= 5:
                search_query = f"{state['last_product_query']} {search_query}"
        docs = vectorstore.similarity_search(search_query, k=5)
        context = "\n---\n".join([
            f"Product: {doc.metadata.get('title', 'Unknown')}\n{doc.page_content}"
            for doc in docs
        ])
        state["semantic_results"] = context
    state["semantic_complete"] = True
    state["last_node"] = "knowledge_search"
    return state 