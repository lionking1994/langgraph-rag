from . import query_processor

def data_retrieval_node(state):
    if state.get("needs_structured_data", False):
        context = {}
        query_to_use = state["query"]
        if state.get("is_product_followup") and state.get("last_product_query"):
            context = {
                "last_product_query": state.get("last_product_query", ""),
                "last_product_answer": state.get("last_product_answer", "")
            }
            reference_words = ["it", "this", "that", "them", "these", "those"]
            if any(word in query_to_use.lower() for word in reference_words) or len(query_to_use.split()) <= 5:
                query_to_use = f"{state['last_product_query']} {query_to_use}"
        result = query_processor.process_query(query_to_use, context)
        state["structured_results"] = result
    state["structured_complete"] = True
    state["last_node"] = "data_retrieval"
    return state 