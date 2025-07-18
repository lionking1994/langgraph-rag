from src.models.agent_state import AgentState
from src.agent.graph_builder import build_graph
import json  

app = build_graph()

def visualize_graph():  
    ascii_diagram = app.get_graph().draw_ascii()  
    mermaid_code = app.get_graph().draw_mermaid()  
    return ascii_diagram, mermaid_code  

def get_graph_mermaid_png():  
    try:  
        return app.get_graph().draw_mermaid_png()  
    except Exception:  
        return None  

def get_last_product_context(chat_history):
    last_query = ""
    last_answer = ""
    for i in range(len(chat_history) - 1, -1, -1):
        msg = chat_history[i]
        if msg.get("role") == "assistant":
            content = msg.get("content", "").lower()
            msg_type = msg.get("type", "").lower()
            is_product_response = (
                msg_type == "product" or 
                any(keyword in content for keyword in ["product", "price", "$", "cost", "expensive", "cheap", "item"])
            )
            if is_product_response:
                last_answer = msg.get("content", "")
                for j in range(i - 1, -1, -1):
                    if chat_history[j].get("role") == "user":
                        last_query = chat_history[j].get("content", "")
                        break
                break
    print(f"Found last product context - Query: '{last_query}', Answer: '{last_answer[:50]}...'")
    return last_query, last_answer

def ask(query: str, chat_history = None):
    if chat_history is None:
        chat_history = []
    last_product_query, last_product_answer = get_last_product_context(chat_history)
    initial_state = {
        "query": query,
        "query_type": "",
        "needs_structured_data": False,
        "needs_semantic_search": False,
        "structured_results": "",
        "semantic_results": "",
        "final_answer": "",
        "chat_history": chat_history,
        "last_product_query": last_product_query,
        "last_product_answer": last_product_answer,
        "is_product_question": False,
        "is_product_followup": False,
        "is_non_product": False,
        "last_node": "START",
        "structured_complete": False,
        "semantic_complete": False
    }
    result = app.invoke(initial_state)
    chat_history.append({"role": "user", "content": query})
    message_type = "product" if (result.get("is_product_question", False) or result.get("is_product_followup", False)) else "non-product"
    if result.get("is_non_product", False):
        message_type = "non-product"
    chat_history.append({
        "role": "assistant",
        "content": result.get("final_answer", "Sorry, I couldn't process your question."),
        "type": message_type
    })
    return result.get("final_answer", "Sorry, I couldn't process your question."), chat_history        