from langgraph.graph import StateGraph, END
from src.models.agent_state import AgentState
from src.nodes import (
    reasoning_node,
    reasoning_router,
    data_retrieval_node,
    knowledge_search_node,
    response_synthesis_node,
    general_chat_node
)

def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("reasoning", reasoning_node)
    graph.add_node("data_retrieval", data_retrieval_node)
    graph.add_node("knowledge_search", knowledge_search_node)
    graph.add_node("response_synthesis", response_synthesis_node)
    graph.add_node("general_chat", general_chat_node)
    graph.set_entry_point("reasoning")
    graph.add_conditional_edges(
        "reasoning",
        reasoning_router,
        {
            "data_retrieval": "data_retrieval",
            "knowledge_search": "knowledge_search",
            "response_synthesis": "response_synthesis",
            "general_chat": "general_chat",
            "end": END
        }
    )
    graph.add_edge("data_retrieval", "reasoning")
    graph.add_edge("knowledge_search", "reasoning")
    graph.add_edge("response_synthesis", "reasoning")
    graph.add_edge("general_chat", "reasoning")
    
    return graph.compile() 