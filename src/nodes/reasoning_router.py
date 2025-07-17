def reasoning_router(state):
    # If we already have a final answer from synthesize or non_product, we should end
    if state.get("final_answer") and state.get("last_node") in ["response_synthesis", "general_chat"]:
        return "end"

    # Use the new next_action field for routing
    next_action = state.get("next_action", "synthesize")
    action_map = {
        "gather_structured": "data_retrieval",
        "gather_semantic": "knowledge_search",
        "synthesize": "response_synthesis",
        "general_chat": "general_chat",
        "end": "end"
    }
    return action_map.get(next_action, "response_synthesis") 