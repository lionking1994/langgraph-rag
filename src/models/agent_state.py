from typing import TypedDict, List, Dict

class AgentState(TypedDict):
    query: str
    query_type: str
    needs_structured_data: bool
    needs_semantic_search: bool
    structured_results: str
    semantic_results: str
    final_answer: str
    chat_history: List[Dict[str, str]]
    last_product_query: str
    last_product_answer: str
    is_product_question: bool
    is_product_followup: bool
    last_node: str  # Track the last executed node
    structured_complete: bool  # Track if structured query is done
    semantic_complete: bool    # Track if semantic search is done
    is_non_product: bool # New field for non-product classification
    # New fields for intelligent routing
    reasoning_step: str  # Current reasoning step (classify, gather_data, synthesize, etc.)
    data_sufficiency: str  # Assessment of whether we have enough data
    next_action: str  # Recommended next action
    reasoning_notes: str  # Notes from reasoning process
    iteration_count: int  # Track how many times we've been through reasoning 