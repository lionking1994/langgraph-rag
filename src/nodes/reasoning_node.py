from . import llm, load_prompt
import json
import re

def reasoning_node(state):
    # If we already have a final answer from a previous node, skip processing
    if state.get("final_answer") and state.get("last_node") in ["response_synthesis", "general_chat"]:
        return state

    # Increment iteration count
    state["iteration_count"] = state.get("iteration_count", 0) + 1

    last_product_query = state.get("last_product_query", "")
    last_product_answer = state.get("last_product_answer", "")
    query = state["query"]
    chat_context = ""
    if last_product_query:
        chat_context = f"Previous product question: {last_product_query}\nPrevious product answer: {last_product_answer[:200]}\n"

    # Prepare data status for prompt
    has_structured_results = bool(state.get("structured_results"))
    has_semantic_results = bool(state.get("semantic_results"))
    structured_results = state.get("structured_results", "")
    semantic_results = state.get("semantic_results", "")
    prompt_template = load_prompt("intelligent_reasoning.txt")
    from langchain.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_template(prompt_template)
    prompt_str = prompt.format(
        query=query,
        reasoning_step=state.get("reasoning_step", "classify"),
        iteration_count=state["iteration_count"],
        last_node=state.get("last_node", "START"),
        needs_structured_data=state.get("needs_structured_data", False),
        structured_complete=state.get("structured_complete", False),
        needs_semantic_search=state.get("needs_semantic_search", False),
        semantic_complete=state.get("semantic_complete", False),
        has_structured_results=has_structured_results,
        has_semantic_results=has_semantic_results,
        structured_results=structured_results,
        semantic_results=semantic_results,
        chat_context=chat_context
    )
    response = llm.invoke(prompt_str).content
    response_clean = re.sub(r"^```(?:json)?|```$", "", response.strip(), flags=re.IGNORECASE | re.MULTILINE).strip()
    try:
        data = json.loads(response_clean)
        state["data_sufficiency"] = data.get("data_sufficiency", "NONE")
        state["next_action"] = data.get("next_action", "synthesize")
        state["reasoning_notes"] = data.get("reasoning_notes", "")
        state["needs_structured_data"] = data.get("needs_structured_data", state.get("needs_structured_data", False))
        state["needs_semantic_search"] = data.get("needs_semantic_search", state.get("needs_semantic_search", False))
        state["is_product_question"] = data.get("is_product_question", False)
        state["is_product_followup"] = data.get("is_product_followup", False)
        state["is_non_product"] = data.get("is_non_product", False)
        state["query_type"] = data.get("query_type", "general")
        state["reasoning_step"] = "decision"
    except Exception as e:
        print("reasoning_node JSON decode error:", e)
        state["data_sufficiency"] = "NONE"
        state["next_action"] = "synthesize"
        state["reasoning_notes"] = f"LLM response parse error: {e}"
        state["reasoning_step"] = "decision"
    state["last_node"] = "reasoning"
    return state 