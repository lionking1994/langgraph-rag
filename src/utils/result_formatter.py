import json
from langchain.prompts import ChatPromptTemplate
from src.utils.prompt_service import PromptService
import re

class ResultFormatter:
    def __init__(self, prompt_service=None):
        self.prompt_service = prompt_service or PromptService()

    def format_results(self, results: list, columns: list, query: str, intent_data: dict, total_count: int = None, llm=None, context_info: str = "") -> str:
        """
        Formats SQL query results using the unified format_results.txt prompt, which now includes all rules for single product, list, count, and follow-up queries.
        """
        if not results:
            return "I couldn't find any matching products or information."
        if llm is None:
            raise ValueError("LLM instance must be provided for formatting results.")
        formatted_data_list = [dict(zip(columns, row)) for row in results[:20]]
        formatted_data = json.dumps(formatted_data_list, indent=2)

        intent = intent_data.get("intent", []) if "intent" in intent_data else intent_data.get("analysis", {}).get("intent", [])
        prompt_template = self.prompt_service.get_prompt("format_results.txt")
        prompt = ChatPromptTemplate.from_template(prompt_template)
        response = llm.invoke(prompt.format(
            context_info=context_info,
            query=query,
            raw_answer=formatted_data,
            intent=intent,
            total_count=total_count
        )).content.strip()
        # print(f"[DEBUG] Formatter LLM response: {response}")
        return response 