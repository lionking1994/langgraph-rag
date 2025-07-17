import os
from dotenv import load_dotenv
from src.config import Config
load_dotenv(dotenv_path=Config.DOTENV_PATH)

from src.utils.prompt_service import PromptService
from src.db.sql_service import SQLService
from src.utils.result_formatter import ResultFormatter
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import json
import re
from typing import List, Dict, Any, Tuple

from src.config import Config

class IntelligentQueryProcessor:
    def __init__(self, db_path=None, llm=None, prompt_service=None, sql_service=None, result_formatter=None):
        self.llm = llm or ChatOpenAI(model="gpt-4o", temperature=0)
        self.prompt_service = prompt_service or PromptService()
        self.db_path = db_path or Config.DB_PATH
        self.sql_service = sql_service or SQLService(db_path or Config.DB_PATH)
        self.result_formatter = result_formatter or ResultFormatter()
        self.product_names = self.get_all_product_names()

    def get_all_product_names(self) -> List[str]:
        sql = "SELECT name FROM products"
        results, _ = self.sql_service.execute_query(sql)
        return [row[0] for row in results]

    def analyze_intent_and_generate_sql(self, query: str) -> Dict[str, Any]:
        product_names_str = "; ".join(self.product_names[:100])
        prompt_template = self.prompt_service.get_prompt("analyze_intent_and_generate_sql.txt")
        prompt = ChatPromptTemplate.from_template(prompt_template)
        response = self.llm.invoke(prompt.format(product_names_str=product_names_str, query=query)).content
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {
                    "analysis": {
                    "intent": ["search"],
                    "entities": {"product_type": query},
                        "filters": []
                    },
                    "sql": ""
                }
        except Exception as e:
            print("Error parsing LLM response:", e)
            return {
                "analysis": {
                "intent": ["search"],
                "entities": {"product_type": query},
                    "filters": []
                },
                "sql": ""
            }

    def execute_query(self, sql: str) -> Tuple[List[Tuple], List[str]]:
        print(f"[DEBUG] Executing SQL: {sql}")
        results, columns = self.sql_service.execute_query(sql)
        print(f"[DEBUG] SQL Results: {results}")
        print(f"[DEBUG] SQL Columns: {columns}")
        return results, columns

    def is_count_query(self, sql: str) -> bool:
        """Check if the SQL query is a COUNT query."""
        sql_lower = sql.strip().lower()
        return sql_lower.startswith('select count(') and 'from' in sql_lower

    def convert_count_to_select_all(self, count_sql: str) -> str:
        """Convert a COUNT query to a SELECT * query with the same WHERE clause."""
        sql_lower = count_sql.strip().lower()
        
        # Extract the FROM clause and everything after it
        if ' from ' in sql_lower:
            from_index = sql_lower.find(' from ')
            from_clause_and_rest = count_sql[from_index:]
            return f"SELECT * {from_clause_and_rest}"
        
        return count_sql

    def format_results(self, results: list, columns: list, query: str, intent_data: dict, total_count: int = None, context_info: str = "") -> str:
        """
        Formats SQL query results using the unified format_results.txt prompt, which now includes all rules for single product, list, count, and follow-up queries.
        """
        return self.result_formatter.format_results(results, columns, query, intent_data, total_count=total_count, llm=self.llm, context_info=context_info)

    def process_query(self, query: str, context: Dict[str, str] = None) -> str:
        corrected_query = query
        print(f"Original: {query} | Corrected: {corrected_query}")
        correction_note = ""
        if corrected_query.lower() != query.lower():
            correction_note = f"(Corrected: '{corrected_query}')\n"
        
        # Enhanced query processing for follow-up questions
        enhanced_query = corrected_query
        specific_product = None
        context_info = ""
        
        if context and context.get("last_product_answer"):
            last_answer = context.get("last_product_answer", "")
            if "**" in last_answer:
                product_matches = re.findall(r'\*\*([^*]+)\*\*: \*\*\\$', last_answer)
                if product_matches:
                    specific_product = product_matches[0].strip()
                    print(f"Extracted specific product: {specific_product}")
        
        if context and context.get("last_product_query"):
            reference_words = ["it", "this", "that", "them", "these", "those"]
            if any(word in corrected_query.lower() for word in reference_words):
                if specific_product:
                    enhanced_query = f"{corrected_query} (referring to specific product: {specific_product})"
                else:
                    enhanced_query = f"{corrected_query} (referring to: {context['last_product_query']})"
            context_info = f"Previous question: {context['last_product_query']}\nPrevious answer: {context.get('last_product_answer', '')}\n"
            if specific_product:
                context_info += f"Specific product being referenced: {specific_product}\n"
        
        intent_sql_data = self.analyze_intent_and_generate_sql(enhanced_query)
        sql = intent_sql_data.get("sql", "")
        print(f"Generated SQL: {sql}")
        
        # Check if this is a count query
        if self.is_count_query(sql):
            print(f"[DEBUG] Detected COUNT query: {sql}")
            count_results, count_columns = self.execute_query(sql)
            
            if count_results and len(count_results) > 0 and count_results[0][0] == 1:
                print(f"[DEBUG] COUNT query returned 1 result, converting to SELECT *")
                # Convert count query to select all query
                select_all_sql = self.convert_count_to_select_all(sql)
                print(f"[DEBUG] Converted to SELECT * query: {select_all_sql}")
                
                # Execute the SELECT * query
                results, columns = self.execute_query(select_all_sql)
                response = self.format_results(results, columns, corrected_query, intent_sql_data, total_count=1, context_info=context_info)
            else:
                # Execute the original count query
                results, columns = count_results, count_columns
                response = self.format_results(results, columns, corrected_query, intent_sql_data, total_count=None, context_info=context_info)
        else:
            # Original logic for non-count queries
            is_list_request = 'search' in intent_sql_data.get('analysis', {}).get('intent', []) and 'count' not in intent_sql_data.get('analysis', {}).get('intent', [])
            total_count = None
            results, columns = [], []
            if is_list_request and sql:
                list_sql = sql
                count_sql = ""
                if " from " in list_sql.lower():
                    from_part = list_sql.lower().split(" from ", 1)[1]
                    if " order by " in from_part:
                        from_part = from_part.split(" order by ", 1)[0]
                    if " limit " in from_part:
                        from_part = from_part.split(" limit ", 1)[0]
                    count_sql = f"SELECT COUNT(id) FROM {from_part}"
                if count_sql:
                    count_results, _ = self.execute_query(count_sql)
                    if count_results and count_results[0]:
                        total_count = count_results[0][0]
                results, columns = self.execute_query(list_sql)
                response = self.format_results(results, columns, corrected_query, intent_sql_data, total_count=total_count, context_info=context_info)
            elif sql:
                results, columns = self.execute_query(sql)
                response = self.format_results(results, columns, corrected_query, intent_sql_data, total_count=total_count, context_info=context_info)
            else:
                response = "I couldn't find any matching products or information."
        
        print(f"[DEBUG] Final formatted response: {response}")
        return response 