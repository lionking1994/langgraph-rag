import sqlite3
from typing import List, Dict, Any, Tuple
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import json
import re
from textblob import TextBlob

class IntelligentQueryProcessor:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.db_path = "products.db"
        self.product_names = self.get_all_product_names()

    def get_all_product_names(self) -> List[str]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT title FROM products")
        product_names = [row[0] for row in cursor.fetchall()]
        conn.close()
        return product_names

    def analyze_intent_and_generate_sql(self, query: str) -> Dict[str, Any]:
        """Single LLM call to extract intent/entities and generate SQL."""
        product_names_str = "; ".join(self.product_names[:100])
        unified_prompt = ChatPromptTemplate.from_template(
            """
You are analyzing a customer query for a bakery product database and generating SQL.

Database Schema:
Table: products
Columns: id, title, original_price, current_price, description, ingredients, category, type, diet, baking_category, review_rating, review_count, max_quantity, related_products, bakers_also_bought, pdf_link, nutrition_facts, product_url, flavor

Available product names in database:
        {product_names_str}

Query: "{query}"

Step 1: Analyze the query and extract:
        1. Primary Intent (choose one or more):
           - count: counting products/types/categories (e.g., "how many", "count", "number of")
           - search: finding specific products
           - compare: comparing products
   - price: price-related queries
           - details: product details/ingredients/instructions
           - filter: filtering by attributes
           - aggregate: statistical queries (avg, sum, etc.)
        
        2. Entities (extract all that apply):
           - product_type: (e.g., cookie, bread, cake, muffin, scone)
           - category: (e.g., baking mix, flour, cake & pie)
   - price_range: (e.g., under $10, more than $60, cheapest, most expensive)
           - price_comparison: (e.g., "more than", "less than", "under", "over", "cheapest", "most expensive")
   - price_value: (numeric value if mentioned)
           - attributes: (e.g., gluten-free, organic)
           - specific_product: (exact product name if mentioned)
           - aggregation_type: (e.g., average, total, minimum, maximum)
   - count_target: (what is being counted)

3. Filters: Any specific conditions mentioned

Step 2: Generate SQL following these rules:
1. For ALL counting queries, even if the user says 'different types', use COUNT(id) to count the number of products, not COUNT(DISTINCT type).
2. For price queries, use current_price
3. For stock queries, use max_quantity
4. NEVER use category column for product type - only use type column
5. The type column contains stringified lists like '["Bread", "Cookies"]'
6. To match types, use LIKE '%"Type"%' with exact capitalization
7. For plurals (cookie/cookies), check both: (type LIKE '%"Cookie"%' OR type LIKE '%"Cookies"%')
8. For general searches, also check title field
9. For "cheaper/more expensive than [Product]" comparisons:
   - Use subquery: current_price < (SELECT current_price FROM products WHERE title = '[Product]')
   - Don't restrict by type unless specified
10. For "most expensive/cheapest" in a category:
    - Use ORDER BY current_price DESC/ASC
11. Limit large results to 20
12. For queries about extremes (highest, cheapest, most), return ALL products with that value:
    - Example: WHERE review_rating = (SELECT MAX(review_rating) FROM products)
13. **If searching for a specific product by its exact name (e.g., 'Pizza Crust Mix'), use WHERE title = 'Pizza Crust Mix' (not LIKE).**

Notes:
- "mix" and "mixes" are generic terms - extract core type only (e.g., "cake mix" → type: "cake")
- Always use singular form for product_type
- "what products"/"show me" = search intent, not count
- Correct minor spelling mistakes
- EVEN IF the user says 'different types', always use COUNT(id) to count the number of products, not COUNT(DISTINCT type).

EXAMPLES:

Query: "How much Pizza Crust Mix do we have in stock?"
{{
  "analysis": {{
    "intent": ["details"],
    "entities": {{
      "specific_product": "Pizza Crust Mix"
    }},
    "filters": []
  }},
  "sql": "SELECT title, max_quantity FROM products WHERE title = 'Pizza Crust Mix'"
}}

Query: "How many bread products do we have?"
{{
  "analysis": {{
    "intent": ["count"],
    "entities": {{
      "product_type": "bread",
      "count_target": "products"
    }},
    "filters": []
  }},
  "sql": "SELECT COUNT(id) FROM products WHERE (type LIKE '%\"Bread\"%' OR type LIKE '%\"Breads\"%')"
}}

Query: "How many cookie products do you have?"
{{
  "analysis": {{
    "intent": ["count"],
    "entities": {{
      "product_type": "cookie",
      "count_target": "products"
    }},
    "filters": []
  }},
  "sql": "SELECT COUNT(id) FROM products WHERE (type LIKE '%\"Cookie\"%' OR type LIKE '%\"Cookies\"%')"
}}

Query: "How many different types of cookie products do you have?"
{{
  "analysis": {{
    "intent": ["count"],
    "entities": {{
      "product_type": "cookie",
      "count_target": "products"
    }},
    "filters": []
  }},
  "sql": "SELECT COUNT(id) FROM products WHERE (type LIKE '%\"Cookie\"%' OR type LIKE '%\"Cookies\"%')"
}}

Query: "Show me all cookie products under $15"
{{
  "analysis": {{
    "intent": ["search", "filter", "price"],
    "entities": {{
      "product_type": "cookie",
      "price_comparison": "under",
      "price_value": 15
    }},
    "filters": ["price under $15"]
  }},
  "sql": "SELECT * FROM products WHERE (type LIKE '%\"Cookie\"%' OR type LIKE '%\"Cookies\"%') AND current_price < 15 LIMIT 20"
}}

Query: "What's the most expensive cake mix?"
{{
  "analysis": {{
    "intent": ["search", "price"],
    "entities": {{
      "product_type": "cake",
      "price_range": "most expensive"
    }},
    "filters": []
  }},
  "sql": "SELECT * FROM products WHERE type LIKE '%\"Cake\"%' ORDER BY current_price DESC"
}}

Query: "How many products are cheaper than Apple Cinnamon Doughnut Mix?"
{{
  "analysis": {{
    "intent": ["count", "compare", "price"],
    "entities": {{
      "specific_product": "Apple Cinnamon Doughnut Mix",
      "price_comparison": "cheaper than",
      "count_target": "products"
    }},
    "filters": ["cheaper than Apple Cinnamon Doughnut Mix"]
  }},
  "sql": "SELECT COUNT(id) FROM products WHERE current_price < (SELECT current_price FROM products WHERE title = 'Apple Cinnamon Doughnut Mix')"
}}

Query: "List all gluten-free bread products"
{{
  "analysis": {{
    "intent": ["search", "filter"],
    "entities": {{
      "product_type": "bread",
      "attributes": "gluten-free"
    }},
    "filters": ["gluten-free"]
  }},
  "sql": "SELECT * FROM products WHERE (type LIKE '%\"Bread\"%' OR type LIKE '%\"Breads\"%') AND diet LIKE '%Gluten-Free%' LIMIT 20"
}}

Query: "What products have the highest review rating?"
{{
  "analysis": {{
    "intent": ["search", "aggregate"],
    "entities": {{
      "aggregation_type": "maximum"
    }},
    "filters": ["highest review rating"]
  }},
  "sql": "SELECT * FROM products WHERE review_rating = (SELECT MAX(review_rating) FROM products)"
}}

Query: "Show me muffin mixes more expensive than $20"
{{
  "analysis": {{
    "intent": ["search", "filter", "price"],
    "entities": {{
      "product_type": "muffin",
      "price_comparison": "more than",
      "price_value": 20
    }},
    "filters": ["price more than $20"]
  }},
  "sql": "SELECT * FROM products WHERE type LIKE '%\"Muffin\"%' AND current_price > 20 LIMIT 20"
}}

Return a JSON object with this structure:
{{
  "analysis": {{
    "intent": ["list of intents"],
    "entities": {{entity_key: entity_value}},
    "filters": ["list of filters"]
  }},
  "sql": "SELECT ... FROM products WHERE ..."
}}
"""
        )
        response = self.llm.invoke(unified_prompt.format(product_names_str=product_names_str, query=query)).content
        print("analyze_intent_and_generate_sql response", response)
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
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(sql)
            results = cursor.fetchall()
            columns = [description[0] for description in cursor.description] if cursor.description else []
            return results, columns
        except Exception as e:
            print(f"SQL Error: {e}")
            print(f"Query: {sql}")
            return [], []
        finally:
            conn.close()

    def format_results(self, results: list, columns: list, query: str, intent_data: dict, total_count: int = None) -> str:
        if not results:
            return "I couldn't find any matching products or information."
        stock_keywords = ['left', 'remaining', 'in stock']
        price_keywords = ['price', 'cost', 'amount', 'how much', 'current price']
        is_stock_query = any(kw in query.lower() for kw in stock_keywords)
        is_price_query = any(kw in query.lower() for kw in price_keywords)
        # If both max_quantity and current_price are present, show both for single product queries
        if 'max_quantity' in columns and 'current_price' in columns and len(results) == 1:
            product_name = ''
            if 'title' in columns:
                product_name = results[0][columns.index('title')]
            left = results[0][columns.index('max_quantity')]
            price = results[0][columns.index('current_price')]
            lower_query = query.lower()
            # Use intent to decide what to show
            intents = []
            if intent_data:
                if 'intent' in intent_data:
                    intents = intent_data['intent']
                elif 'analysis' in intent_data and 'intent' in intent_data['analysis']:
                    intents = intent_data['analysis']['intent']
            # If both price and stock are asked
            if ('price' in intents and 'count' in intents) or (any(kw in lower_query for kw in ['stock', 'left', 'remaining', 'in stock']) and any(kw in lower_query for kw in ['price', 'cost', 'how much', 'current price'])):
                if product_name:
                    return f"There are {left} {product_name} left in stock. The price is ${price}."
                else:
                    return f"There are {left} items left in stock. The price is ${price}."
            # If only price is asked
            elif 'price' in intents or any(kw in lower_query for kw in ['price', 'cost', 'how much', 'current price']):
                if product_name:
                    return f"The price of the {product_name} is **${price}**."
                else:
                    return f"The price is **${price}**."
            # If only stock is asked
            elif 'count' in intents or any(kw in lower_query for kw in ['stock', 'left', 'remaining', 'in stock']):
                if product_name:
                    return f"There are {left} {product_name} left in stock."
                else:
                    return f"There are {left} items left in stock."
            else:
                # Default: show both
                if product_name:
                    return f"There are {left} {product_name} left in stock. The price is ${price}."
                else:
                    return f"There are {left} items left in stock. The price is ${price}."
        # If only price is present and the query is about price/cost for a single product
        if is_price_query and 'current_price' in columns and len(results) == 1:
            price = results[0][columns.index('current_price')]
            product_name = ''
            if 'title' in columns:
                product_name = results[0][columns.index('title')]
            if product_name:
                return f"The price of {product_name} is ${price}."
            else:
                return f"The price is ${price}."
        # If only stock is present and the query is about stock for a single product
        if is_stock_query and 'max_quantity' in columns and len(results) == 1:
            product_name = ''
            import re
            match = re.search(r'how many (.*?) (left|remaining|in stock)', query, re.IGNORECASE)
            if match:
                product_name = match.group(1).strip()
            left = results[0][columns.index('max_quantity')]
            if product_name:
                return f"There are {left} {product_name} left in stock."
            else:
                return f"There are {left} items left in stock."
        # If multiple products, show a Markdown list of all products with all available fields
        if len(results) > 1 and ('title' in columns or 'current_price' in columns):
            max_list = 20
            show_count = min(len(results), max_list)
            lines = []
            if show_count > 5:
                lines.append(f"Here are {show_count} products available:")
            for row in results[:max_list]:
                name = row[columns.index('title')] if 'title' in columns else ''
                line = f"- {name}" if name else "- Product"
                # --- Custom: Rating and Categories ---
                rating = None
                reviews = None
                if 'review_rating' in columns:
                    rating = row[columns.index('review_rating')]
                if 'review_count' in columns:
                    reviews = row[columns.index('review_count')]
                if rating and reviews:
                    line += f"\n    - Rating: {rating} ({reviews} reviews)"
                elif rating:
                    line += f"\n    - Rating: {rating} (not reviewed yet)"
                elif reviews:
                    line += f"\n    - Rating: not reviewed yet"
                else:
                    line += f"\n    - Rating: not reviewed yet"
                # Categories: merge category, baking_category, type
                cat_fields = []
                for cat_col in ['category', 'baking_category', 'type']:
                    if cat_col in columns:
                        val = row[columns.index(cat_col)]
                        if val:
                            try:
                                parsed = json.loads(val) if isinstance(val, str) and (val.startswith('[') or val.startswith('{')) else val
                                if isinstance(parsed, list):
                                    val = ', '.join(str(x) for x in parsed)
                                else:
                                    val = str(parsed)
                            except Exception:
                                pass
                            if val:
                                cat_fields.append(val)
                if cat_fields:
                    line += f"\n    - Categories: {', '.join(cat_fields)}"
                # --- End custom ---
                skip_cols = {'title', 'review_rating', 'review_count', 'category', 'baking_category', 'type'}
                for idx, col in enumerate(columns):
                    if col in skip_cols:
                        continue
                    value = row[idx]
                    if value is None or value == '':
                        continue
                    try:
                        parsed = json.loads(value) if isinstance(value, str) and (value.startswith('[') or value.startswith('{')) else value
                        if isinstance(parsed, list):
                            value = ', '.join(str(x) for x in parsed)
                        elif isinstance(parsed, dict):
                            value = ', '.join(f"{k}: {v}" for k, v in parsed.items())
                        else:
                            value = parsed
                    except Exception:
                        pass
                    field_name = col.replace('_', ' ').capitalize()
                    if 'url' in col and isinstance(value, str) and value.startswith('http'):
                        value = f"[{value}]({value})"
                    line += f"\n    - {field_name}: {value}"
                lines.append(line)
            return "\n".join(lines)
        # Default formatting for other queries
        formatted_data = []
        for row in results[:20]:
            row_dict = dict(zip(columns, row))
            formatted_data.append(row_dict)
        intro_text = ""
        if total_count is not None:
            intro_text = f"There are {total_count} matching products."
            if total_count > len(results):
                intro_text += f" Here are the first {len(results)}:"
            elif total_count > 0:
                intro_text += " Here they are:"
        prompt = ChatPromptTemplate.from_template("""
        Convert these database results into a natural, conversational response for the customer.
        {intro}
        Original Question: {query}
        Query Intent: {intent}
        Results:
        {results}
        Guidelines:
        1. Start with the provided introductory sentence (if any).
        2. If the "Results" list is not empty, you MUST display it as a formatted list for the user. Do not summarize it or omit it.
        3. Format product lists nicely using Markdown, including product `title` and `current_price`.
        4. Be conversational and friendly.
        5. Highlight important information like ratings, categories, and dietary info if present in the results.
        Response:
        """)
        response = self.llm.invoke(prompt.format(
            intro=intro_text,
            query=query,
            intent=intent_data.get("intent", []) if "intent" in intent_data else intent_data.get("analysis", {}).get("intent", []),
            results=json.dumps(formatted_data, indent=2)
        )).content.strip()
        return response
    
    def process_query(self, query: str, context: Dict[str, str] = None) -> str:
        corrected_query = query
        print(f"Original: {query} | Corrected: {corrected_query}")
        correction_note = ""
        if corrected_query.lower() != query.lower():
            correction_note = f"(Corrected: '{corrected_query}')\n"
        enhanced_query = corrected_query
        if context and context.get("last_product_query"):
            reference_words = ["it", "this", "that", "them", "these", "those"]
            if any(word in corrected_query.lower() for word in reference_words):
                enhanced_query = f"{corrected_query} (referring to: {context['last_product_query']})"
        intent_sql_data = self.analyze_intent_and_generate_sql(enhanced_query)
        sql = intent_sql_data.get("sql", "")
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
            if context:
                response = self.format_results(results, columns, corrected_query, intent_sql_data, total_count=total_count)
            else:
                response = self.format_results(results, columns, corrected_query, intent_sql_data, total_count=total_count)
        elif sql:
            results, columns = self.execute_query(sql)
            if context:
                response = self.format_results(results, columns, corrected_query, intent_sql_data, total_count=total_count)
            else:
                response = self.format_results(results, columns, corrected_query, intent_sql_data, total_count=total_count)
        else:
            response = "I couldn't find any matching products or information."
        return correction_note + response