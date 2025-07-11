import json
import sqlite3
import re
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# Load and process JSON data
with open("products.json", "r") as f:
    raw_data = json.load(f)

# Find all unique keys in products.json
all_keys = set()
for product in raw_data:
    all_keys.update(product.keys())

# List of columns for the products table
columns = list(all_keys)

# SQLite type mapping (default to TEXT, but use INTEGER/REAL for known fields)
type_map = {
    "max_quantity": "INTEGER",
    "review_count": "INTEGER",
    "review_rating": "REAL",
    "original_price": "REAL",
    "current_price": "REAL",
    # Add more if you want to force types
}

def get_sql_type(key):
    return type_map.get(key, "TEXT")

# Create SQLite database with dynamic schema
conn = sqlite3.connect('products.db')
cursor = conn.cursor()

# Drop existing table
cursor.execute('DROP TABLE IF EXISTS products')

# Build CREATE TABLE statement dynamically
col_defs = []
for col in columns:
    sql_type = get_sql_type(col)
    col_defs.append(f'"{col}" {sql_type}')
col_defs_str = ',\n    '.join(col_defs)
create_table_sql = f'''
CREATE TABLE products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    {col_defs_str}
)
'''
cursor.execute(create_table_sql)

# Create indexes for better performance
if "title" in columns:
    cursor.execute('CREATE INDEX idx_products_title ON products(title)')
if "category" in columns:
    cursor.execute('CREATE INDEX idx_products_category ON products(category)')
if "type" in columns:
    cursor.execute('CREATE INDEX idx_products_type ON products(type)')
if "current_price" in columns:
    cursor.execute('CREATE INDEX idx_products_current_price ON products(current_price)')
if "original_price" in columns:
    cursor.execute('CREATE INDEX idx_products_original_price ON products(original_price)')
if "review_rating" in columns:
    cursor.execute('CREATE INDEX idx_products_review_rating ON products(review_rating)')
if "diet" in columns:
    cursor.execute('CREATE INDEX idx_products_diet ON products(diet)')
if "baking_category" in columns:
    cursor.execute('CREATE INDEX idx_products_baking_category ON products(baking_category)')

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

all_texts = []
all_metadatas = []

for idx, product in enumerate(raw_data):
    # Prepare values for insertion
    values = []
    for col in columns:
        value = product.get(col, None)
        # Convert lists/dicts to JSON strings for storage
        if isinstance(value, (list, dict)):
            value = json.dumps(value)
        # Convert empty string to None for nullable fields
        if value == "":
            value = None
        # Try to coerce types for known fields
        if col == "review_count" and value is not None:
            try:
                value = int(value)
            except Exception:
                value = None
        if col == "review_rating" and value is not None:
            try:
                value = float(value)
            except Exception:
                value = None
        if col == "max_quantity" and value is not None:
            try:
                value = int(value)
            except Exception:
                value = None
        # Handle price fields - convert string prices to float
        if col in ["original_price", "current_price"] and value is not None:
            try:
                # Remove currency symbols and convert to float
                if isinstance(value, str):
                    value = re.sub(r'[^\d.]', '', value)
                value = float(value) if value else None
            except Exception:
                value = None
        values.append(value)

    # Insert into products table
    placeholders = ', '.join(['?'] * len(columns))
    column_names = ', '.join([f'"{col}"' for col in columns])
    insert_sql = f'INSERT INTO products ({column_names}) VALUES ({placeholders})'
    cursor.execute(insert_sql, values)
    product_id = cursor.lastrowid

    # Create comprehensive text for vector store
    # Use a selection of fields for the text chunk
    comprehensive_text = f"""
    Product: {product.get('title', '')}
    Current Price: ${product.get('current_price', '')}
    Original Price: ${product.get('original_price', '')}
    Category: {product.get('category', '')}
    Type: {product.get('type', '')}
    Baking Category: {product.get('baking_category', '')}
    Diet: {product.get('diet', '')}
    Description: {product.get('description', '')}
    Ingredients: {product.get('ingredients', '')}
    Review Rating: {product.get('review_rating', '')}
    Review Count: {product.get('review_count', '')}
    """
    chunks = text_splitter.split_text(comprehensive_text)
    for chunk in chunks:
        all_texts.append(chunk)
        all_metadatas.append({
            "product_id": product_id,
            "title": product.get('title', ''),
            "category": product.get('category', ''),
            "type": product.get('type', ''),
            "current_price": product.get('current_price', ''),
            "original_price": product.get('original_price', ''),
            "diet": product.get('diet', ''),
            "baking_category": product.get('baking_category', ''),
            "review_rating": product.get('review_rating', ''),
            "review_count": product.get('review_count', '')
        })

conn.commit()
conn.close()

# Create enhanced vector store
vectorstore = FAISS.from_texts(
    texts=all_texts, 
    embedding=embeddings, 
    metadatas=all_metadatas
)
vectorstore.save_local("faiss_mix")

print("Database and vector store created with columns matching products.json!")
print(f"Total products processed: {len(raw_data)}")
print("Price-related indexes created for current_price and original_price fields")