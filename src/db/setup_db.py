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
    "origin_price": "REAL",
    "price": "REAL",
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
if "name" in columns:
    cursor.execute('CREATE INDEX idx_products_name ON products(name)')
if "category" in columns:
    cursor.execute('CREATE INDEX idx_products_category ON products(category)')
if "type" in columns:
    cursor.execute('CREATE INDEX idx_products_type ON products(type)')
if "price" in columns:
    cursor.execute('CREATE INDEX idx_products_price ON products(price)')
if "origin_price" in columns:
    cursor.execute('CREATE INDEX idx_products_origin_price ON products(origin_price)')
if "review_rating" in columns:
    cursor.execute('CREATE INDEX idx_products_review_rating ON products(review_rating)')
if "diet" in columns:
    cursor.execute('CREATE INDEX idx_products_diet ON products(diet)')
if "baking_category" in columns:
    cursor.execute('CREATE INDEX idx_products_baking_category ON products(baking_category)')
if "url" in columns:
    cursor.execute('CREATE INDEX idx_products_url ON products(url)')
if "flag" in columns:
    cursor.execute('CREATE INDEX idx_products_flag ON products(flag)')
if "discount_multiple_buy" in columns:
    cursor.execute('CREATE INDEX idx_products_discount_multiple_buy ON products(discount_multiple_buy)')

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
        if col in ["origin_price", "price"] and value is not None:
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
    Product: {product.get('name', '')}
    Price: ${product.get('price', '')}
    Original Price: ${product.get('origin_price', '')}
    Category: {product.get('category', '')}
    Type: {product.get('type', '')}
    Baking Category: {product.get('baking_category', '')}
    Diet: {product.get('diet', '')}
    Description: {product.get('description', '')}
    Ingredients: {product.get('ingredients', '')}
    Review Rating: {product.get('review_rating', '')}
    Review Count: {product.get('review_count', '')}
    URL: {product.get('url', '')}
    """
    chunks = text_splitter.split_text(comprehensive_text)
    for chunk in chunks:
        all_texts.append(chunk)
        all_metadatas.append({
            "product_id": product_id,
            "name": product.get('name', ''),
            "category": product.get('category', ''),
            "type": product.get('type', ''),
            "price": product.get('price', ''),
            "origin_price": product.get('origin_price', ''),
            "diet": product.get('diet', ''),
            "baking_category": product.get('baking_category', ''),
            "review_rating": product.get('review_rating', ''),
            "review_count": product.get('review_count', ''),
            "url": product.get('url', '')
        })

    # Add related products to vector store
    related_products = product.get('related_products', [])
    if isinstance(related_products, str):
        try:
            related_products = json.loads(related_products)
        except Exception:
            related_products = []
    if isinstance(related_products, list):
        for related_name in related_products:
            if not related_name:
                continue
            # Try to find the related product in the dataset
            related_info = next((p for p in raw_data if p.get('name', '') == related_name), None)
            if related_info:
                related_text = f"""
                Product: {product.get('name', '')}
                Related Product: {related_name}
                Related Product Details:
                Price: ${related_info.get('price', '')}
                Original Price: ${related_info.get('origin_price', '')}
                Category: {related_info.get('category', '')}
                Type: {related_info.get('type', '')}
                Description: {related_info.get('description', '')}
                URL: {related_info.get('url', '')}
                """
            else:
                related_text = f"Product: {product.get('name', '')} is related to: {related_name}"
            related_chunks = text_splitter.split_text(related_text)
            for chunk in related_chunks:
                all_texts.append(chunk)
                all_metadatas.append({
                    "product_id": product_id,
                    "name": product.get('name', ''),
                    "related_product": related_name,
                    "relation_type": "related_product"
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
print("Price-related indexes created for price and origin_price fields")