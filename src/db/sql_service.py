import sqlite3
import os
from typing import List, Tuple
from src.config import Config

class SQLService:
    def __init__(self, db_path: str = None):
        db_path = os.environ.get('DB_PATH', db_path or Config.DB_PATH)
        abs_db_path = os.path.abspath(db_path)
        print(f"Using database at: {abs_db_path}")  # <-- This prints the absolute path
        self.db_path = abs_db_path

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