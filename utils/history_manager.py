"""
User history management module for storing and retrieving query history.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path

class HistoryManager:
    """Manages user query history using SQLite database."""
    
    def __init__(self, db_path="user_history.db"):
        """
        Initialize history manager with database path.
        
        Parameters:
            db_path (str): Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._init_db()
    
    def _init_db(self):
        """Initialize database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    query_text TEXT,
                    response_text TEXT,
                    entities TEXT
                )
            ''')
            conn.commit()
    
    def save_query(self, user_id, query_text, response_text, entities):
        """
        Save user query and response to history.
        
        Parameters:
            user_id (str): Unique identifier for user
            query_text (str): Original user query
            response_text (str): Generated response
            entities (dict): Extracted entities from query
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO user_queries (user_id, query_text, response_text, entities)
                VALUES (?, ?, ?, ?)
            ''', (user_id, query_text, response_text, json.dumps(entities)))
            conn.commit()
    
    def get_user_history(self, user_id, limit=10):
        """
        Retrieve query history for a specific user.
        
        Parameters:
            user_id (str): Unique identifier for user
            limit (int): Maximum number of history items to return
            
        Returns:
            list: User query history
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT timestamp, query_text, response_text 
                FROM user_queries 
                WHERE user_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (user_id, limit))
            
            results = cursor.fetchall()
            return [
                {
                    'timestamp': row[0],
                    'query': row[1],
                    'response': row[2]
                }
                for row in results
            ]
    
    def clear_user_history(self, user_id):
        """
        Clear history for a specific user.
        
        Parameters:
            user_id (str): Unique identifier for user
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    DELETE FROM user_queries 
                    WHERE user_id = ?
                ''', (user_id,))
                conn.commit()
                
                # Check if any rows were affected
                if cursor.rowcount > 0:
                    return True
                else:
                    return False
        except Exception as e:
            print(f"Error clearing history: {e}")
            return False
    
    def reset_all_history(self):
        """
        Reset all user history (complete database wipe).
        Use with caution!
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    DELETE FROM user_queries
                ''')
                conn.commit()
                return True
        except Exception as e:
            print(f"Error resetting all history: {e}")
            return False
    
    def get_history_stats(self, user_id=None):
        """
        Get statistics about the history database.
        
        Parameters:
            user_id (str): Optional user ID to get stats for specific user
            
        Returns:
            dict: Statistics about the history
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if user_id:
                    # Get stats for specific user
                    cursor.execute('''
                        SELECT COUNT(*) as total_queries,
                               MIN(timestamp) as first_query,
                               MAX(timestamp) as last_query
                        FROM user_queries 
                        WHERE user_id = ?
                    ''', (user_id,))
                else:
                    # Get overall stats
                    cursor.execute('''
                        SELECT COUNT(*) as total_queries,
                               COUNT(DISTINCT user_id) as unique_users,
                               MIN(timestamp) as first_query,
                               MAX(timestamp) as last_query
                        FROM user_queries
                    ''')
                
                result = cursor.fetchone()
                columns = [description[0] for description in cursor.description]
                
                return dict(zip(columns, result))
                
        except Exception as e:
            print(f"Error getting history stats: {e}")
            return {}