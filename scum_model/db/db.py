import pymongo
from typing import Dict, List, Any, Optional, Union


class MongoDBManager:
    """
    A class to manage MongoDB connections and operations.
    
    This class provides methods for common database operations like creating,
    reading, updating, and deleting documents in MongoDB collections.
    """
    def __init__(self, host: str = 'localhost', port: int = 27017, 
                 database: str = None, username: str = None, 
                 password: str = None, connection_string: str = None):
        self.client = None
        self.db = None
        
        try:
            # Connect using connection string if provided
            if connection_string:
                self.client = pymongo.MongoClient(connection_string)
            # Otherwise connect using individual parameters
            else:
                if username and password:
                    self.client = pymongo.MongoClient(
                        host=host,
                        port=port,
                        username=username,
                        password=password
                    )
                else:
                    self.client = pymongo.MongoClient(host=host, port=port)
            
            # Select database if provided
            if database:
                self.db = self.client[database]
                
            # Test connection
            self.client.admin.command('ping')
            print("MongoDB connection established successfully.")
            
        except Exception as e:
            print(f"Failed to connect to MongoDB: {e}")
            raise
    
    def select_database(self, database_name: str) -> None:
        self.db = self.client[database_name]
    
    def list_databases(self) -> List[str]:
        return self.client.list_database_names()
    
    def list_collections(self) -> List[str]:
        if self.db is None:
            raise ValueError("No database selected. Call select_database first.")
        return self.db.list_collection_names()
    
    def insert_one(self, collection: str, document: Dict[str, Any]) -> str:
        if self.db is None:
            raise ValueError("No database selected. Call select_database first.")
        
        # If '_id' is already in the document, MongoDB will use that instead of generating one
        result = self.db[collection].insert_one(document)
        return str(result.inserted_id)

    def insert_many(self, collection: str, documents: List[Dict[str, Any]]) -> List[str]:
        if self.db is None:
            raise ValueError("No database selected. Call select_database first.")
        
        # Each document can have its own '_id' field
        result = self.db[collection].insert_many(documents)
        return [str(id) for id in result.inserted_ids]
    
    def find_one(self, collection: str, query: Dict[str, Any] = None, 
                projection: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        if self.db is None:
            raise ValueError("No database selected. Call select_database first.")
        
        return self.db[collection].find_one(query, projection)
    
    def find_many(self, collection: str, query: Dict[str, Any] = None, 
                 projection: Dict[str, Any] = None, 
                 sort: List[tuple] = None, 
                 limit: int = 0, 
                 skip: int = 0) -> List[Dict[str, Any]]:
        if self.db is None:
            raise ValueError("No database selected. Call select_database first.")
        
        cursor = self.db[collection].find(query, projection)
        
        if sort:
            cursor = cursor.sort(sort)
        
        if skip:
            cursor = cursor.skip(skip)
        
        if limit:
            cursor = cursor.limit(limit)
        
        return list(cursor)
    
    def update_one(self, collection: str, query: Dict[str, Any], 
                  update: Dict[str, Any], upsert: bool = False) -> int:
        if self.db is None:
            raise ValueError("No database selected. Call select_database first.")
        
        result = self.db[collection].update_one(query, update, upsert=upsert)
        return result.modified_count
    
    def update_many(self, collection: str, query: Dict[str, Any], 
                   update: Dict[str, Any], upsert: bool = False) -> int:
        if self.db is None:
            raise ValueError("No database selected. Call select_database first.")
        
        result = self.db[collection].update_many(query, update, upsert=upsert)
        return result.modified_count
    
    def delete_one(self, collection: str, query: Dict[str, Any]) -> int:
        if self.db is None:
            raise ValueError("No database selected. Call select_database first.")
        
        result = self.db[collection].delete_one(query)
        return result.deleted_count
    
    def delete_many(self, collection: str, query: Dict[str, Any]) -> int:
        if self.db is None:
            raise ValueError("No database selected. Call select_database first.")
        
        result = self.db[collection].delete_many(query)
        return result.deleted_count
    
    def count_documents(self, collection: str, query: Dict[str, Any] = None) -> int:
        if self.db is None:
            raise ValueError("No database selected. Call select_database first.")
        
        return self.db[collection].count_documents(query or {})
    
    def aggregate(self, collection: str, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.db is None:
            raise ValueError("No database selected. Call select_database first.")
        
        return list(self.db[collection].aggregate(pipeline))
    
    def create_index(self, collection: str, keys: Union[str, List[tuple]], 
                    unique: bool = False, sparse: bool = False) -> str:
        if self.db is None:
            raise ValueError("No database selected. Call select_database first.")
        
        if isinstance(keys, str):
            keys = [(keys, pymongo.ASCENDING)]
            
        return self.db[collection].create_index(keys, unique=unique, sparse=sparse)
    
    def drop_index(self, collection: str, index_name: str) -> None:
        if self.db is None:
            raise ValueError("No database selected. Call select_database first.")
        
        self.db[collection].drop_index(index_name)
    
    def drop_collection(self, collection: str) -> None:
        if self.db is None:
            raise ValueError("No database selected. Call select_database first.")
        
        self.db.drop_collection(collection)
    
    def close(self) -> None:
        if self.client:
            self.client.close()
            print("MongoDB connection closed.")