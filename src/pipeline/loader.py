from typing import List, Dict, Any
from pymongo import MongoClient, DESCENDING, UpdateOne
import logging

from schemas import TransformedModelData

logger = logging.getLogger(__name__)

class ModelLoader:
    """Load transformed data into MongoDB"""
    
    def __init__(self, uri: str, database: str, collection: str):
        self.client = MongoClient(uri)
        self.db = self.client[database]
        self.collection = self.db[collection]
        
    def setup_indexes(self):
        """Create necessary indexes"""
        indexes = [
            ("model_id", DESCENDING),
            ("category", DESCENDING),
            ("popularity_score", DESCENDING),
            ("downloads", DESCENDING),
            ("processed_at", DESCENDING)
        ]
        
        for field, direction in indexes:
            self.collection.create_index([(field, direction)])
            
    def bulk_upsert(self, models: List[TransformedModelData]) -> Dict[str, Any]:
        """Bulk upsert models"""
        try:
            operations = [
                UpdateOne(
                    {"model_id": model.model_id},
                    {"$set": model.dict()},
                    upsert=True
                )
                for model in models
            ]
            
            result = self.collection.bulk_write(operations)
            
            return {
                "matched": result.matched_count,
                "modified": result.modified_count,
                "upserted": result.upserted_count
            }
            
        except Exception as e:
            logger.error(f"Error in bulk upsert: {str(e)}")
            raise