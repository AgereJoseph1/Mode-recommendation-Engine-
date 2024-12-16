from typing import Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime

class ModelConfig(BaseModel):
    """Configuration for model processing"""
    min_downloads: int = Field(default=1000, description="Minimum number of downloads")
    min_likes: int = Field(default=50, description="Minimum number of likes")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", description="Model for embeddings")
    batch_size: int = Field(default=100, description="Batch size for processing")