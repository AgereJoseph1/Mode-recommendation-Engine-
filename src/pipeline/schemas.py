from pydantic import BaseModel, Field
from typing import List, Optional,Dict,Any
from datetime import datetime

class RawModelData(BaseModel):
    """Schema for raw model data"""
    model_id: str
    downloads: int = 0
    likes: int = 0
    tags: List[str] = []
    last_modified: Optional[datetime] = None
    card_data: Dict[str, Any] = {}

class TransformedModelData(BaseModel):
    """Schema for transformed model data"""
    model_id: str
    name: str
    category: str
    description: str = ""
    downloads: int
    likes: int
    popularity_score: float
    tags: List[str]
    languages: List[str]
    framework: str
    task_type: str
    size_bytes: int = 0
    license: str = ""
    metadata_embedding: List[float]
    last_modified: datetime
    processed_at: datetime