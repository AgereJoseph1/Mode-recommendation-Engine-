# src/pipeline/transformers.py
from typing import Dict, Any, List
import pandas as pd
from sentence_transformers import SentenceTransformer
  # Increase timeout to 30 seconds

from schemas import RawModelData

class ModelTransformer:
    """Transform raw model data into processed format"""
    
    def __init__(self, embedding_model: str):
        self.embedding_model = SentenceTransformer(embedding_model, model_kwargs={'timeout': 30})
        
    def transform_metadata(self, data: RawModelData) -> Dict[str, Any]:
        """Transform and clean metadata"""
        card_data = data.card_data or {}
        
        return {
            'name': card_data.get('model-name', data.model_id),
            'description': card_data.get('description', '').strip(),
            'languages': card_data.get('language', []),
            'framework': self._detect_framework(data.tags),
            'license': card_data.get('license', ''),
            'task_type': card_data.get('task', 'unknown')
        }
        
    def _detect_framework(self, tags: List[str]) -> str:
        """Detect model framework from tags"""
        framework_mapping = {
            'pytorch': ['pytorch', 'torch'],
            'tensorflow': ['tensorflow', 'tf'],
            'jax': ['jax', 'flax']
        }
        
        for framework, keywords in framework_mapping.items():
            if any(keyword in tag.lower() for tag in tags for keyword in keywords):
                return framework
        return 'unknown'
        
    def calculate_popularity_score(self, downloads: int, likes: int) -> float:
        """Calculate normalized popularity score"""
        download_weight = 0.7
        likes_weight = 0.3
        
        # Normalize values (using log scale for downloads)
        if downloads > 0:
            normalized_downloads = np.log10(downloads) / 10  # Assuming max log10 value of 10
        else:
            normalized_downloads = 0
            
        normalized_likes = min(likes / 1000, 1)  # Cap at 1000 likes
        
        return (download_weight * normalized_downloads + 
                likes_weight * normalized_likes)
                
    def create_metadata_embedding(self, data: Dict[str, Any]) -> List[float]:
        """Create embedding from metadata"""
        metadata_text = f"{data['name']} {data['description']} {' '.join(data['tags'])} {data['task_type']}"
        return self.embedding_model.encode(metadata_text).tolist()