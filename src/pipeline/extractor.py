from typing import Iterator, Dict, Any
from huggingface_hub import HfApi
import logging

from schemas import RawModelData

logger = logging.getLogger(__name__)

class ModelExtractor:
    """Extract model data from HuggingFace"""
    
    def __init__(self, api_token: str):
        self.api = HfApi(token=api_token)
        
    def extract_models(self, category: str, limit: int) -> Iterator[RawModelData]:
        """Extract models by category"""
        try:
            models = self.api.list_models(
                filter=category,
                limit=limit
            )
            
            for model in models:
                try:
                    yield RawModelData(
                        model_id=model.id,
                        downloads=getattr(model, 'downloads', 0),
                        likes=getattr(model, 'likes', 0),
                        tags=getattr(model, 'tags', []),
                        last_modified=getattr(model, 'lastModified', None),
                        card_data=getattr(model, 'cardData', {})
                    )
                except Exception as e:
                    logger.error(f"Error extracting model {model.id}: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            raise
