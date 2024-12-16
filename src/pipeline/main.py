import os
from typing import List, Dict, Any
import logging
from datetime import datetime
from venv import logger

from dotenv import load_dotenv
from config import ModelConfig
from extractor import ModelExtractor
from custom_transformers import ModelTransformer
from loader import ModelLoader
from monitor import PipelineMonitor

class ModelPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.monitor = PipelineMonitor()
        
        # Initialize components
        self.extractor = ModelExtractor(os.getenv('HUGGINGFACE_TOKEN'))
        self.transformer = ModelTransformer(config.embedding_model)
        self.loader = ModelLoader(
            uri=os.getenv('MONGODB_URI'),
            database=os.getenv('MONGODB_DATABASE'),
            collection=os.getenv('MONGODB_COLLECTION')
        )
        
        # Setup database
        self.loader.setup_indexes()
        
    def process_category(self, category: str) -> Dict[str, Any]:
        """Process models for a category"""
        try:
            transformed_models = []
            
            # Extract models
            for raw_model in self.extractor.extract_models(
                category, 
                self.config.batch_size
            ):
                try:
                    # Skip if doesn't meet quality criteria
                    if (raw_model.downloads < self.config.min_downloads or
                        raw_model.likes < self.config.min_likes):
                        continue
                    
                    # Transform metadata
                    metadata = self.transformer.transform_metadata(raw_model)
                    
                    # Create model document
                    transformed = TransformedModelData(
                        model_id=raw_model.model_id,
                        category=category,
                        downloads=raw_model.downloads,
                        likes=raw_model.likes,
                        popularity_score=self.transformer.calculate_popularity_score(
                            raw_model.downloads, 
                            raw_model.likes
                        ),
                        tags=raw_model.tags,
                        last_modified=raw_model.last_modified or datetime.now(),
                        processed_at=datetime.now(),
                        metadata_embedding=self.transformer.create_metadata_embedding(metadata),
                        **metadata
                    )
                    
                    transformed_models.append(transformed)
                    self.monitor.record_success(category)
                    
                except Exception as e:
                    logger.error(f"Error processing model: {str(e)}")
                    self.monitor.record_failure(category)
                    continue
            
            # Load transformed models
            load_result = self.loader.bulk_upsert(transformed_models)
            
            return {
                "category": category,
                "processed": len(transformed_models),
                "load_result": load_result
            }
            
        except Exception as e:
            logger.error(f"Error processing category {category}: {str(e)}")
            return {
                "category": category,
                "error": str(e)
            }
            
    def run_pipeline(self, categories: List[str] = None) -> Dict[str, Any]:
        """Run the complete pipeline"""
        self.monitor.start_pipeline()
        
        if not categories:
            categories = [
                "text-classification",
                "text-generation",
                "image-classification",
                "object-detection"
            ]
            
        try:
            results = []
            for category in categories:
                logger.info(f"Processing category: {category}")
                result = self.process_category(category)
                results.append(result)
                
            return {
                "status": "success",
                "category_results": results,
                "metrics": self.monitor.get_metrics()
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
                "metrics": self.monitor.get_metrics()
            }

def main():
    load_dotenv()
    
    config = ModelConfig(
        min_downloads=1000,
        min_likes=50,
        batch_size=100
    )
    
    pipeline = ModelPipeline(config)
    result = pipeline.run_pipeline()
    
    print("\nPipeline Results:")
    print("-" * 50)
    print(f"Status: {result['status']}")
    
    if result["status"] == "success":
        metrics = result["metrics"]
        print(f"\nDuration: {metrics['duration_seconds']} seconds")
        print(f"Processed: {metrics['processed_count']}")
        print(f"Success Rate: {metrics['success_rate']}%")
        
        print("\nBy Category:")
        for category in result["category_results"]:
            print(f"\n{category['category']}:")
            print(f"  Processed: {category['processed']}")
            if 'load_result' in category:
                print(f"  Loaded: {category['load_result']['upserted']}")

if __name__ == "__main__":
    main()
