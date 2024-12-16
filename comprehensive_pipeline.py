import os
from datetime import datetime, UTC
from typing import Dict, List, Any, Optional
from huggingface_hub import HfApi, hf_hub_download
import re
import json

from typing import List, Any
import dask.dataframe as dd
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import HfApi
from pymongo import MongoClient, DESCENDING
from sentence_transformers import SentenceTransformer
from prefect import flow, task, get_run_logger
from dask.distributed import Client
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Define default categories
DEFAULT_CATEGORIES = [
    "text-classification",
    "text-generation",
    "image-classification",
    "object-detection",
    "text-to-image",
    "image-to-text",
    "summarization",
    "question-answering"
]

class PipelineParams(BaseModel):
    """Pipeline parameters with validation"""
    categories: List[str] = Field(default=DEFAULT_CATEGORIES, min_items=1)
    models_per_category: int = Field(default=50, gt=0)
    min_downloads: int = Field(default=100, ge=0)
    min_likes: int = Field(default=10, ge=0)

@task(retries=3, retry_delay_seconds=60)
def fetch_models(category: str, limit: int = 50) -> List[Dict]:
    """Fetch models from HuggingFace"""
    logger = get_run_logger()
    api = HfApi(token=os.getenv('HUGGINGFACE_TOKEN'))
    
    if not os.getenv('HUGGINGFACE_TOKEN'):
        raise ValueError("HUGGINGFACE_TOKEN environment variable is not set")
    
    try:
        logger.info(f"Fetching {category} models...")
        models = list(api.list_models(
            filter=category,
            limit=limit,
            sort="downloads",
            direction=-1
        ))
        logger.info(f"Fetched {len(models)} {category} models")
        return models
    except Exception as e:
        logger.error(f"Error fetching {category} models: {e}")
        raise

import re
import json
from huggingface_hub import hf_hub_download
from typing import List, Any
import pandas as pd
from prefect import task, get_run_logger

@task
def extract_metadata(models: List[Any]) -> pd.DataFrame:
    """Extract metadata from models, including description and config details."""
    logger = get_run_logger()
    
    if not models:
        logger.warning("No models provided to extract_metadata")
        return pd.DataFrame()
    
    metadata_list = []

    for model in models:
        try:
            # Base metadata
            metadata = {
                'model_id': model.id,
                'downloads': getattr(model, 'downloads', 0),
                'likes': getattr(model, 'likes', 0),
                'tags': getattr(model, 'tags', []),
                'pipeline_tag': getattr(model, 'pipeline_tag', ''),
                'last_modified': getattr(model, 'lastModified', None),
                'license': getattr(model, 'license', 'unknown'),
                'author': getattr(model, 'author', 'unknown'),
                'languages': [],
                'description': 'No description available',
                'task': '',
                'architecture': '',
                'vocab_size': None
            }

            # Fetch model card (README.md)
            try:
                readme_path = hf_hub_download(repo_id=model.id, filename="README.md")
                with open(readme_path, "r", encoding="utf-8") as f:
                    model_card_content = f.read()

                    # Extract Description
                    description_match = re.search(r"(##\s*Model description.*?)(##|$)", model_card_content, re.DOTALL)
                    if description_match:
                        metadata['description'] = description_match.group(1).replace("##", "").strip()
                    else:
                        fallback_match = re.search(r"^(?!##)([^\n]+)\n", model_card_content, re.MULTILINE)
                        if fallback_match:
                            metadata['description'] = fallback_match.group(1).strip()

                    # Extract Task from Model Card
                    task_match = re.search(r"(##\s*Task.*?)(##|$)", model_card_content, re.DOTALL)
                    if task_match:
                        task_content = task_match.group(1).replace("##", "").strip()
                        metadata['task'] = task_content.split("\n")[0].strip()
                    elif metadata['pipeline_tag']:
                        metadata['task'] = metadata['pipeline_tag']

            except Exception as e:
                logger.warning(f"Could not fetch description/task for {model.id}: {e}")
                metadata['task'] = metadata['pipeline_tag']

            # Fetch config.json for architecture details
            try:
                config_path = hf_hub_download(repo_id=model.id, filename="config.json")
                with open(config_path, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
                    metadata['architecture'] = config_data.get('_name_or_path', 'unknown')
                    metadata['vocab_size'] = config_data.get('vocab_size', None)
            except Exception as e:
                logger.warning(f"Could not fetch config for {model.id}: {e}")

            metadata_list.append(metadata)

        except Exception as e:
            logger.warning(f"Error processing model {getattr(model, 'id', 'unknown')}: {e}")
            continue

    # Convert metadata to DataFrame
    df = pd.DataFrame(metadata_list)
    logger.info(f"Processed {len(df)} models with descriptions, tasks, and architectures")
    return df

@task(retries=2)
def create_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    """Create embeddings using Dask with better error handling"""
    logger = get_run_logger()
    
    if df.empty:
        logger.warning("Empty DataFrame provided to create_embeddings")
        return df
    
    # Initialize the model once outside the mapping function
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def embed_text(text: str) -> List[float]:
        """Create embedding for a single text"""
        try:
            if pd.isna(text) or not isinstance(text, str) or text.strip() == '':
                return []
            return model.encode(text).tolist()
        except Exception as e:
            logger.warning(f"Error creating embedding for text: {e}")
            return []
    
    # Reduce partition size to avoid memory issues
    npartitions = min(len(df) // 50 + 1, os.cpu_count() or 4)
    ddf = dd.from_pandas(df, npartitions=npartitions)
    
    logger.info(f"Creating embeddings with {npartitions} partitions...")
    try:
        # Apply embedding function to the description
        ddf['embedding'] = ddf['description'].map(embed_text, meta=('embedding', 'object'))
        result = ddf.compute(scheduler='threads')
        logger.info("Embeddings created successfully")
        return result
    except Exception as e:
        logger.error(f"Error in embedding creation: {e}")
        raise

@task(retries=2)
def save_to_mongodb(df: pd.DataFrame, category: str) -> Dict[str, Any]:
    """Save processed data to MongoDB with connection pooling"""
    logger = get_run_logger()
    client = None
    
    if df.empty:
        logger.warning(f"No data to save for category {category}")
        return {"category": category, "processed": 0}
    
    # Validate MongoDB connection parameters
    required_env_vars = ['MONGODB_URI', 'MONGODB_DATABASE', 'MONGODB_COLLECTION']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    try:
        client = MongoClient(
            os.getenv('MONGODB_URI'),
            maxPoolSize=50,
            retryWrites=True,
            w='majority'
        )
        db = client[os.getenv('MONGODB_DATABASE')]
        collection = db[os.getenv('MONGODB_COLLECTION')]
        
        # Create indexes
        collection.create_index([("model_id", DESCENDING)], unique=True)
        collection.create_index([("category", DESCENDING)])
        collection.create_index([("downloads", DESCENDING)])
        collection.create_index([("likes", DESCENDING)])
        
        # Prepare records with proper datetime handling
        records = df.to_dict('records')
        current_time = datetime.now(UTC)
        
        for record in records:
            record['category'] = category
            record['processed_at'] = current_time
            
            # Handle non-serializable objects
            record = {k: (v.isoformat() if isinstance(v, datetime) else v) 
                     for k, v in record.items()}
            
            try:
                collection.update_one(
                    {"model_id": record["model_id"]},
                    {"$set": record},
                    upsert=True
                )
            except Exception as e:
                logger.error(f"Error updating record {record.get('model_id')}: {e}")
                raise
        
        return {
            "category": category,
            "processed": len(records),
            "timestamp": current_time.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error saving to MongoDB: {e}")
        raise
    finally:
        if client:
            client.close()

@flow(name="process_category")
def process_category(
    category: str, 
    limit: int = 50,
    min_downloads: int = 100,
    min_likes: int = 10
) -> Dict[str, Any]:
    """Process a single category with better error handling"""
    logger = get_run_logger()
    
    try:
        # Fetch models
        models = fetch_models(category, limit)
        
        if not models:
            logger.warning(f"No models found for category {category}")
            return {"category": category, "processed": 0}
        
        # Extract metadata
        df = extract_metadata(models)
        
        if df.empty:
            logger.warning(f"No metadata extracted for category {category}")
            return {"category": category, "processed": 0}
        
        # Filter by quality criteria
        df = df[
            (df['downloads'] >= min_downloads) &
            (df['likes'] >= min_likes)
        ].copy()
        
        if df.empty:
            logger.info(f"No models meet quality criteria for category {category}")
            return {"category": category, "processed": 0}
        
        # Create embeddings
        df_with_embeddings = create_embeddings(df)
        
        # Save to MongoDB
        result = save_to_mongodb(df_with_embeddings, category)
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing category {category}: {e}")
        return {
            "category": category,
            "error": str(e),
            "timestamp": datetime.now(UTC).isoformat()
        }

@flow(name="model_pipeline")
def run_pipeline(params: PipelineParams) -> Dict[str, Any]:
    """Main pipeline flow with improved error handling and reporting"""
    logger = get_run_logger()
    client = None
    
    try:
        # Setup Dask client with explicit memory limits
        client = Client(n_workers=4, threads_per_worker=2, memory_limit='4GB')
        logger.info(f"Dask dashboard available at: {client.dashboard_link}")
        
        results = {
            "total_processed": 0,
            "categories": {},
            "start_time": datetime.now(UTC).isoformat(),
            "params": params.dict()
        }
        
        for category in params.categories:
            result = process_category(
                category=category,
                limit=params.models_per_category,
                min_downloads=params.min_downloads,
                min_likes=params.min_likes
            )
            results["categories"][category] = result
            if "processed" in result:
                results["total_processed"] += result["processed"]
        
        results["end_time"] = datetime.now(UTC).isoformat()
        return results
        
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise
    finally:
        if client:
            client.close()

def main():
    try:
        # Create pipeline parameters with validation
        params = PipelineParams(
            models_per_category=50,
            min_downloads=100,
            min_likes=10
        )
        
        # Run pipeline
        print("Starting distributed pipeline...")
        results = run_pipeline(params)
        
        # Print results
        print("\nPipeline Results:")
        print("-" * 50)
        print(f"Total models processed: {results['total_processed']}")
        print(f"Start time: {results['start_time']}")
        print(f"End time: {results['end_time']}")
        print("\nBy Category:")
        for category, stats in results["categories"].items():
            print(f"\n{category}:")
            if "processed" in stats:
                print(f"  Processed: {stats['processed']}")
            if "error" in stats:
                print(f"  Error: {stats['error']}")
    
    except Exception as e:
        print(f"Fatal error in pipeline: {e}")
        raise

if __name__ == "__main__":
    main()