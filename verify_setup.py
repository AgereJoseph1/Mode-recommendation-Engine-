# verify_setup.py
import os
from dotenv import load_dotenv
from huggingface_hub import HfApi
from pymongo import MongoClient

def verify_setup():
    # Load environment variables
    load_dotenv()
    
    # Get credentials
    hf_token = os.getenv('HUGGINGFACE_TOKEN')
    mongo_uri = os.getenv('MONGODB_URI')
    
    if not all([hf_token, mongo_uri]):
        print("❌ Missing environment variables!")
        return False
        
    try:
        # Test HuggingFace connection
        api = HfApi(token=hf_token)
        user = api.whoami()
        print(f"✅ HuggingFace connection successful: {user['name']}")
        
        # Test MongoDB connection
        client = MongoClient(mongo_uri)
        client.admin.command('ping')
        print("✅ MongoDB connection successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing connections...")
    verify_setup()