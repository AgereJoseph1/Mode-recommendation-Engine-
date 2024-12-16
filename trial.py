from huggingface_hub import hf_hub_download
import os

# Function to fetch and read model card (README.md)
def fetch_model_card(model_id):
    try:
        # Download the README.md file
        readme_path = hf_hub_download(repo_id=model_id, filename="README.md")
        
        # Read the content of the model card
        with open(readme_path, "r", encoding="utf-8") as file:
            model_card_content = file.read()
        return model_card_content
    except Exception as e:
        print(f"Failed to fetch model card for {model_id}: {e}")
        return None

# Example usage
model_id = "papluca/xlm-roberta-base-language-detection"  # Replace with your model ID
model_card = fetch_model_card(model_id)

if model_card:
    print(f"Model Card Content for {model_id}:\n{model_card[3:]}")  # Print first 500 characters
