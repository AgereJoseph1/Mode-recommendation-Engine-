import streamlit as st
import numpy as np
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from typing import Dict, List, Tuple
import spacy
import json
import plotly.express as px
import plotly.graph_objects as go
import certifi
import os
import time
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import torch
import subprocess
import sys

os.environ['SSL_CERT_DIR'] = '/etc/ssl/certs'  # On Unix-like systems

def check_dependencies():
    try:
        import torch
        import sentence_transformers
        st.info(f"PyTorch version: {torch.__version__}")
        st.info(f"Sentence-transformers version: {sentence_transformers.__version__}")
    except ImportError as e:
        st.error(f"Missing dependency: {str(e)}")
        st.info("Installing required packages...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "torch", "sentence-transformers"])
        st.info("Please restart the application")
        st.stop()

class AdvancedUseCaseRecommender:
    def __init__(self, mongodb_uri, db_name="model_metadata", collection_name="models"):
        try:
            # Initialize SentenceTransformer with error handling
            try:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
            except Exception as e:
                st.error(f"Failed to initialize SentenceTransformer: {str(e)}")
                subprocess.run(['pip', 'install', '--force-reinstall', 'sentence-transformers'])
                self.model = SentenceTransformer('all-MiniLM-L6-v2')

            if not mongodb_uri:
                raise ValueError("MongoDB URI is required")

            max_retries = 3
            retry_count = 0

            while retry_count < max_retries:
                try:
                    self.client = MongoClient(
                        mongodb_uri,
                        ssl=True,
                        ssl_cert_reqs='CERT_NONE',
                        tlsCAFile=certifi.where(),
                        connectTimeoutMS=30000,
                        serverSelectionTimeoutMS=30000,
                        retryWrites=True,
                        w='majority'
                    )
                    self.client.admin.command('ping')
                    self.db = self.client[db_name]
                    self.collection = self.db[collection_name]
                    break
                except Exception as e:
                    retry_count += 1
                    if retry_count == max_retries:
                        st.error(f"Failed to connect to MongoDB after {max_retries} attempts. Error: {str(e)}")
                        raise Exception(f"MongoDB connection failed: {str(e)}")
                    time.sleep(1)

            # Initialize spaCy
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except OSError:
                st.info("Downloading required language model...")
                subprocess.run([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'])
                self.nlp = spacy.load('en_core_web_sm')

            # Debug prints
            st.write("Connected to DB:", self.db.name)
            st.write("Using Collection:", self.collection.name)

        except Exception as e:
            st.error(f"Initialization failed: {str(e)}")
            raise

    def extract_keywords(self, text: str) -> List[str]:
        doc = self.nlp(text)
        keywords = []
        keywords.extend([ent.text.lower() for ent in doc.ents])
        keywords.extend([token.text.lower() for token in doc if (token.pos_ in ['NOUN', 'VERB']) and len(token.text) > 2])
        return list(set(keywords))

    def analyze_requirements(self, description: str) -> Dict:
        desc_lower = description.lower()
        requirements = {
            'performance_critical': any(term in desc_lower for term in ['fast', 'quick', 'speed', 'performance']),
            'multilingual': any(term in desc_lower for term in ['language', 'multilingual', 'multi-lingual']),
            'real_time': any(term in desc_lower for term in ['real-time', 'real time', 'realtime', 'live']),
            'production_ready': any(term in desc_lower for term in ['production', 'deploy', 'deployment']),
            'size_constraints': any(term in desc_lower for term in ['small', 'lightweight', 'mobile', 'edge']),
            'domain_specific': None
        }

        domains = {
            'medical': ['medical', 'healthcare', 'clinical', 'diagnosis'],
            'financial': ['financial', 'finance', 'banking', 'trading'],
            'social': ['social media', 'twitter', 'facebook', 'instagram'],
            'academic': ['academic', 'scientific', 'research', 'papers'],
            'legal': ['legal', 'law', 'contract', 'regulatory']
        }
        
        for domain, terms in domains.items():
            if any(term in desc_lower for term in terms):
                requirements['domain_specific'] = domain
                break

        return requirements

    def recommend_for_usecase(
        self, 
        usecase_description: str, 
        requirements: Dict,
        top_k: int = 5,
        min_downloads: int = 0,
        min_likes: int = 0
    ) -> List[Tuple[Dict, float, Dict]]:
        usecase_embedding = self.model.encode(usecase_description)
        usecase_keywords = self.extract_keywords(usecase_description)

        st.write("Fetching models from MongoDB with no filtering:")
        models = list(self.collection.find({}))  # fetch all for debugging

        st.write(f"Number of models retrieved: {len(models)}")
        if models:
            st.write("Sample model:", models[0])
            # Check embedding for the first model
            if 'embedding' in models[0]:
                st.write("Type of embedding field in first model:", type(models[0]['embedding']))
                if isinstance(models[0]['embedding'], list):
                    st.write("First 5 elements of embedding:", models[0]['embedding'][:5])
                else:
                    st.write("Embedding is not a list. Check data type in DB.")
            else:
                st.write("No 'embedding' field in the first model.")

        filtered_models = [m for m in models if 
                           m.get('downloads', 0) >= min_downloads and 
                           m.get('likes', 0) >= min_likes and
                           'embedding' in m and isinstance(m['embedding'], list)]
        
        st.write(f"Number of models after filtering and embedding check: {len(filtered_models)}")

        if not filtered_models:
            st.warning("No models with embeddings found after filtering.")
            return []

        model_embeddings = np.array([m['embedding'] for m in filtered_models])
        if model_embeddings.size == 0:
            st.warning("No valid embeddings found in models.")
            return []

        base_similarities = cosine_similarity([usecase_embedding], model_embeddings)[0]

        scored_models = []
        for idx, (model, base_score) in enumerate(zip(filtered_models, base_similarities)):
            requirement_score = self._calculate_requirement_score(model, requirements)
            keyword_score = self._calculate_keyword_score(model, usecase_keywords)
            final_score = (0.5 * base_score) + (0.3 * requirement_score) + (0.2 * keyword_score)

            feature_matches = self._explain_matches(
                model, 
                requirements, 
                usecase_keywords,
                base_score,
                requirement_score,
                keyword_score
            )

            scored_models.append((model, final_score, feature_matches))

        scored_models.sort(key=lambda x: x[1], reverse=True)
        return scored_models[:top_k]

    def _calculate_requirement_score(self, model: Dict, requirements: Dict) -> float:
        score = 0.0
        matches = 0

        # Convert model to text once, with default=str to avoid ObjectId errors
        model_text = json.dumps(model, default=str).lower()

        if requirements['performance_critical'] and model.get('model_size', float('inf')) < 1000000000:
            score += 1
            matches += 1

        if requirements['multilingual'] and len(model.get('languages', [])) > 1:
            score += 1
            matches += 1

        if requirements['real_time']:
            if any(term in model_text for term in ['real-time', 'real time', 'fast', 'quick']):
                score += 1
                matches += 1

        if requirements['production_ready'] and model.get('downloads', 0) > 10000:
            score += 1
            matches += 1

        if requirements['size_constraints'] and model.get('model_size', float('inf')) < 500000000:
            score += 1
            matches += 1

        if requirements['domain_specific']:
            if requirements['domain_specific'] in model_text:
                score += 2
                matches += 1

        return score / matches if matches > 0 else 0.0

    def _calculate_keyword_score(self, model: Dict, keywords: List[str]) -> float:
        # Use default=str to avoid issues with non-serializable fields
        model_text = json.dumps(model, default=str).lower()
        matched_keywords = sum(1 for keyword in keywords if keyword in model_text)
        return matched_keywords / len(keywords) if keywords else 0

    def _explain_matches(
        self, 
        model: Dict, 
        requirements: Dict,
        keywords: List[str],
        base_score: float,
        req_score: float,
        keyword_score: float
    ) -> Dict:
        explanation = {
            'semantic_similarity': base_score,
            'requirement_score': req_score,
            'keyword_score': keyword_score,
            'matched_requirements': [],
            'matched_keywords': [],
            'performance_metrics': {
                'downloads': model.get('downloads', 0),
                'likes': model.get('likes', 0),
                'size': model.get('model_size', 0)
            }
        }

        # We can re-use model_text if needed, but here let's just do a minimal approach
        model_text = json.dumps(model, default=str).lower()

        if requirements['performance_critical'] and model.get('model_size', float('inf')) < 1000000000:
            explanation['matched_requirements'].append('Meets performance requirements')

        if requirements['multilingual'] and len(model.get('languages', [])) > 1:
            explanation['matched_requirements'].append(
                f"Supports multiple languages: {', '.join(model.get('languages', []))}"
            )

        explanation['matched_keywords'] = [kw for kw in keywords if kw in model_text]

        return explanation

def create_score_chart(scores: Dict) -> go.Figure:
    categories = list(scores.keys())
    values = list(scores.values())

    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False
    )

    return fig

def format_recommendation_advanced(model: Dict, score: float, explanation: Dict) -> Tuple[str, go.Figure]:
    score_summary = {
        'Semantic Match': explanation['semantic_similarity'],
        'Requirement Match': explanation['requirement_score'],
        'Keyword Match': explanation['keyword_score']
    }

    fig = create_score_chart(score_summary)

    model_id = model.get('model_id', 'Unknown Model')
    task = model.get('task', 'N/A')
    downloads = model.get('downloads', 0)
    likes = model.get('likes', 0)
    size_mb = model.get('model_size', 0) / 1_000_000.0
    description = model.get('description', 'No description available')
    matched_reqs = explanation['matched_requirements']
    matched_reqs_text = '\n'.join(f"- {r}" for r in matched_reqs)
    matched_keywords = explanation['matched_keywords']
    matched_kw_text = ', '.join(matched_keywords) if matched_keywords else 'No direct keyword matches'
    # Handle tags which may be string or array
    tags_val = model.get('tags', 'No tags')
    if isinstance(tags_val, list):
        tags = ', '.join(tags_val)
    else:
        tags = str(tags_val)

    text = f"""
    ### {model_id} (Overall Score: {score:.2%})

    **Task:** {task}  
    **Performance Metrics:**  
    - Downloads: {downloads:,}  
    - Likes: {likes}  
    - Model Size: {size_mb:.1f}MB

    **Description:**  
    {description}

    **Matching Features:**  
    {matched_reqs_text if matched_reqs_text else 'None'}

    **Matching Keywords:**  
    {matched_kw_text}

    **Tags:** {tags if tags else 'No tags'}
    """

    return text, fig

def main():
    st.title("🤖 Advanced AI Model Recommender")
    st.write("Advanced model recommendations based on your specific use case requirements")

    check_dependencies()

    if "mongodb_uri" not in st.secrets:
        st.error("MongoDB URI not found in secrets!")
        return

    try:
        recommender = AdvancedUseCaseRecommender(
            mongodb_uri=st.secrets["mongodb_uri"],
            db_name=st.secrets.get("mongodb_database", "model_metadata"),
            collection_name=st.secrets.get("mongodb_collection", "models")
        )
    except Exception as e:
        st.error(f"Failed to initialize recommender: {str(e)}")
        st.info("Please check your MongoDB connection settings and try again.")
        return

    user_description = st.text_area(
        "Describe your use case in detail",
        height=150,
        placeholder="Example: I need a fast, multilingual sentiment analysis model for processing customer feedback in real-time..."
    )

    with st.expander("Specific Requirements", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            performance_critical = st.checkbox("Performance Critical")
            multilingual = st.checkbox("Multilingual Support")
            real_time = st.checkbox("Real-time Processing")

        with col2:
            production_ready = st.checkbox("Production Ready")
            size_constraints = st.checkbox("Size Constraints")

        domain = st.selectbox(
            "Specific Domain",
            ["None", "Medical", "Financial", "Social", "Academic", "Legal"]
        )

    with st.expander("Filtering Options"):
        col1, col2 = st.columns(2)

        with col1:
            num_recommendations = st.slider(
                "Number of recommendations",
                min_value=1,
                max_value=10,
                value=5
            )

            min_downloads = st.number_input(
                "Minimum downloads",
                min_value=0,
                value=1000,
                step=1000
            )

        with col2:
            min_likes = st.number_input(
                "Minimum likes",
                min_value=0,
                value=100,
                step=10
            )

    if st.button("Find Models") and user_description:
        with st.spinner("Analyzing requirements and finding optimal models..."):
            requirements = {
                'performance_critical': performance_critical,
                'multilingual': multilingual,
                'real_time': real_time,
                'production_ready': production_ready,
                'size_constraints': size_constraints,
                'domain_specific': domain.lower() if domain != "None" else None
            }

            recommendations = recommender.recommend_for_usecase(
                user_description,
                requirements,
                num_recommendations,
                min_downloads,
                min_likes
            )

            if recommendations:
                st.subheader("📊 Recommended Models")

                for model, score, explanation in recommendations:
                    text, fig = format_recommendation_advanced(model, score, explanation)

                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.markdown(text)

                    with col2:
                        st.plotly_chart(fig, use_container_width=True)

                    st.markdown("---")
            else:
                st.warning("No models found matching your criteria. Try adjusting the filters.")

    st.markdown("---")
    st.markdown("""
    💡 **Tips for Better Results:**
    - Describe your use case in detail
    - Specify any technical requirements
    - Mention your target domain
    - Include performance requirements
    - Specify language requirements
    """)

if __name__ == "__main__":
    main()
