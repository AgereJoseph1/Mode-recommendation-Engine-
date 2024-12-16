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
                        tls=True,  # Use 'tls' instead of 'ssl'
                        tlsAllowInvalidCertificates=True,  # Skip certificate verification
                        tlsCAFile=certifi.where(),  # Use the CA certificates
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
        min_likes: int = 0,
        performance_weight=0.5,
        requirement_weight=0.3,
        keyword_weight=0.2
    ) -> List[Tuple[Dict, float, Dict]]:
        usecase_embedding = self.model.encode(usecase_description)
        usecase_keywords = self.extract_keywords(usecase_description)

        # Fetch models from MongoDB
        models = list(self.collection.find({}))

        filtered_models = [m for m in models if 
                           m.get('downloads', 0) >= min_downloads and 
                           m.get('likes', 0) >= min_likes and
                           'embedding' in m and isinstance(m['embedding'], list)]

        if not filtered_models:
            return []

        model_embeddings = np.array([m['embedding'] for m in filtered_models])
        if model_embeddings.size == 0:
            return []

        base_similarities = cosine_similarity([usecase_embedding], model_embeddings)[0]

        # Normalize weights so they sum to 1
        total_weight = performance_weight + requirement_weight + keyword_weight
        performance_weight /= total_weight
        requirement_weight /= total_weight
        keyword_weight /= total_weight

        scored_models = []
        for idx, (model, base_score) in enumerate(zip(filtered_models, base_similarities)):
            requirement_score = self._calculate_requirement_score(model, requirements)
            keyword_score_val = self._calculate_keyword_score(model, usecase_keywords)
            final_score = (performance_weight * base_score) + \
                          (requirement_weight * requirement_score) + \
                          (keyword_weight * keyword_score_val)

            feature_matches = self._explain_matches(
                model, 
                requirements, 
                usecase_keywords,
                base_score,
                requirement_score,
                keyword_score_val
            )

            scored_models.append((model, final_score, feature_matches))

        scored_models.sort(key=lambda x: x[1], reverse=True)
        return scored_models[:top_k]

    def _calculate_requirement_score(self, model: Dict, requirements: Dict) -> float:
        score = 0.0
        matches = 0

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

def store_feedback(model_id: str, feedback: str):
    # Placeholder: In a real scenario, you might store this in MongoDB or another database.
    st.write(f"Feedback received for {model_id}: {feedback}")
    print(f"Feedback for {model_id}: {feedback}")  # Logging to console

def main():
    st.set_page_config(page_title="AI Model Recommender", layout="wide")
    
    # Basic styling
    st.markdown("""
    <style>
    .title {
        font-size: 2em; 
        font-weight: bold; 
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 1.2em;
        color: #666;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='title'>🤖 Advanced AI Model Recommender</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Tailor recommendations based on your requirements</div>", unsafe_allow_html=True)

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

    with st.expander("Adjust Scoring Weights"):
        st.write("Use these sliders to adjust how the final score is computed:")
        performance_weight = st.slider("Semantic Similarity Weight (Performance)", 0.0, 1.0, 0.5)
        requirement_weight = st.slider("Requirement Match Weight", 0.0, 1.0, 0.3)
        keyword_weight = st.slider("Keyword Match Weight", 0.0, 1.0, 0.2)
        st.write("These weights will be normalized so they sum up to 1.")

    with st.expander("Specify Requirements & Filtering"):
        st.markdown("**Specific Requirements**")
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

        st.markdown("**Filtering Options**")
        fcol1, fcol2 = st.columns(2)
        with fcol1:
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

        with fcol2:
            min_likes = st.number_input(
                "Minimum likes",
                min_value=0,
                value=100,
                step=10
            )

    st.markdown("**Note:** This system uses content-based filtering by comparing your description and requirements to model metadata and embeddings.")

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
                min_likes,
                performance_weight,
                requirement_weight,
                keyword_weight
            )

            if recommendations:
                st.subheader("📊 Recommended Models")
                for idx, (model, score, explanation) in enumerate(recommendations):
                    text, fig = format_recommendation_advanced(model, score, explanation)

                    colA, colB = st.columns([2, 1])
                    with colA:
                        st.markdown(text)
                        feedback = st.radio(
                            f"Feedback for {model.get('model_id', 'Unknown Model')}:",
                            ["No feedback", "Like", "Dislike"],
                            horizontal=True
                        )
                        if feedback != "No feedback":
                            store_feedback(model.get('model_id', 'Unknown'), feedback)
                    with colB:
                        st.plotly_chart(fig, use_container_width=True, key=f"plot_{idx}")

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
