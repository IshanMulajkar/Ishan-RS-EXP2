import streamlit as st
import pickle
import numpy as np
from scipy.stats import spearmanr
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .content-box {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    .header-text {
        font-size: 3rem;
        font-weight: bold;
        color: white;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .student-info {
        text-align: center;
        color: white;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    .recommendation-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .rating-badge {
        background: #ffd700;
        color: #333;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin-top: 0.5rem;
    }
    .metric-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("⚠️ Model file 'model.pkl' not found!")
        return None

model = load_model()

if model is not None:
    train_matrix = model["train_matrix"]
    user_means = model["user_means"]
    rating_scale = model.get("rating_scale", [1, 2, 3, 4, 5])

    # Functions
    def spearman_similarity(u_ratings, v_ratings, min_overlap=3):
        common = u_ratings.index.intersection(v_ratings.index)
        u_vals = u_ratings.loc[common].dropna()
        v_vals = v_ratings.loc[common].dropna()
        common = u_vals.index.intersection(v_vals.index)
        if len(common) < min_overlap:
            return 0.0, len(common)
        u_vals = u_vals.loc[common]
        v_vals = v_vals.loc[common]
        rho, p = spearmanr(u_vals, v_vals)
        if np.isnan(rho):
            return 0.0, len(common)
        return float(rho), len(common)

    def predict_rating(user_id, item_id, k=10, min_overlap=3):
        if item_id not in train_matrix.columns:
            return None
        if user_id not in train_matrix.index:
            return None
        neighbors = train_matrix[train_matrix[item_id].notna()].index.tolist()
        if len(neighbors) == 0:
            return user_means.get(user_id, train_matrix.stack().mean())
        sims = []
        for nb in neighbors:
            if nb == user_id:
                continue
            rho, overlap = spearman_similarity(
                train_matrix.loc[user_id], 
                train_matrix.loc[nb], 
                min_overlap=min_overlap
            )
            if rho != 0.0:
                sims.append((nb, rho))
        if len(sims) == 0:
            return user_means.get(user_id, train_matrix.stack().mean())
        sims_sorted = sorted(sims, key=lambda x: abs(x[1]), reverse=True)[:k]
        num = 0.0
        den = 0.0
        target_mean = user_means.get(user_id, train_matrix.stack().mean())
        for nb, rho in sims_sorted:
            nb_mean = user_means.get(nb, train_matrix.stack().mean())
            nb_rating = train_matrix.at[nb, item_id]
            if np.isnan(nb_rating):
                continue
            num += rho * (nb_rating - nb_mean)
            den += abs(rho)
        if den == 0.0:
            return target_mean
        pred = target_mean + num / den
        pred = float(np.clip(pred, min(rating_scale), max(rating_scale)))
        return pred

    def recommend_top_n(user_id, n=5, k=10, min_overlap=3):
        if user_id not in train_matrix.index:
            return []
        unrated = train_matrix.columns[train_matrix.loc[user_id].isna()]
        preds = []
        for item in unrated:
            pr = predict_rating(user_id, item, k=k, min_overlap=min_overlap)
            if pr is not None:
                preds.append((item, pr))
        preds_sorted = sorted(preds, key=lambda x: x[1], reverse=True)
        return preds_sorted[:n]

    # Header
    st.markdown('<div class="header-text">🎬 Movie Recommendation System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="student-info">Name: Ishan Mulajkar | UID: 2023701004 | EXP 2: Recommendation System</div>',
        unsafe_allow_html=True
    )

    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/movie-projector.png", width=80)
        st.title("⚙️ Configuration")
        st.markdown("---")
        
        users = list(train_matrix.index)
        selected_user = st.selectbox(
            "👤 Select User ID",
            options=users,
            index=0
        )
        
        top_n = st.slider(
            "📊 Number of Recommendations",
            min_value=1,
            max_value=20,
            value=5,
            step=1
        )
        
        k_neighbors = st.slider(
            "👥 Number of Neighbors (k)",
            min_value=1,
            max_value=50,
            value=10,
            step=1
        )
        
        min_overlap = st.slider(
            "🔗 Minimum Overlap",
            min_value=1,
            max_value=10,
            value=3,
            step=1
        )
        
        st.markdown("---")
        get_recommendations = st.button("🚀 Get Recommendations", use_container_width=True)

    # Main content
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Total Users", len(train_matrix.index))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Total Items", len(train_matrix.columns))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        user_ratings = train_matrix.loc[selected_user].notna().sum()
        st.metric("User Ratings", user_ratings)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Get recommendations
    if get_recommendations or selected_user:
        with st.spinner("🔮 Generating personalized recommendations..."):
            recs = recommend_top_n(
                selected_user, 
                n=top_n, 
                k=k_neighbors, 
                min_overlap=min_overlap
            )
        
        if recs:
            st.markdown('<div class="content-box">', unsafe_allow_html=True)
            st.subheader(f"🎯 Top {len(recs)} Recommendations for User: {selected_user}")
            
            for idx, (item, rating) in enumerate(recs, 1):
                st.markdown(f"""
                    <div class="recommendation-card">
                        <h3>#{idx} - {item}</h3>
                        <div class="rating-badge">⭐ Predicted Rating: {rating:.2f}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show recommendations as dataframe
            st.markdown('<div class="content-box">', unsafe_allow_html=True)
            st.subheader("📋 Detailed View")
            df_recs = pd.DataFrame(recs, columns=["Item", "Predicted Rating"])
            df_recs.insert(0, "Rank", range(1, len(df_recs) + 1))
            st.dataframe(df_recs, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("⚠️ No recommendations available for this user with the current settings.")
    
    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="student-info">Built with ❤️ using Streamlit | Collaborative Filtering with Spearman Correlation</div>',
        unsafe_allow_html=True
    )
else:
    st.error("❌ Unable to load the recommendation model. Please ensure 'model.pkl' exists in the same directory.")
