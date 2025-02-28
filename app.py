import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

# Title and description
st.title("Keyword Clustering App")
st.write("""
Upload a CSV file with your keywords and their search volumes.
This app will cluster similar keywords (based on semantic similarity) and designate the highest–volume keyword in each cluster as the seed.
""")

# File uploader
uploaded_file = st.file_uploader("Upload your keyword CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.dataframe(data.head())

    # Let user map fields
    keyword_col = st.selectbox("Select the Keyword column:", data.columns)
    volume_col = st.selectbox("Select the Search Volume column:", data.columns)
    
    # Optionally, allow user to adjust clustering threshold
    threshold = st.slider("Clustering distance threshold (lower => more clusters)", 0.1, 1.0, 0.3, 0.05)

    if st.button("Run Clustering"):
        # Preprocess: drop NaNs and strip keyword strings
        data = data.dropna(subset=[keyword_col, volume_col])
        data[keyword_col] = data[keyword_col].astype(str).str.strip()

        # Compute embeddings for each keyword
        st.write("Computing embeddings for keywords...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        keywords = data[keyword_col].tolist()
        embeddings = model.encode(keywords, show_progress_bar=True)
        
        # Compute cosine similarity and convert to distance
        cosine_sim = cosine_similarity(embeddings)
        cosine_dist = 1 - cosine_sim  # distance = 1 - similarity
        
        # Perform Agglomerative Clustering with a distance threshold
        st.write("Clustering keywords...")
        clustering_model = AgglomerativeClustering(
            n_clusters=None,
            affinity='precomputed',
            linkage='average',
            distance_threshold=threshold
        )
        cluster_labels = clustering_model.fit_predict(cosine_dist)
        data["cluster"] = cluster_labels

        # For each cluster, pick the keyword with highest volume as the seed
        cluster_summary = []
        for cluster in np.unique(cluster_labels):
            cluster_data = data[data["cluster"] == cluster]
            seed_row = cluster_data.loc[cluster_data[volume_col].idxmax()]
            children_count = len(cluster_data) - 1
            cluster_summary.append({
                "Seed Keyword": seed_row[keyword_col],
                "Children Count": children_count,
                "Total Cluster Size": len(cluster_data)
            })

        seed_df = pd.DataFrame(cluster_summary).sort_values(by="Total Cluster Size", ascending=False)

        st.write("### Cluster Summary (Seed Keywords)")
        st.dataframe(seed_df)

        # Option to download the results
        csv = seed_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Cluster Summary CSV",
            data=csv,
            file_name="cluster_summary.csv",
            mime="text/csv"
        )
