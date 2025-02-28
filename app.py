import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

# Sidebar: About the App
st.sidebar.title("About the App")
st.sidebar.markdown("""
This app clusters similar keywords based on their semantic similarity.

**How It Works:**
- **Embedding:** The app computes vector embeddings for each keyword using a pre-trained language model (SentenceTransformer).
- **Clustering:** Using Agglomerative Clustering with cosine distance, it groups semantically similar keywords.
- **Seed Selection:** Within each cluster, the keyword with the highest search volume is chosen as the seed.
- **Output:** The app displays a summary table showing the seed keywords along with the number of related (child) keywords.

**Why It's Valuable:**
- **Keyword Grouping:** Reduce redundancy by identifying core keywords.
- **Simplified Analysis:** Manage large lists by focusing on high-value seed keywords.
- **Efficiency:** Automate the process of grouping and de-duplication, saving time in keyword research.
""")

# Main Title and Instructions
st.title("Keyword Clustering App (Paste Mode)")
st.write("""
**Instructions:**  
Paste your keyword list below in a two–column, tab-delimited format.  
The first column should be the **Keyword** and the second column the **Volume**.  

If your data file **does not include a header row**, check the **"Ignore first row"** checkbox so that the first row is treated as data rather than a header.
""")

ignore_first_row = st.checkbox("Ignore first row (my data does NOT have a header row)", value=False)
pasted_data = st.text_area("Paste your keyword list here:", height=200)

if pasted_data:
    try:
        if ignore_first_row:
            data = pd.read_csv(StringIO(pasted_data), delimiter="\t", header=None)
            data.columns = ["Keyword", "Volume"]
        else:
            data = pd.read_csv(StringIO(pasted_data), delimiter="\t")
    except Exception as e:
        st.error(f"Error parsing data: {e}")
    else:
        st.write("### Data Preview")
        st.dataframe(data.head())

        cols = list(data.columns)
        keyword_col = st.selectbox("Select the Keyword column:", cols, index=cols.index("Keyword") if "Keyword" in cols else 0)
        volume_col = st.selectbox("Select the Search Volume column:", cols, index=cols.index("Volume") if "Volume" in cols else 1)

        threshold = st.slider("Clustering distance threshold (lower => more clusters)", 0.1, 1.0, 0.3, 0.05)

        if st.button("Run Clustering"):
            data = data.dropna(subset=[keyword_col, volume_col])
            data[keyword_col] = data[keyword_col].astype(str).str.strip()

            st.write("Computing embeddings for keywords...")
            model = SentenceTransformer('all-MiniLM-L6-v2')
            keywords = data[keyword_col].tolist()
            embeddings = model.encode(keywords, show_progress_bar=True)

            st.write("Clustering keywords...")
            cosine_sim = cosine_similarity(embeddings)
            cosine_dist = 1 - cosine_sim

            clustering_model = AgglomerativeClustering(
                n_clusters=None,
                metric='precomputed',
                linkage='average',
                distance_threshold=threshold
            )
            cluster_labels = clustering_model.fit_predict(cosine_dist)
            data["cluster"] = cluster_labels

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

            csv = seed_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Cluster Summary CSV",
                data=csv,
                file_name="cluster_summary.csv",
                mime="text/csv"
            )
