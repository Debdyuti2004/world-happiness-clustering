# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram

from src.preprocessing import load_and_clean_data, scale_features
from src.clustering import perform_kmeans
from src.visualize import summarize_clusters

st.set_page_config(page_title="🌍 World Happiness Clustering Dashboard", layout="wide")

# ------------------------------------------
# APP TITLE
# ------------------------------------------
st.title("🌍 World Happiness Clustering Dashboard")
st.markdown("Explore clusters of countries based on happiness indicators.")

# ------------------------------------------
# STEP 1: UPLOAD OR LOAD DEFAULT DATA
# ------------------------------------------
st.header("📁 Step 1: Load Dataset")

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Custom dataset loaded successfully!")
else:
    st.info("Using default dataset from `data/world_happiness.csv`...")
    df = load_and_clean_data()

st.dataframe(df.head())

# STEP 2: SCALE FEATURES
st.header("⚙️ Step 2: Scale Numeric Features")
try:
    X_scaled, scaler, numeric_cols = scale_features(df)
    st.success(f"✅ Features scaled successfully! Columns used: {', '.join(numeric_cols)}")
except ValueError as e:
    st.error(f"Error: {e}")


# STEP 3: FIND OPTIMAL K (Elbow Method)
st.header("📊 Step 3: Find Optimal Number of Clusters (Elbow Method)")

max_k = st.slider("Select max k for Elbow Method", min_value=3, max_value=15, value=10)
if st.button("Show Elbow Plot"):
    from sklearn.cluster import KMeans
    wcss = []
    for i in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, max_k + 1), wcss, marker="o")
    ax.set_title("Elbow Method - Optimal k")
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("WCSS")
    st.pyplot(fig)

# STEP 4: RUN K-MEANS CLUSTERING
st.header("🧩 Step 4: Run K-Means Clustering")

k = st.slider("Select number of clusters (k)", min_value=2, max_value=10, value=4)

if st.button("Run K-Means Clustering"):
    df_clustered, kmeans_model = perform_kmeans(X_scaled, df.copy(), k=k)
    st.success(f"✅ K-Means applied with {k} clusters!")

    # Show DataFrame with cluster labels
    st.subheader("Clustered Data (first 10 rows)")
    st.dataframe(df_clustered.head(10))

    # Scatter Plot of Clusters
    st.subheader("Cluster Visualization (GDP vs Life Expectancy)")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        x=df_clustered['GDP per capita'],
        y=df_clustered['Healthy life expectancy'],
        hue=df_clustered['Cluster'],
        palette='Set2',
        s=100,
        ax=ax
    )
    ax.set_xlabel("GDP per capita")
    ax.set_ylabel("Healthy life expectancy")
    ax.set_title("Country Clusters by GDP and Life Expectancy")
    st.pyplot(fig)

    # Cluster Summary
    st.subheader("📈 Cluster Summary")
    summary = summarize_clusters(df_clustered)
    st.dataframe(summary)

# STEP 5: HIERARCHICAL CLUSTERINGS
st.header("🌲 Step 5: Hierarchical Clustering (Dendrogram)")

if st.button("Show Hierarchical Dendrogram"):
    linked = linkage(X_scaled, method="ward")

    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(linked, labels=df["Country or region"].values, leaf_rotation=90, ax=ax)
    ax.set_title("Hierarchical Clustering Dendrogram")
    ax.set_xlabel("Country")
    ax.set_ylabel("Euclidean Distance")
    st.pyplot(fig)
