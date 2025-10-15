# main.py
from src.preprocessing import load_and_clean_data, scale_features
from src.clustering import find_optimal_clusters, perform_kmeans, perform_hierarchical
from src.visualize import plot_clusters, summarize_clusters

# -----------------------------
# STEP 1: Load and preprocess data
# -----------------------------
print("🔹 Step 1: Loading and Preprocessing Dataset...")
df = load_and_clean_data()
X_scaled, scaler = scale_features(df)

# -----------------------------
# STEP 2: Find optimal number of clusters
# -----------------------------
print("\n🔹 Step 2: Finding Optimal Number of Clusters...")
find_optimal_clusters(X_scaled, max_k=10)

# -----------------------------
# STEP 3: Apply K-Means
# -----------------------------
k = 4  # You can adjust after checking elbow plot
print(f"\n🔹 Step 3: Applying K-Means with k={k}...")
df, kmeans_model = perform_kmeans(X_scaled, df, k=k)

# -----------------------------
# STEP 4: Visualize and Analyze
# -----------------------------
print("\n🔹 Step 4: Visualizing Clusters and Summarizing Data...")
plot_clusters(df)
summary = summarize_clusters(df)

# -----------------------------
# STEP 5 (Optional): Hierarchical Clustering
# -----------------------------
print("\n🔹 Step 5: Performing Hierarchical Clustering...")
perform_hierarchical(X_scaled, df)

