# main.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from src.preprocessing import load_and_clean_data, scale_features
from src.clustering import perform_kmeans, perform_hierarchical
from src.visualize import plot_clusters, summarize_clusters

def main():
    # STEP 1: Load & Clean Data
    df = load_and_clean_data()
    print("\n✅ Head of the dataset:")
    print(df.head())

    # STEP 2: Scale Features
    X_scaled, scaler, numeric_cols = scale_features(df)
    print(f"Scaled columns: {numeric_cols}")

    # STEP 3: Elbow Method for Optimal k
    max_k = 10  # can be changed
    wcss = []
    for i in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, max_k + 1), wcss, marker='o')
    plt.title("Elbow Method - Optimal k")
    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS")
    plt.show()

    print("\n🔹 Look at the elbow plot to decide the optimal number of clusters (k).")
    
    # STEP 4: K-Means Clustering
    k = int(input("\nEnter number of clusters (k) based on elbow plot: "))
    df_clustered, kmeans_model = perform_kmeans(X_scaled, df.copy(), k=k)
    print(f"\n✅ K-Means applied with {k} clusters!")
    
    # Cluster Summary
    summary = summarize_clusters(df_clustered)
    print("\n📈 Cluster Summary:")
    print(summary)

    # STEP 5: Scatter Plot of Clusters
    fig1 = plot_clusters(df_clustered)
    fig1.show()

    # STEP 6: Hierarchical Clustering
    fig2 = perform_hierarchical(X_scaled, df)
    fig2.show()

if __name__ == "__main__":
    main()
