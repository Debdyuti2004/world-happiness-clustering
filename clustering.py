# clustering.py
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
import seaborn as sns

def find_optimal_clusters(X_scaled, max_k=10):
    """
    Uses the Elbow Method to determine optimal number of clusters.
    """
    wcss = []
    for i in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
    
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, max_k + 1), wcss, marker='o')
    plt.title('Elbow Method - Optimal k')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.show()

def perform_kmeans(X_scaled, df, k=4):
    """
    Performs KMeans clustering and adds cluster labels to dataframe.
    """
    kmeans = KMeans(n_clusters=k, random_state=42)
    y_kmeans = kmeans.fit_predict(X_scaled)
    df['Cluster'] = y_kmeans
    print(f"✅ K-Means clustering done with {k} clusters!")
    return df, kmeans

def perform_hierarchical(X_scaled, df):
    """
    Performs hierarchical clustering and plots dendrogram.
    """
    linked = linkage(X_scaled, method='ward')
    plt.figure(figsize=(10, 6))
    dendrogram(linked, labels=df['Country or region'].values, leaf_rotation=90)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Country')
    plt.ylabel('Euclidean Distance')
    plt.show()
