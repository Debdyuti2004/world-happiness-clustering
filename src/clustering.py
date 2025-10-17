import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram

def find_optimal_clusters(X_scaled, max_k=10):
    """
    Returns the elbow plot figure for optimal clusters.
    """
    wcss = []
    for i in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, max_k + 1), wcss, marker='o')
    ax.set_title('Elbow Method - Optimal k')
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('WCSS')
    return fig

def perform_kmeans(X_scaled, df, k=4):
    """
    Performs KMeans clustering and adds cluster labels to dataframe.
    Returns df with cluster labels and the KMeans model.
    """
    kmeans = KMeans(n_clusters=k, random_state=42)
    y_kmeans = kmeans.fit_predict(X_scaled)
    df['Cluster'] = y_kmeans
    return df, kmeans

def perform_hierarchical(X_scaled, df):
    """
    Performs hierarchical clustering and returns dendrogram figure.
    """
    linked = linkage(X_scaled, method='ward')
    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(linked, labels=df['Country or region'].values, leaf_rotation=90, ax=ax)
    ax.set_title('Hierarchical Clustering Dendrogram')
    ax.set_xlabel('Country')
    ax.set_ylabel('Euclidean Distance')
    return fig