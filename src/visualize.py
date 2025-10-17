import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_clusters(df):
    """
    Returns a scatter plot figure of clusters based on GDP and Life Expectancy.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        x=df['GDP per capita'],
        y=df['Healthy life expectancy'],
        hue=df['Cluster'],
        palette='Set2',
        s=100,
        ax=ax
    )
    ax.set_title('Country Clusters by GDP and Life Expectancy')
    ax.set_xlabel('GDP per capita')
    ax.set_ylabel('Healthy life expectancy')
    return fig


def summarize_clusters(df):
    """
    Returns cluster-wise summary statistics.
    """
    summary = df.groupby('Cluster').mean(numeric_only=True)
    return summary
