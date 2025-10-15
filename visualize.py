# visualize.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_clusters(df):
    """
    Visualize clusters based on GDP and Life Expectancy.
    """
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        x=df['GDP per capita'],
        y=df['Healthy life expectancy'],
        hue=df['Cluster'],
        palette='Set2',
        s=100
    )
    plt.title('Country Clusters by GDP and Life Expectancy')
    plt.xlabel('Log GDP per capita')
    plt.ylabel('Healthy life expectancy')
    plt.show()

def summarize_clusters(df):
    """
    Prints and returns cluster-wise summary statistics.
    """
    summary = df.groupby('Cluster').mean(numeric_only=True)
    print("\n📊 Cluster Summary:\n")
    print(summary)
    return summary
