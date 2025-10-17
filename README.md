# 🌍 World Happiness Clustering

This project performs unsupervised machine learning on the World Happiness Report dataset to uncover patterns among countries based on key happiness indicators.  
It uses both **K-Means** and **Hierarchical Clustering** to group countries with similar socio-economic and well-being characteristics, and provides both a **Python script** and an **interactive Streamlit dashboard** for exploration.

---

## 📊 Overview
The World Happiness Report ranks countries by their happiness scores using several key features:

- GDP per capita  
- Social support  
- Healthy life expectancy  
- Freedom to make life choices  
- Perceptions of corruption  

This project identifies clusters of countries that share similar happiness characteristics and visualizes the results for meaningful insights.

---

## 🧠 Features
✅ Data loading and preprocessing  
✅ Feature scaling using StandardScaler  
✅ Optimal cluster selection using Elbow Method  
✅ K-Means clustering implementation  
✅ Hierarchical clustering with dendrograms  
✅ Cluster visualization and summary statistics  
✅ **Interactive Streamlit dashboard (`app.py`) for exploration**

---

## 🏗️ Project Structure

DMDW_Project/
│
├── data/
│ └── world_happiness.csv # Dataset file
│
├── src/
│ ├── preprocessing.py # Data cleaning & feature scaling
│ ├── clustering.py # KMeans & Hierarchical clustering
│ ├── visualize.py # Plots and cluster summaries
│
├── app.py # Streamlit dashboard
├── main.py # Runs clustering & visualizations via console
├── requirements.txt # Required libraries
└── README.md # Project documentation


---

## ⚙️ Installation

1. **Clone this repository**
   ```bash
   git clone https://github.com/Debdyuti2004/world-happiness-clustering.git
   cd world-happiness-clustering

2. Install dependencies

pip install -r requirements.txt


Usage
Run the project from the root directory:
python main.py

Run Streamlit dashboard (app.py):
streamlit run app.py


Upload your own CSV or use the default dataset.

Select number of clusters using sliders.

Visualize K-Means clusters and hierarchical dendrogram interactively.
OUTPUT:

Cluster Summary:

         GDP per capita  Social support  ...  Freedom to make life choices  Perceptions of corruption    
Cluster                                  ...
0              0.966704        1.300222  ...                      0.476926                   0.079241    
1              1.076647        1.306765  ...                      0.257412                   0.054000    
2              1.377115        1.484500  ...                      0.536154                   0.268654    
3              0.395000        0.841333  ...                      0.304643                   0.098905    

📦 Dependencies

Python 3.8+
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy

🧩 Methods Used

StandardScaler – to normalize feature scales
KMeans (from sklearn.cluster) – to partition countries into clusters
Agglomerative Clustering – for hierarchical analysis
Elbow Method – to find optimal cluster count
Matplotlib & Seaborn – for plots and visual summaries

💡 Insights

Cluster 2 has the highest values for GDP, social support, and freedom → likely represents developed, high-happiness countries.
Cluster 3 has the lowest GDP and social support → likely represents low-happiness or developing countries.
The other clusters (0, 1) are in between, showing reasonable variation.

📜 License

This project is open-source under the MIT License.
You’re free to use, modify, and distribute it for educational or research purposes.

✨ Author

Debdyuti Chakraborty
💻 Computer Science Engineering Student At KIIT University
