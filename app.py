import streamlit as st 
import pickle 
import matplotlib.pyplot as plt

# Load model
with open('kmeans_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
    
# Set the page config
st.set_page_config(page_title='K-means Clustering App', layout='centered')

# Set title
st.title('K-means Clustering Visualizer by Boonyawat Jitratthanasaweat 6531501074')

# Display cluster centers
st.subheader('Example Data for Visualization')
st.markdown('This demo uses example data (2D) to illustrate clustering results.') 

# Load from a served dataset or generate synthetic data
from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=300, centers=loaded_model.n_clusters, cluster_std=0.60, random_state=0)

# Predict using the loaded model 
y_kmeans = loaded_model.predict(X) 

# Plotting
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
plt.scatter(loaded_model.cluster_centers_[:, 0], loaded_model.cluster_centers_[:, 1], s=300, c='red')
plt.title('k-Means Clustering')

# Show the plot in Streamlit
st.pyplot(plt)
