import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Streamlit app
def main():
    st.title("KMeans Clustering App on Train data")

    # File uploader
    # uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
    # For testing locally, you can uncomment the next line and comment out the line above
    uploaded_file = "D:/python/projects/company assignment/train.xlsx"
    
    if uploaded_file is not None:
        # Load data
        df = pd.read_excel(uploaded_file)
        st.write("Data Preview:")
        st.write(df.head())

        # Extract features
        x = df.iloc[:, :18]

        # Standardizing the features
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        
        # Perform KMeans clustering
        num_clusters = st.slider("Select number of clusters", 2, 10, 2)
        if st.button("Train Model"):
            kmeans = KMeans(n_clusters=num_clusters, max_iter=300, random_state=42)
            kmeans.fit(x_scaled)
            df['Cluster'] = kmeans.labels_

            # Visualize clusters using PCA
            pca = PCA(n_components=2)
            pca_components = pca.fit_transform(x_scaled)
            df['PCA1'] = pca_components[:, 0]
            df['PCA2'] = pca_components[:, 1]

            st.write("Cluster Visualization:")
            fig, ax = plt.subplots()
            scatter = ax.scatter(df['PCA1'], df['PCA2'], c=df['Cluster'], cmap='rainbow')
            legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
            ax.add_artist(legend1)
            st.pyplot(fig)

            # Input new data point
            st.write("Input new data point to predict its cluster:")
            new_data_point = []
            new_data_point = [-76,-83,-70,-66,-64,-72,-64,-69,-60,-76,-83,-78,-81,-81,-81,-70,-60,-60]
            st.write("Predicting with New data points",new_data_point)
            # for i in range(18):
            #     value = st.number_input(f"Input value for feature {i+1}", value=0.0)
            #     new_data_point.append(value)

            # if st.button("Predict Cluster"):
            new_data_point_scaled = scaler.transform([new_data_point])
            predicted_cluster = kmeans.predict(new_data_point_scaled)
            st.write(f'The new data point belongs to cluster {predicted_cluster[0]}') 

if __name__ == "__main__":
    main()