# Combine customer profile and transaction data
clustering_data = customer_features.copy()
clustering_data.drop(columns=['CustomerID'], inplace=True)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(clustering_data)

# Add cluster labels to the dataset
customer_features['Cluster'] = clusters

# Evaluate clustering using Davies-Bouldin Index
db_index = davies_bouldin_score(clustering_data, clusters)
print(f"Davies-Bouldin Index: {db_index}")

# Visualize clusters using PCA
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(clustering_data)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, cmap='viridis')
plt.title('Customer Clusters (PCA Visualization)')
plt.show()