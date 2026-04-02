import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load Dataset
data = pd.read_csv("Social_Network_Ads.csv")

# Select features
X = data[["Age", "EstimatedSalary"]]

# Data Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method
inertia = []
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure()
plt.plot(k_values, inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.grid()
plt.show()

# Applying K-Means (k=3)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

data["Cluster"] = clusters

# This is for PCA Visualization
pca = PCA(n_components=2)
pca_data = pca.fit_transform(X_scaled)

plt.figure()
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=clusters)
plt.title("PCA Cluster Visualization for Age and Salary")
plt.xlabel("Age and Salary PCA Component 1")
plt.ylabel("Age and Salary PCA Component 2")
plt.grid()
plt.show()

# It's the Cluster Summary
print("\nCluster Summary:")
print(data.groupby("Cluster")[["Age", "EstimatedSalary", "Purchased"]].mean())

for i in range(3):
    print(f"\nCluster {i} examples:")
    print(data[data["Cluster"] == i].head(2))