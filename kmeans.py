import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.cluster import KMeans

# Step 1: Generate 50 random points within a 70x70 grid
num_nodes = 50
grid_size = 70
np.random.seed(0)
X = np.random.rand(num_nodes, 2) * grid_size
weights = np.random.rand(num_nodes)  # Generate random weights for the data points

# Step 2: Implement Weighted K-means clustering with more balanced cluster weights
class BalancedWeightedKMeans:
    def __init__(self, n_clusters, max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, X, sample_weight=None):
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])
        
        self.kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', max_iter=1, tol=self.tol, random_state=0, n_init=1)
        self.cluster_centers_ = self.kmeans.fit(X, sample_weight=sample_weight).cluster_centers_
        self.labels_ = self.kmeans.labels_
        
        for i in range(self.max_iter - 1):
            self.kmeans = KMeans(n_clusters=self.n_clusters, init=self.cluster_centers_, max_iter=1, tol=self.tol, random_state=0, n_init=1)
            self.cluster_centers_ = self.kmeans.fit(X, sample_weight=sample_weight).cluster_centers_
            self.labels_ = self.kmeans.labels_

            # Adjust clusters to balance weights
            cluster_weights = self.calculate_cluster_weights(X, sample_weight)
            max_weight = np.max(cluster_weights)
            min_weight = np.min(cluster_weights)
            
            if max_weight - min_weight <= self.tol:
                break

            yield self.labels_, self.cluster_centers_

    def calculate_cluster_weights(self, X, sample_weight):
        cluster_weights = np.zeros(self.n_clusters)
        for i in range(self.n_clusters):
            cluster_weights[i] = np.sum(sample_weight[self.labels_ == i])
        return cluster_weights

# Define the number of clusters
num_clusters = 7

# Initialize the Balanced Weighted K-means model
balanced_weighted_kmeans = BalancedWeightedKMeans(n_clusters=num_clusters)

# Step 3: Create the animation
fig, ax = plt.subplots()

def update(frame):
    labels, centers = frame
    ax.clear()
    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', s=weights*100, alpha=0.6)  # Scale marker size by weight
    ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=100, label='Centers')
    ax.set_title('Balanced Weighted K-means Clustering')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.legend()

# Calculate the number of frames for the animation
frames = list(balanced_weighted_kmeans.fit(X, sample_weight=weights))

ani = animation.FuncAnimation(fig, update, frames=frames, interval=1000,  cache_frame_data=False)

plt.show()