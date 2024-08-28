import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Custom weighted K-means implementation
class WeightedKMeans:
    def __init__(self, n_clusters, max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, sample_weight=None, initial_centroids=None):
        n_samples, n_features = X.shape

        # Initialize random weights if not provided
        if sample_weight is None:
            sample_weight = np.random.rand(n_samples)
        
        # Initialize centroids, using provided initial centroids or random points from dataset
        if initial_centroids is not None:
            centroids = initial_centroids
        else:
            initial_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
            centroids = X[initial_indices]

        for iteration in range(self.max_iter):
            # Compute distances and assign clusters
            distances = np.zeros((n_samples, self.n_clusters))
            for i, centroid in enumerate(centroids):
                distances[:, i] = np.linalg.norm(X - centroid, axis=1)

            # Assign clusters based on closest centroid
            labels = np.argmin(distances, axis=1)

            # Calculate new centroids with weights
            new_centroids = np.zeros((self.n_clusters, n_features))
            for i in range(self.n_clusters):
                cluster_points = X[labels == i]
                cluster_weights = sample_weight[labels == i]
                if np.sum(cluster_weights) > 0:
                    new_centroids[i] = np.average(cluster_points, axis=0, weights=cluster_weights)
                else:
                    new_centroids[i] = centroids[i]  # No change if no points are assigned

            # Check for convergence
            if np.all(np.linalg.norm(new_centroids - centroids, axis=1) < self.tol):
                break

            centroids = new_centroids

        self.cluster_centers_ = centroids
        self.labels_ = labels

        return self

# Define the grid size and scale
grid_size = (70, 70)
block_size = 100  # Each block represents 100 meters

# Initialize users
num_users = 50
positions = np.random.randint(0, grid_size[0], size=(num_users, 2))
directions = np.random.randint(0, 4, size=num_users)  # 0: up, 1: down, 2: left, 3: right
pause_times = np.zeros(num_users, dtype=int)
# Number of clusters
n_clusters = 2
# Define attraction points
num_attraction_points = 3
attraction_points = np.random.randint(0, grid_size[0], size=(num_attraction_points, 2))

# Define the number of time steps for the simulation
time_steps = 300  # Use 100 time steps for better observation

# Function to move a user in the current direction
def move(position, direction):
    if direction == 0 and position[0] > 0:  # up
        position[0] -= 1
    elif direction == 1 and position[0] < grid_size[0] - 1:  # down
        position[0] += 1
    elif direction == 2 and position[1] > 0:  # left
        position[1] -= 1
    elif direction == 3 and position[1] < grid_size[1] - 1:  # right
        position[1] += 1
    return position

# Function to change direction towards an attraction point, restricted to grid lines
def change_direction_towards(position, attraction_point):
    if position[0] < attraction_point[0]:
        return 1  # down
    elif position[0] > attraction_point[0]:
        return 0  # up
    elif position[1] < attraction_point[1]:
        return 3  # right
    elif position[1] > attraction_point[1]:
        return 2  # left
    return np.random.randint(0, 4)  # random direction if already at the attraction point

# Simulate the user movements
user_paths = np.zeros((num_users, time_steps, 2), dtype=int)
centroid_paths = np.zeros((time_steps, n_clusters, 2))  # To store the path of each centroid

for t in range(time_steps):
    for i in range(num_users):
        if pause_times[i] > 0:
            # User is pausing, decrement the pause time
            pause_times[i] -= 1
        else:
            # Move user in the current direction
            positions[i] = move(positions[i], directions[i])
            
            # Decide whether to change direction, pause, or move towards an attraction point
            if np.random.rand() < 0.1:  # 10% chance to change direction
                # Move towards the nearest attraction point with a 50% probability
                if np.random.rand() < 0.5:
                    distances = np.linalg.norm(attraction_points - positions[i], axis=1)
                    nearest_attraction_point = attraction_points[np.argmin(distances)]
                    directions[i] = change_direction_towards(positions[i], nearest_attraction_point)
                else:
                    directions[i] = np.random.randint(0, 4)
            elif np.random.rand() < 0.05:  # 5% chance to pause
                pause_times[i] = np.random.randint(1, 6)
        
        # Store the new position
        user_paths[i, t] = positions[i]


    # Fit K-means to get the centroid positions at each time step
    weighted_kmeans = WeightedKMeans(n_clusters)

    
    
    if t == 0:
        weighted_kmeans.fit(positions, sample_weight=np.random.rand(num_users))  # Initial fit with random weights
    else:
        # Fit with initial centroids from the previous timestep
        weighted_kmeans.fit(positions, sample_weight=np.random.rand(num_users), initial_centroids=centroid_paths[t-1])

    # Store the centroid positions
    centroid_paths[t] = weighted_kmeans.cluster_centers_



# Randomly assign initial weights to users
weights = np.random.rand(num_users)

# Prepare the plot
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(0, grid_size[1] - 1)
ax.set_ylim(0, grid_size[0] - 1)
ax.set_xticks(np.arange(0, grid_size[1], step=5))
ax.set_xticklabels(np.arange(0, grid_size[1] * block_size, step=5 * block_size))
ax.set_yticks(np.arange(0, grid_size[0], step=5))
ax.set_yticklabels(np.arange(0, grid_size[0] * block_size, step=5 * block_size))
ax.set_title('Animated Weighted K-Means Clustering on User Mobility')
ax.set_xlabel('X Position (in meters)')
ax.set_ylabel('Y Position (in meters)')
ax.grid(True)

# Plot attraction points
for ap in attraction_points:
    ax.plot(ap[1], ap[0], 'rx', markersize=10, label='Attraction Point')

# Initialize scatter plot for users
scatter = ax.scatter([], [], c=[], cmap='viridis', s=100, alpha=0.7)
# Initialize empty plot for cluster centers with a different marker (e.g., star)
cluster_centers_scatter = ax.scatter([], [], s=200, c='red', marker='*', label='Cluster Center')



# Update function for animation
def update(frame):
    # Determine current fractional frame to allow smooth interpolation
    frame_idx = frame // 10  # Base frame index
    alpha = (frame % 10) / 10.0  # Interpolation factor

    if frame_idx < time_steps - 1:
        current_positions = user_paths[:, frame_idx]
        next_positions = user_paths[:, frame_idx + 1]


        current_centroids = centroid_paths[frame_idx]
        next_centroids = centroid_paths[frame_idx + 1]



        # Interpolate positions
        interpolated_positions = (1 - alpha) * current_positions + alpha * next_positions

        # Interpolate centroids
        interpolated_centroids = (1 - alpha) * current_centroids + alpha * next_centroids

        # Use stored labels to maintain consistency in color
        previous_labels = weighted_kmeans.labels_
        print("================")
        print(current_positions)
        print(current_centroids)
        print(previous_labels)
        print("================")

        # Fit weighted K-means on interpolated positions using previous centroids
        weighted_kmeans.fit(interpolated_positions, sample_weight=weights, initial_centroids=current_centroids)
        
        # Update the scatter plot with new positions and colors
        scatter.set_offsets(interpolated_positions)
        scatter.set_array(previous_labels)

        # Update the cluster centers scatter
        cluster_centers_scatter.set_offsets(interpolated_centroids)

    return scatter, cluster_centers_scatter

# Create animation
ani = animation.FuncAnimation(fig, update, frames=time_steps * 10, interval=50, blit=True)

# Display animation
plt.show()