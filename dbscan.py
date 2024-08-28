import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.cluster import DBSCAN

# Define the grid size and scale
grid_size = (70, 70)
block_size = 100  # Each block represents 100 meters

# Initialize users (e.g., 50 users)
num_users = 50
positions = np.random.randint(0, grid_size[0], size=(num_users, 2))
directions = np.random.randint(0, 4, size=num_users)  # 0: up, 1: down, 2: left, 3: right
pause_times = np.zeros(num_users, dtype=int)

# Define attraction points
num_attraction_points = 3
attraction_points = np.random.randint(0, grid_size[0], size=(num_attraction_points, 2))

# Define the number of time steps for the simulation
time_steps = 300  # Use 300 time steps for better observation

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
cluster_labels = np.full((time_steps, num_users), -1)  # To store the labels of each user

# Randomly assign initial weights to users
weights = np.random.rand(num_users)

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

    # Fit DBSCAN to get the cluster labels at each time step
    dbscan = DBSCAN(eps=5, min_samples=2)
    dbscan.fit(positions)
    cluster_labels[t] = dbscan.labels_

# Prepare the plot
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(0, grid_size[1] - 1)
ax.set_ylim(0, grid_size[0] - 1)
ax.set_xticks(np.arange(0, grid_size[1], step=5))
ax.set_xticklabels(np.arange(0, grid_size[1] * block_size, step=5 * block_size))
ax.set_yticks(np.arange(0, grid_size[0], step=5))
ax.set_yticklabels(np.arange(0, grid_size[0] * block_size, step=5 * block_size))
ax.set_title('Animated DBSCAN Clustering on User Mobility')
ax.set_xlabel('X Position (in meters)')
ax.set_ylabel('Y Position (in meters)')
ax.grid(True)

# Plot attraction points
for ap in attraction_points:
    ax.plot(ap[1], ap[0], 'rx', markersize=10, label='Attraction Point')

# Initialize scatter plot for users
scatter = ax.scatter([], [], c=[], cmap='viridis', s=100, alpha=0.7)

# Update function for animation
def update(frame):
    # Determine current fractional frame to allow smooth interpolation
    frame_idx = frame // 10  # Base frame index
    alpha = (frame % 10) / 10.0  # Interpolation factor

    if frame_idx < time_steps - 1:
        current_positions = user_paths[:, frame_idx]
        next_positions = user_paths[:, frame_idx + 1]

        # Interpolate positions
        interpolated_positions = (1 - alpha) * current_positions + alpha * next_positions

        # Use stored labels to maintain consistency in color
        labels = cluster_labels[frame_idx]

        # Update the scatter plot with new positions and colors
        scatter.set_offsets(interpolated_positions)
        scatter.set_array(labels)

    return scatter,

# Create animation
ani = animation.FuncAnimation(fig, update, frames=time_steps * 10, interval=50, blit=True)

# Display animation
plt.show()