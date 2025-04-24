import numpy as np
import matplotlib.pyplot as plt

# Load the data
agent_trajectories = np.load('agent_trajectories.npy')  # Shape: (num_agents, max_episodes, steps_per_episode, 2)
episode_rewards = np.load('episode_rewards.npy')  # Shape: (max_episodes,)

num_agents = agent_trajectories.shape[0]
max_episodes = agent_trajectories.shape[1]
Tpos = np.array([[1430, 1200], [1000, 800], [500, 400], [200, 1500]])  # Target positions

# --- Plotting the Trajectories in the Last Episode ---
plt.figure(figsize=(10, 8))
plt.title(f'Trajectories in Last Episode (Episode {max_episodes - 10})')
plt.xlabel('X Position')
plt.ylabel('Y Position')

for agent_id in range(num_agents):
    last_episode_trajectory = agent_trajectories[agent_id, max_episodes - 10]  # Get trajectory from the last episode
    plt.plot(
        last_episode_trajectory[:, 0], last_episode_trajectory[:, 1],
        label=f'Agent {agent_id + 1}'
    )

# Plot target locations
plt.scatter(Tpos[:, 0], Tpos[:, 1], color='red', marker='o', s=50, label='Targets')  # Plot targets

plt.legend()
plt.grid(True)
plt.show()

# --- Plotting Rewards Over Episodes ---
plt.figure(figsize=(10, 6))
plt.plot(episode_rewards)
plt.title('Total Rewards per Episode')
plt.xlabel('Episode Number')
plt.ylabel('Total Reward')
plt.grid(True)
plt.show()