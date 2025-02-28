'''
This example uses the FrozenLake-v1 environment from the gym library.
The Q-Learning algorithm updates the Q-table based on the agent's experiences,
and the learned policy is tested at the end.
'''
import numpy as np
import gym

# Initialize the environment
env = gym.make('FrozenLake-v1', is_slippery=False)

# Set parameters
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 0.1  # Exploration rate
num_episodes = 1000

# Initialize Q-table
Q = np.zeros((env.observation_space.n, env.action_space.n))

# Q-Learning algorithm
for episode in range(num_episodes):
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]  # Extract the first element if state is a tuple
    state = int(state)  # Ensure state is an integer
    done = False

    while not done:
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore action space
        else:
            action = np.argmax(Q[state, :])  # Exploit learned values

        next_state, reward, done, *_ = env.step(action)
        if isinstance(next_state, tuple):
            next_state = next_state[0]  # Extract the first element if next_state is a tuple
        next_state = int(next_state)  # Ensure next_state is an integer

        # Update Q-table
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])

        state = next_state

# Display the learned Q-values
print("Learned Q-values:")
print(Q)

# Test the learned policy
state = env.reset()
if isinstance(state, tuple):
    state = state[0]  # Extract the first element if state is a tuple
state = int(state)  # Ensure state is an integer
env.render()
done = False

while not done:
    action = np.argmax(Q[state, :])
    state, reward, done, *_ = env.step(action)
    if isinstance(state, tuple):
        state = state[0]  # Extract the first element if state is a tuple
    state = int(state)  # Ensure state is an integer
    env.render()