import numpy as np
import random
import matplotlib.pyplot as plt

dimensions = (4, 4)
initial_state = (0, 0)
goal_state = (3, 3)
actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
symbol_actions = ['↑', '↓', '←', '→']

# Q-table initialization: similar to q-learning sample, but with no obstacles
num_states = dimensions[0] * dimensions[1]
num_actions = len(actions)
Q = np.zeros((num_states, num_actions))

'''
SARSA parameters:
    - alpha = 0.1. Learning rate. How much info overrides old info. 
        High alpha = faster learning, less stable
        Low alpha = slower learning, more stable
    - gamma = 0.99. Discount factor. Importance of future rewards, how much the agent values future rewards compared to immediate rewards.
        High gamma: Future rewards are more important, promoting long-term strategies.
        Low gamma: Immediate rewards are more important, promoting short-term strategies. 
    - epsilon = 0.2. Exploration rate. Probability of choosing a random action instead of the best one.
        High epsilon: More exploration, less exploitation.
        Low epsilon: Less exploration, more exploitation.
    - episodes = 1000. Number of training episodes the agent will go through
        High episodes: More training opportunities, potentially better policy.
        Low episodes: Fewer training opportunities, potentially less effective policy.        
'''

alpha = 0.1
gamma = 0.99
epsilon = 0.2
episodes = 1000

#state to a linear index: state consists of a tuple (x, y), but algorithm uses a unique index in Q-table
def state_to_index(state):
    return state[0] * dimensions[1] + state[1]

# determines if the action will be by exploration or exploitation
def select_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, num_actions - 1)
    else:
        return np.argmax(Q[state_to_index(state)])

# apply action
def apply_accion(estado, action_idx):
    action = actions[action_idx]
    new_state = tuple(np.add(estado, action) % np.array(dimensions))

    if new_state == goal_state:
        reward = 1
    else:
        reward = -1

    return new_state, reward, new_state == goal_state
