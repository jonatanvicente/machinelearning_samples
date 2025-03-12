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
def apply_action(state, action_idx):
    action = actions[action_idx]
    # np.add => suma el estado actual con la acción que se va a tomar
    #       si state = tupla con posición actual en cuadrícula (x,y) y action = tupla con acción a tomar (-1,0, p.ej para moverse hacia arriba)
    #       resultado será la nueva posición del agente
    # np.array => si el agente se mueve más allá de los límites de la cuadrícula, se aparecerá por el lado opuesto. Convierte las dimensiones
    #       de la cuadrícula en un array para poder hacer la operación de módulo (aplica elemento por elemento)
    # tuple => convierte el array resultante en una tupla np.add() devuelve un array, y las operaciones siguientes también devuelven arrays
    new_state = tuple(np.add(state, action) % np.array(dimensions))

    if new_state == goal_state:
        reward = 1
    else:
        reward = -1

    return new_state, reward, new_state == goal_state

# training
for episodio in range(episodes):
    state = initial_state
    action_idx = select_action(state)
    ended = False

    while not ended:
        new_state, reward, ended = apply_action(state, action_idx)
        new_action_idx = select_action(new_state)

        index = state_to_index(state)
        Q[index, action_idx] += alpha * (
                    reward + gamma * Q[state_to_index(new_state), new_action_idx] - Q[index, action_idx])

        state, action_idx = new_state, new_action_idx

# Visualize
polithic_symbols = np.empty(dimensions, dtype='<U2')

for i in range(dimensions[0]):
    for j in range(dimensions[1]):
        state = (i, j)
        better_action = np.argmax(Q[state_to_index(state)])
        polithic_symbols[i, j] = symbol_actions[better_action]

print(polithic_symbols) # better movement for each state

