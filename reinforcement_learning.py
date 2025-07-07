# Implements Q-learning for a simple grid world environment

import numpy as np

# Define grid world
GRID_SIZE = 4
ACTIONS = ['up', 'down', 'left', 'right']
REWARDS = np.zeros((GRID_SIZE, GRID_SIZE))
REWARDS[3, 3] = 1  # Goal state
Q_TABLE = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))

# Q-learning parameters
LEARNING_RATE = 0.8
DISCOUNT_FACTOR = 0.95
EPISODES = 1000

# Simple environment transition
def get_next_state(state, action):
    i, j = state
    if action == 'up' and i > 0: i -= 1
    elif action == 'down' and i < GRID_SIZE - 1: i += 1
    elif action == 'left' and j > 0: j -= 1
    elif action == 'right' and j < GRID_SIZE - 1: j += 1
    return (i, j)

# Q-learning algorithm
for _ in range(EPISODES):
    state = (0, 0)
    while state != (3, 3):
        action_idx = np.argmax(Q_TABLE[state[0], state[1]])
        next_state = get_next_state(state, ACTIONS[action_idx])
        reward = REWARDS[next_state[0], next_state[1]]
        Q_TABLE[state[0], state[1], action_idx] += LEARNING_RATE * (
            reward + DISCOUNT_FACTOR * np.max(Q_TABLE[next_state[0], next_state[1]]) - 
            Q_TABLE[state[0], state[1], action_idx]
        )
        state = next_state

# Print optimal policy
for i in range(GRID_SIZE):
    for j in range(GRID_SIZE):
        if (i, j) == (3, 3):
            print("G", end=" ")
        else:
            best_action = ACTIONS[np.argmax(Q_TABLE[i, j])]
            print(best_action[0], end=" ")
    print()