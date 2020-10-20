import random

from environment import utils
from environment.PursuitState import PursuitState
import numpy as np

from environment.utils import agent_directions


def build_mdp_features(state, joint_actions):

    """Features used for environment models"""

    coordinates = PursuitState.features(state)
    actions_one_hot = actions_one_hot_encoding(joint_actions)

    # St + At (one hot encoded)
    mdp_features = np.concatenate((coordinates, actions_one_hot))

    return mdp_features


def transition_network_features(state, joint_actions, entity):

    total_entities = state.num_agents + 1
    prey = total_entities - 1

    identifier = np.zeros(total_entities)
    identifier[entity] = 1.0

    state_features = state.features_relative_agent(entity) if entity != prey else state.features_relative_prey()
    actions_one_hot = actions_one_hot_encoding(joint_actions)
    features = np.concatenate((identifier, state_features, actions_one_hot))

    return features


def actions_one_hot_encoding(joint_actions):
    """One hot encoding for all actions"""
    num_agents = len(joint_actions)
    num_actions = len(agent_directions())
    actions_one_hot = np.zeros((num_agents, num_actions))
    actions_one_hot[range(num_agents), joint_actions] = 1
    actions_one_hot = actions_one_hot.reshape((num_agents * num_actions,))
    return actions_one_hot

def compute_direction(state, next_state, entity):
    w, h = state.world_size
    state = state.features()
    next_state = next_state.features()
    source = state[entity*2], state[entity*2+1]
    target = next_state[entity*2], next_state[entity*2+1]
    direction = utils.direction(source, target, w, h)
    return direction


def remove_collisions(state, next_state):

    """Randomly rolls back colliding agents in a state st+1 back to their original positions in state st"""

    vacancy_constant = state.world_size[0] + 99

    old_positions = state.features()
    positions = next_state.features()
    num_agents = 4

    indices = list(range(num_agents + 1))
    random.shuffle(indices)

    collision = False

    for target in indices:

        x1 = positions[target * 2 + 0]
        y1 = positions[target * 2 + 1]

        for other in range(num_agents + 1):

            if target == other:
                continue

            x2 = positions[other * 2 + 0]
            y2 = positions[other * 2 + 1]

            collision = x2 == x1 and y2 == y1

            if collision:
                break

        positions = roll_back(target, positions, old_positions, vacancy_constant) if collision else positions

    corrected_next_state = PursuitState.from_features(positions, state.world_size)

    return corrected_next_state


def roll_back(agent, positions, old_positions, vacancy_placeholder):

    positions[agent * 2 + 0] = vacancy_placeholder
    positions[agent * 2 + 1] = vacancy_placeholder

    oldx = old_positions[agent * 2 + 0]
    oldy = old_positions[agent * 2 + 1]

    num_agents = 4

    occupied = False
    current_occupant = -1

    for other in range(num_agents + 1):

        if agent == other:
            continue

        x_2 = positions[other * 2 + 0]
        y_2 = positions[other * 2 + 1]
        occupied = oldx == x_2 and oldy == y_2

        if occupied:
            current_occupant = other
            break

    positions = roll_back(current_occupant, positions, old_positions, vacancy_placeholder) if occupied else positions

    positions[agent * 2 + 0] = oldx
    positions[agent * 2 + 1] = oldy

    return positions


def compute_displacement(state, next_state, entity):

    """Computes displacement for all agents, i.e., st+1 - st"""

    num_actions = len(agent_directions())
    movement_vector = np.zeros((1, num_actions + 1))

    num_agents = len(state.agents_positions)
    is_prey = entity == num_agents

    if is_prey:
        x = state.prey_positions[0][0]
        y = state.prey_positions[0][1]
        x_new = next_state.prey_positions[0][0]
        y_new = next_state.prey_positions[0][1]
    else:
        x = state.agents_positions[entity][0]
        y = state.agents_positions[entity][1]
        x_new = next_state.agents_positions[entity][0]
        y_new = next_state.agents_positions[entity][1]

    columns = state.world_size[0]
    rows = state.world_size[1]

    moves = [(x_new - x) % columns == 1,    # Moved Right
             (x - x_new) % columns == 1,    # Moved Left
             (y_new - y) % rows == 1,       # Moved Down
             (y - y_new) % rows == 1]       # Moved Up

    stayed_in_place = True not in moves

    if stayed_in_place:
        movement_vector[0][0] = 1
    else:
        # Find out which way the entity went (R, L, D or U)
        for a in range(num_actions):
            if moves[a]:
                movement_vector[0][a + 1] = 1
                break

    return movement_vector


def infer_prey_action(state, next_state):
    """
    Returns the action for the prey (0, 1, 2, 3, 4, 5)
     """
    num_actions = len(agent_directions())

    x = state.prey_positions[0][0]
    y = state.prey_positions[0][1]

    x_new = next_state.prey_positions[0][0]
    y_new = next_state.prey_positions[0][1]

    columns = state.world_size[0]
    rows = state.world_size[1]

    moves = [(x_new - x) % columns == 1,    # Moved Right
             (x - x_new) % columns == 1,    # Moved Left
             (y_new - y) % rows == 1,       # Moved Down
             (y - y_new) % rows == 1]       # Moved Up

    stayed_in_place = True not in moves

    if stayed_in_place:
        return 0
    else:  # Find out which way the prey went (R, L, D or U)
        for a in range(num_actions):
            if moves[a]: return a + 1
