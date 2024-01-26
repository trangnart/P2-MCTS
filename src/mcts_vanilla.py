from mcts_node import MCTSNode
from p2_t3 import Board
from random import choice
from math import sqrt, log

num_nodes = 100
explore_faction = 2.

def traverse_nodes(node: MCTSNode, board: Board, state, bot_identity: int):
    """ Traverses the tree until the end criterion are met.
    e.g. find the best expandable node (node with untried action) if it exist,
    or else a terminal node

    Args:
        node:       A tree node from which the search is traversing.
        board:      The game setup.
        state:      The state of the game.
        identity:   The bot's identity, either 1 or 2

    Returns:
        node: A node from which the next stage of the search can proceed.
        state: The state associated with that node

    """
    while node.child_nodes and not node.untried_actions and not board.is_ended(state):
        is_opponent = board.current_player(state) != bot_identity

        best_node = max(node.child_nodes.values(), key=lambda x: ucb(x, is_opponent))
        node = best_node

        state = board.next_state(state, node.parent_action)

    return node, state

def expand_leaf(node: MCTSNode, board: Board, state):
    """ Adds a new leaf to the tree by creating a new child node for the given node (if it is non-terminal).

    Args:
        node:   The node for which a child will be added.
        board:  The game setup.
        state:  The state of the game.

    Returns:
        node: The added child node
        state: The state associated with that node

    """
    if len(node.untried_actions) > 0:
        action = node.untried_actions.pop(0)
        state = board.next_state(state, action)
        n = MCTSNode(parent=node,parent_action=action, action_list=board.legal_actions(state))
        node.child_nodes[action] = n

    return node, state

def rollout(board: Board, state):
    """ Given the state of the game, the rollout plays out the remainder randomly.

    Args:
        board:  The game setup.
        state:  The state of the game.

    Returns:
        state: The terminal game state

    """
    while not board.is_ended(state):
        legal_action = choice(board.legal_actions(state))
        state = board.next_state(state, legal_action)

    return state

def backpropagate(node: MCTSNode|None, won: bool):
    """ Navigates the tree from a leaf node to the root, updating the win and visit count of each node along the path.

    Args:
        node:   A leaf node.
        won:    An indicator of whether the bot won or lost the game.

    """
    if won:
        node.wins += 1
    node.visits += 1

    if node.parent is not None:
        backpropagate(node.parent, won)

def ucb(node: MCTSNode, is_opponent: bool):
    """ Calcualtes the UCB value for the given node from the perspective of the bot

    Args:
        node:   A node.
        is_opponent: A boolean indicating whether or not the last action was performed by the MCTS bot
    Returns:
        The value of the UCB function for the given node
    """
    if node.visits == 0:
        return float('inf')
    exploitation = node.wins / node.visits
    exploration = explore_faction * sqrt(log(node.parent.visits) / node.visits)
    return exploitation + (-exploration if is_opponent else exploration)

def get_best_action(root_node: MCTSNode):
    """ Selects the best action from the root node in the MCTS tree

    Args:
        root_node:   The root node
    Returns:
        action: The best action from the root node

    """
    valid_child_nodes = filter(lambda x: x[1].visits > 0, root_node.child_nodes.items())

    if not valid_child_nodes:
        return None

    best_action, best_win = max(valid_child_nodes, key=lambda x: x[1].wins / x[1].visits)
    return best_action

def is_win(board: Board, state, identity_of_bot: int):
    # checks if state is a win state for identity_of_bot
    outcome = board.points_values(state)
    assert outcome is not None, "is_win was called on a non-terminal state"
    return outcome[identity_of_bot] == 1

def think(board: Board, current_state):
    """ Performs MCTS by sampling games and calling the appropriate functions to construct the game tree.

    Args:
        board:  The game setup.
        current_state:  The current state of the game.

    Returns:    The action to be taken from the current state

    """
    bot_identity = board.current_player(current_state) # 1 or 2
    root_node = MCTSNode(parent=None, parent_action=None, action_list=board.legal_actions(current_state))

    for _ in range(num_nodes):
        state = current_state
        node = root_node

        leaf_node, state = traverse_nodes(node, board, state, bot_identity)

        expand_node, state = expand_leaf(leaf_node, board, state)

        state = rollout(board, state)

        won = is_win(board, state, bot_identity)
        backpropagate(expand_node, won)

    best_action = get_best_action(root_node)
    print(f"Action chosen: {best_action}")
    return best_action