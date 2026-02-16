# EVOLVE-BLOCK-START
import random

import numpy as np

from env import Action


def get_best_move(board: np.ndarray):
    """
    The board is presented as a 4x4 array.
    Each element is either 0 (empty) or the power of 2 representing the tile value.
    For example, a tile with value 8 is represented as 3 (2^3),
    The function should return one of the Action enum values
    (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT)
    indicating which move to make.

    Args:
        board: np.ndarray of shape (4, 4) representing the current game state
    Returns:
        An Action enum value indicating the chosen move
    """
    # randomly select move
    return random.choice(list(Action))


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
"""
Finding the shortest sequence for a specific seed of the famous 2048 game
see discussion on hand-coded heuristic approaches
https://stackoverflow.com/questions/22342854/what-is-the-optimal-algorithm-for-the-game-2048
"""

from env import play_2048


def run_2048(*args, **kwargs):
    return play_2048(get_best_move)


if __name__ == "__main__":
    from env import render_str

    boards, actions, max_val_reached, reached_2048, reached_max_steps, is_timed_out = (
        run_2048()
    )
    for board, action in zip(boards, actions):
        print(action)
        render_str(board)
        print("=" * 60)

    print(f"{len(actions)=}")
    print(f"{max_val_reached=}")
    print(f"{reached_max_steps=}")
    print(f"{reached_2048=}")
    print(f"{is_timed_out=}")
