import random
import signal
from enum import Enum
from typing import Callable, Any

import numpy as np


def play_2048(get_best_move: Callable):
    env = Env2048(max_steps=None)
    board = env.reset(seed=42)
    done = False
    reward = 0
    boards = [board]
    actions = []
    get_best_move_fn = time_limit(seconds=0.11)(get_best_move)
    while not done:
        selected_move, is_timed_out = get_best_move_fn(board)
        if is_timed_out:
            return boards, actions, env.max_value_reached, False, False, is_timed_out

        board, reward, done = env.step(selected_move)
        boards.append(board)
        actions.append(selected_move)
        if reward > 0 or done:
            break

    reached_max_steps = len(actions) == env.max_steps
    reached_2048 = reward
    return (
        boards,
        actions,
        env.max_value_reached,
        reached_2048,
        reached_max_steps,
        is_timed_out,
    )


def time_limit(seconds: float = 1.0, fallback: Callable[[], Any] | None = None):
    """
    Decorator that enforces a wall-clock time limit on a function using UNIX signals.
    Supports sub-second precision via setitimer.

    Behavior:
    - If the function exceeds `seconds`, it is interrupted and the wrapper returns
        a tuple (fallback_result, True). Otherwise, it returns (result, False).

    Notes:
        - Uses signal.SIGALRM with signal.setitimer (Linux/Unix only) and must run on the main thread.
    - To keep the 2048 game running, we default to a random valid Action when timed out.
    """

    class TimeoutErrorInternal(Exception):
        pass

    def _handler(signum, frame):  # type: ignore[no-untyped-def]
        raise TimeoutErrorInternal()

    def _decorator(func: Callable):
        def _wrapped(*args, **kwargs):  # type: ignore[no-untyped-def]
            old_handler = signal.signal(signal.SIGALRM, _handler)
            timed_out = False
            try:
                # Use ITIMER_REAL to allow sub-second timeouts
                timeout = float(seconds)
                if timeout > 0:
                    signal.setitimer(signal.ITIMER_REAL, timeout)
                result = func(*args, **kwargs)
                return result, timed_out
            except TimeoutErrorInternal:
                timed_out = True
                # Choose fallback result if provided; otherwise random Action to keep env valid
                if fallback is not None:
                    fb = fallback()
                else:
                    fb = random.choice(list(Action))
                return fb, timed_out
            finally:
                # Cancel timer and restore previous handler
                try:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                except Exception:
                    # Fallback to alarm(0) if setitimer unavailable
                    try:
                        signal.alarm(0)
                    except Exception:
                        pass
                signal.signal(signal.SIGALRM, old_handler)

        return _wrapped

    return _decorator


def render_str(board: np.ndarray):
    """
    Print a human-readable representation of the board.
    """
    out = ""
    for i in range(4):
        row = []
        for j in range(4):
            if board[i, j] == 0:
                row.append("    .")
            else:
                value = 2 ** board[i, j]
                row.append(f"{value:5d}")
        out += "|" + "|".join(row) + "|\n"
    print(out)
    print("-" * 25)
    print()
    return out


class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class Env2048:
    """
    2048 Game Environment

    The board is represented as a 4x4 numpy array where each element stores
    the power of 2. For example:
    - 0 represents an empty tile
    - 1 represents 2^1 = 2
    - 2 represents 2^2 = 4
    - 3 represents 2^3 = 8
    - etc.

    This representation is more efficient than storing actual values.
    """

    def __init__(self, seed=None, max_steps=None):
        """
        Initialize the 2048 game environment.

            Args:
                seed: Random seed for reproducibility
                max_steps: Maximum number of steps allowed per episode (None = unlimited)
        """

        self.board = np.zeros((4, 4), dtype=np.int32)
        self.score = 0
        self.game_over = False
        self.max_steps = max_steps
        self.current_step = 0

    def reset(self, seed: int | None = None):
        """
        Reset the game and return the initial board state.

        Returns:
            np.ndarray: The initial 4x4 board with 2 random tiles
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.board = np.zeros((4, 4), dtype=np.int32)
        self.score = 0
        self.game_over = False
        self.current_step = 0

        # Add two initial tiles
        self._add_random_tile()
        self._add_random_tile()

        return self.board.copy()

    def step(self, action: Action):
        """
        Execute one step of the game.

        Args:
            action: Action enum (UP, DOWN, LEFT, RIGHT)

        Returns:
            tuple: (board, reward, done)
                - board: np.ndarray of shape (4, 4)
                - reward: 1 if 2048 tile is reached, 0 otherwise
                - done: bool, whether the game is over or 2048 is reached
        """
        if self.game_over:
            return self.board.copy(), 0, True

        # Increment step counter
        self.current_step += 1

        # Execute the move
        moved = self._execute_move(action)

        # Check if 2048 tile was reached (power of 11 means 2^11 = 2048)
        reached_2048 = np.any(self.board == 11)
        reward = 1 if reached_2048 else 0

        # If the board changed, add a new tile
        if moved:
            self._add_random_tile()

            # Check if game is over: reached 2048, no moves left, or max steps exceeded
            max_steps_exceeded = (
                self.max_steps is not None and self.current_step >= self.max_steps
            )
            self.game_over = self._is_game_over() or reached_2048 or max_steps_exceeded

        return self.board.copy(), reward, self.game_over

    @property
    def max_value_reached(self):
        """Return the maximum tile value reached on the board."""
        if np.all(self.board == 0):
            return 0
        max_power = np.max(self.board)
        return int(2**max_power)

    def _execute_move(self, action: Action):
        """
        Execute a move and update the board.

        Returns:
            bool: True if the board changed, False otherwise
        """
        # Store the board state before the move
        before = self.board.copy()

        if action == Action.LEFT:
            self._move_left()
        elif action == Action.RIGHT:
            self._move_right()
        elif action == Action.UP:
            self._move_up()
        elif action == Action.DOWN:
            self._move_down()

        # Check if board changed
        return not np.array_equal(before, self.board)

    def _move_left(self):
        """Move and merge tiles to the left."""
        for i in range(4):
            self.board[i] = self._merge_line(self.board[i])

    def _move_right(self):
        """Move and merge tiles to the right."""
        for i in range(4):
            # Reverse, merge, reverse back
            self.board[i] = self._merge_line(self.board[i][::-1])[::-1]

    def _move_up(self):
        """Move and merge tiles upward."""
        self.board = self.board.T
        self._move_left()
        self.board = self.board.T

    def _move_down(self):
        """Move and merge tiles downward."""
        self.board = self.board.T
        self._move_right()
        self.board = self.board.T

    def _merge_line(self, line):
        """
        Merge a single line (row or column) to the left.

        This is the core algorithm:
        1. Move all non-zero tiles to the left
        2. Merge adjacent equal tiles
        3. Move merged tiles to the left again

        Args:
            line: 1D array of 4 elements

        Returns:
            np.ndarray: Merged line
        """
        # Remove zeros (slide tiles together)
        non_zero = line[line != 0]

        # Merge adjacent equal tiles
        merged = []
        i = 0
        while i < len(non_zero):
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                # Merge: increment power (e.g., 2 -> 3 means 4 -> 8)
                merged_value = non_zero[i] + 1
                merged.append(merged_value)
                # Add score (actual tile value)
                self.score += 2**merged_value
                i += 2
            else:
                merged.append(non_zero[i])
                i += 1

        # Pad with zeros to make it length 4
        result = np.zeros(4, dtype=np.int32)
        result[: len(merged)] = merged

        return result

    def _add_random_tile(self):
        """
        Add a random tile (90% chance of 2, 10% chance of 4) to an empty position.
        In our representation: 90% chance of 1 (2^1=2), 10% chance of 2 (2^2=4).
        """
        # Find empty positions
        empty_positions = np.argwhere(self.board == 0)

        if len(empty_positions) == 0:
            return

        # Choose random empty position
        pos = empty_positions[np.random.randint(len(empty_positions))]

        # 90% chance of 2 (power=1), 10% chance of 4 (power=2)
        value = 1 if np.random.random() < 0.9 else 2

        self.board[pos[0], pos[1]] = value

    def _is_game_over(self):
        """
        Check if the game is over (no valid moves remaining).

        Returns:
            bool: True if game is over, False otherwise
        """
        # If there are empty cells, game is not over
        if np.any(self.board == 0):
            return False

        # Check for possible merges horizontally
        for i in range(4):
            for j in range(3):
                if self.board[i, j] == self.board[i, j + 1]:
                    return False

        # Check for possible merges vertically
        for i in range(3):
            for j in range(4):
                if self.board[i, j] == self.board[i + 1, j]:
                    return False

        # No moves available
        return True

    def render_str(self):
        render_str(self.board)


if __name__ == "__main__":
    # Interactive mode - play 2048 in the terminal
    print("=" * 50)
    print("Welcome to 2048!")
    print("=" * 50)
    print("\nControls:")
    print("  w or ↑ : Move UP")
    print("  s or ↓ : Move DOWN")
    print("  a or ← : Move LEFT")
    print("  d or → : Move RIGHT")
    print("  q      : Quit game")
    print("\nGoal: Reach the 2048 tile!")
    print("=" * 50)

    # Initialize game
    env = Env2048()
    board = env.reset(seed=None)

    print("\nStarting board:")
    env.render_str()
    print(f"Score: {env.score} | Max Tile: {env.max_value_reached}")

    move_count = 0

    # Game loop
    while not env.game_over:
        # Get user input
        user_input = input("\nYour move (w/a/s/d or q to quit): ").strip().lower()

        if user_input == "q":
            print("\nThanks for playing!")
            break

        # Map input to action
        action_map = {
            "w": Action.UP,
            "s": Action.DOWN,
            "a": Action.LEFT,
            "d": Action.RIGHT,
            "↑": Action.UP,
            "↓": Action.DOWN,
            "←": Action.LEFT,
            "→": Action.RIGHT,
        }

        if user_input not in action_map:
            print("Invalid input! Use w/a/s/d or q to quit.")
            continue

        action = action_map[user_input]

        # Execute move
        board, reward, done = env.step(action)
        move_count += 1

        # Clear screen effect (print newlines)
        print("\n" * 2)
        print("=" * 50)
        print(f"Move #{move_count} - {action.name}")
        print("=" * 50)
        env.render_str()
        print(f"Score: {env.score} | Max Tile: {env.max_value_reached}")

        # Check for win condition
        if reward == 1:
            print("\n" + "=" * 50)
            print("🎉 CONGRATULATIONS! YOU REACHED 2048! 🎉")
            print("=" * 50)
            print(f"Total moves: {move_count}")
            print(f"Final score: {env.score}")
            break

        # Check for game over
        if done and reward == 0:
            print("\n" + "=" * 50)
            print("Game Over! No more moves available.")
            print("=" * 50)
            print(f"Total moves: {move_count}")
            print(f"Final score: {env.score}")
            print(f"Max tile reached: {env.max_value_reached}")
            break

    print("\nFinal board:")
    env.render_str()
