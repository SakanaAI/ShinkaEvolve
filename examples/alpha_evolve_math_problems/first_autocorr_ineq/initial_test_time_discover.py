# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize

def get_good_direction_to_move_into(
    sequence: list[float],
) -> list[float] | None:
    """Returns the direction to move into the sequence."""
    n = len(sequence)
    sum_sequence = np.sum(sequence)
    normalized_sequence = [x * np.sqrt(2 * n) / sum_sequence for x in sequence]
    rhs = np.max(np.convolve(normalized_sequence, normalized_sequence))
    g_fun = solve_convolution_lp(normalized_sequence, rhs)
    if g_fun is None:
        return None
    sum_sequence = np.sum(g_fun)
    normalized_g_fun = [x * np.sqrt(2 * n) / sum_sequence for x in g_fun]
    t = 0.01
    new_sequence = [
        (1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)
    ]
    return new_sequence

def solve_convolution_lp(f_sequence, rhs):
    """Solves the convolution LP for a given sequence and RHS."""
    n = len(f_sequence)
    c = -np.ones(n)
    a_ub = []
    b_ub = []
    for k in range(2 * n - 1):
        row = np.zeros(n)
        for i in range(n):
            j = k - i
            if 0 <= j < n:
                row[j] = f_sequence[i]
        a_ub.append(row)
        b_ub.append(rhs)

    # Non-negativity constraints: b_i >= 0
    a_ub_nonneg = -np.eye(n)  # Negative identity matrix for b_i >= 0
    b_ub_nonneg = np.zeros(n)  # Zero vector

    a_ub = np.vstack([a_ub, a_ub_nonneg])
    b_ub = np.hstack([b_ub, b_ub_nonneg])

    result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub)

    if result.success:
        g_sequence = result.x
        return g_sequence
    else:
        print('LP optimization failed.')
        return None
     

def run() -> list[float]:
    """Function to search for the best coefficient sequence."""
    best_sequence = [np.random.random()] * np.random.randint(100,1000)
    h_function = get_good_direction_to_move_into(best_sequence)
    if h_function is None:
        best_sequence[1] = (best_sequence[1] + np.random.rand()) % 1
    else:
        best_sequence = h_function

    return best_sequence

# EVOLVE-BLOCK-END


if __name__ == "__main__":
    sequence = run()
    print(f"Found sequence: {sequence}")




