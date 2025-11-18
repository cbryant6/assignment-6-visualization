# Visualization
## 1. Grid World Visualization – Part I

```python

import numpy as np
import matplotlib.pyplot as plt

def display_grid(locations: np.ndarray, grid_shape=(10, 10)) -> None:
    """
    Display robot locations on a 2D grid.

    locations: array of shape (num_robots, 2) with [row, col]
    grid_shape: (n_rows, n_cols) of the grid
    """
    n_rows, n_cols = grid_shape

    rows = locations[:, 0]
    cols = locations[:, 1]

    plt.figure()
    plt.scatter(cols, rows, s=100)  # x=col, y=row

    # Make the grid visible and fully shown
    plt.xlim(-0.5, n_cols - 0.5)
    plt.ylim(-0.5, n_rows - 0.5)
    plt.gca().invert_yaxis()  # so row 0 is at the top like an array

    plt.xticks(range(n_cols))
    plt.yticks(range(n_rows))
    plt.grid(True, which="both", linestyle="--", alpha=0.4)

    plt.xlabel("Column index")
    plt.ylabel("Row index")
    plt.title("Robot Locations in Grid World")

    plt.show()
```

## 2. Grid World – Part II: update_location

```python
def update_location(location: np.ndarray,
                    action: str,
                    grid_shape=(10, 10)) -> np.ndarray:
    """
    Update a robot's location given an action, respecting grid boundaries.

    location: np.array([row, col])
    action: one of {"up", "down", "left", "right"}
    grid_shape: (n_rows, n_cols)

    Returns a new np.array([new_row, new_col]).
    """
    n_rows, n_cols = grid_shape
    row, col = int(location[0]), int(location[1])

    if action == "up":
        new_row, new_col = max(row - 1, 0), col
    elif action == "down":
        new_row, new_col = min(row + 1, n_rows - 1), col
    elif action == "left":
        new_row, new_col = row, max(col - 1, 0)
    elif action == "right":
        new_row, new_col = row, min(col + 1, n_cols - 1)
    else:
        raise ValueError(f"Unknown action: {action}")

    return np.array([new_row, new_col])
```

## 3. UPDATED simulate_disease WITH ALL NEW REQUIREMENTS

```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_disease(initial_probs,
                     adjacency_matrix,
                     infection_rate,
                     recovery_rate,
                     horizon):
    """
    Simulate how infection probabilities evolve over time AND visualize results.

    NEW FEATURES ADDED:
      - Tracks history for entire horizon (shape: [horizon+1, N])
      - Prints min, max, mean, std of final probabilities
      - Plots infection evolution for all individuals
    """

    initial_probs = initial_probs.copy()
    N = len(initial_probs)

    # Store history for plotting
    history = np.zeros((horizon + 1, N))
    history[0] = initial_probs

    prob = initial_probs.copy()

    for t in range(1, horizon + 1):

        infection_from_neighbors = np.zeros(N)

        # ----- your exact original logic preserved -----
        for i in range(N):
            neighbors = np.where(adjacency_matrix[i] == 1)[0]
            if len(neighbors) > 0:
                infection_from_neighbors[i] = (
                    1 - np.prod(1 - infection_rate * prob[neighbors])
                )

        recovery_effect = recovery_rate * prob

        prob = prob + infection_from_neighbors - recovery_effect
        prob = np.clip(prob, 0, 1)
        # -------------------------------------------------

        history[t] = prob

    # ========== NEW REQUIREMENT 1: PRINT STATISTICS ==========
    final_probs = history[-1]
    print("Final Infection Probability Statistics:")
    print(f"  Min:  {final_probs.min():.4f}")
    print(f"  Max:  {final_probs.max():.4f}")
    print(f"  Mean: {final_probs.mean():.4f}")
    print(f"  Std:  {final_probs.std():.4f}")

    # ========== NEW REQUIREMENT 2: PLOT EVOLUTION ==========
    plt.figure(figsize=(8,5))
    time_axis = np.arange(horizon + 1)

    for i in range(N):
        plt.plot(time_axis, history[:, i], alpha=0.6)

    plt.xlabel("Time Step")
    plt.ylabel("Infection Probability")
    plt.title("Evolution of Infection Probabilities Over Time")
    plt.grid(True, linestyle="--", alpha=0.3)
    # no legends required
    plt.tight_layout()
    plt.show()

    # ========== Return full history for future analysis ==========
    return history
```







