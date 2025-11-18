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

## 4. Image Downsampling
### 4.1 Grayscale image

```python
import numpy as np
import matplotlib.pyplot as plt

def downsample_image(image: np.ndarray, downsampling_rate: int) -> np.ndarray:
    """
    Downsample a grayscale (2D) or color (3D) image using block averaging.

    image:
        - grayscale: shape (H, W)
        - color:     shape (H, W, C)  (e.g., C=3 for RGB)
    downsampling_rate: size of the square kernel (block)

    Returns:
        downsampled image as a numpy array.
    """
    r = downsampling_rate
    if r <= 0:
        raise ValueError("downsampling_rate must be a positive integer")

    if image.ndim == 2:
        # ----- Grayscale -----
        H, W = image.shape
        Hc = (H // r) * r
        Wc = (W // r) * W

        img_cropped = image[:Hc, :Wc]

        # reshape to (Hc/r, r, Wc/r, r) and average over the 2x axes
        img_ds = img_cropped.reshape(Hc // r, r, Wc // r, r).mean(axis=(1, 3))

    elif image.ndim == 3:
        # ----- Color (RGB) -----
        H, W, C = image.shape
        Hc = (H // r) * r
        Wc = (W // r) * r

        img_cropped = image[:Hc, :Wc, :]

        # reshape to (Hc/r, r, Wc/r, r, C) and average over block axes
        img_ds = img_cropped.reshape(Hc // r, r, Wc // r, r, C).mean(axis=(1, 3))

    else:
        raise ValueError("Image must be 2D (grayscale) or 3D (color).")

    # Show the result
    plt.figure()
    if img_ds.ndim == 2:
        plt.imshow(img_ds, cmap="gray")
    else:
        # For RGB, make sure values are in displayable range (0–255 or 0–1)
        plt.imshow(np.clip(img_ds, 0, 255).astype(np.uint8))
    plt.title(f"Downsampled image (rate={r})")
    plt.axis("off")
    plt.show()

    return img_ds
```

### 4.2 Testing with the given sample image

Grayscale

```python
from PIL import Image

img = Image.open("sample_image_2.jpg").convert("L")  # grayscale
img_array = np.array(img)

img_ds = downsample_image(img_array, downsampling_rate=5)
```

Color:

```python
img_color = Image.open("sample_image_2.jpg")  # keep RGB
img_color_array = np.array(img_color)

img_ds_color = downsample_image(img_color_array, downsampling_rate=5)
```

## 5. Word Embedding
### 5.1 Cosine similarity metric

```python
import numpy as np
import matplotlib.pyplot as plt

embeddings = {
    "king": [
        1.0, 1.0, 1.0, 0.0, 0.8, 0.9, 0.0, 0.0, 0.0, 0.0
    ] + [0.0]*40,
    "queen": [
        1.0, -1.0, 1.0, 0.0, 0.8, 0.9, 0.0, 0.0, 0.0, 0.0
    ] + [0.0]*40,
    "man": [
        0.0, 1.0, 1.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0
    ] + [0.0]*40,
    "woman": [
        0.0, -1.0, 1.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0
    ] + [0.0]*40,
    "apple": [
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.7, 0.2, 0.6, 0.2
    ] + [0.0]*40,
    "banana": [
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.6, 0.8, 0.6, 0.7
    ] + [0.0]*40
}

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    """
    a = np.array(a)
    b = np.array(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
```

### 5.2 Pairwise similarity visualization

```python
words = list(embeddings.keys())
vecs = [np.array(embeddings[w]) for w in words]
n = len(words)

sim_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        sim_matrix[i, j] = cosine_similarity(vecs[i], vecs[j])

plt.figure()
plt.imshow(sim_matrix, interpolation="nearest")
plt.colorbar(label="Cosine similarity")
plt.xticks(range(n), words, rotation=45, ha="right")
plt.yticks(range(n), words)
plt.title("Word Similarity Heatmap")
plt.tight_layout()
plt.show()
```

### 5.3 king – man + woman

```python
king = np.array(embeddings["king"])
queen = np.array(embeddings["queen"])
man = np.array(embeddings["man"])
woman = np.array(embeddings["woman"])

query_vec = king - man + woman

# Find the nearest word by cosine similarity
best_word = None
best_score = -1.0
for word, emb in embeddings.items():
    score = cosine_similarity(query_vec, emb)
    if score > best_score:
        best_score = score
        best_word = word

print("Result of king - man + woman:", best_word)
print("Cosine similarity:", best_score)
```
Result of king - man + woman: queen





