import numpy as np
import matplotlib.pyplot as plt
from rbm import init_RBM, train_RBM, entree_sortie_RBM, sortie_entree_RBM


def init_DBN(layer_sizes):
    """Initialize a DBN as a list of RBMs.
    E.g. [320, 200, 100] -> 2 RBMs: 320->200, 200->100."""
    rbms = []
    for i in range(len(layer_sizes) - 1):
        rbms.append(init_RBM(layer_sizes[i], layer_sizes[i + 1]))
    return rbms


def train_DBN(dbn, data, epochs=100, lr=0.1, batch_size=128, verbose=True):
    """Train DBN greedily layer by layer."""
    current_data = data.copy()
    for i, rbm in enumerate(dbn):
        if verbose:
            print(f"Training RBM layer {i+1}/{len(dbn)} ({rbm['W'].shape[0]} -> {rbm['W'].shape[1]})")
        train_RBM(rbm, current_data, epochs=epochs, lr=lr, batch_size=batch_size, verbose=verbose)
        # Transform data for next layer (use probabilities)
        current_data = entree_sortie_RBM(rbm, current_data)


def generer_image_DBN(dbn, n_gibbs=1000, n_images=10, image_shape=None, save_path=None, title="DBN Generated Images"):
    """Generate images from DBN: Gibbs sampling at top layer, then top-down propagation."""
    p = dbn[0]["W"].shape[0]
    if image_shape is None:
        side = int(np.sqrt(p))
        image_shape = (side, side) if side * side == p else (20, 16)

    top_rbm = dbn[-1]
    q = top_rbm["W"].shape[1]

    # Initialize top hidden layer randomly
    v = (np.random.rand(n_images, top_rbm["W"].shape[0]) > 0.5).astype(np.float64)

    # Gibbs sampling at the top RBM
    for _ in range(n_gibbs):
        h_prob = entree_sortie_RBM(top_rbm, v)
        h_sample = (np.random.rand(*h_prob.shape) < h_prob).astype(np.float64)
        v_prob = sortie_entree_RBM(top_rbm, h_sample)
        v = (np.random.rand(*v_prob.shape) < v_prob).astype(np.float64)

    # Top-down propagation using probabilities (smoother images)
    current = v
    for i in range(len(dbn) - 2, -1, -1):
        current = sortie_entree_RBM(dbn[i], current)

    # Display
    cols = min(n_images, 5)
    rows = (n_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        if i < n_images:
            ax.imshow(current[i].reshape(image_shape), cmap="gray")
        ax.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    return current
