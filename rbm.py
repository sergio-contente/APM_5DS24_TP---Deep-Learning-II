import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def init_RBM(p, q):
    """Initialize an RBM with p visible and q hidden units."""
    return {
        "W": np.random.randn(p, q) * 0.01,
        "a": np.zeros(p),
        "b": np.zeros(q),
    }


def entree_sortie_RBM(rbm, V):
    """Forward pass: visible -> hidden. Returns P(h|v)."""
    return sigmoid(V @ rbm["W"] + rbm["b"])


def sortie_entree_RBM(rbm, H):
    """Backward pass: hidden -> visible. Returns P(v|h)."""
    return sigmoid(H @ rbm["W"].T + rbm["a"])


def train_RBM(rbm, data, epochs=100, lr=0.1, batch_size=128, verbose=True):
    """Train RBM using Contrastive Divergence-1."""
    n = data.shape[0]
    errors = []

    for epoch in range(epochs):
        perm = np.random.permutation(n)
        epoch_error = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            v0 = data[idx]
            bs = v0.shape[0]

            # Positive phase
            h0_prob = entree_sortie_RBM(rbm, v0)
            h0_sample = (np.random.rand(*h0_prob.shape) < h0_prob).astype(np.float64)

            # Negative phase (use sampled h0 for reconstruction)
            v1_prob = sortie_entree_RBM(rbm, h0_sample)
            h1_prob = entree_sortie_RBM(rbm, v1_prob)

            # Gradients (use probabilities of h0 for the positive phase gradient)
            rbm["W"] += lr * (v0.T @ h0_prob - v1_prob.T @ h1_prob) / bs
            rbm["a"] += lr * np.mean(v0 - v1_prob, axis=0)
            rbm["b"] += lr * np.mean(h0_prob - h1_prob, axis=0)

            epoch_error += np.mean((v0 - v1_prob) ** 2)
            n_batches += 1

        mse = epoch_error / n_batches
        errors.append(mse)
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  RBM Epoch {epoch+1}/{epochs} - MSE: {mse:.6f}")

    return errors


def generer_image_RBM(rbm, n_gibbs=1000, n_images=10, image_shape=None):
    """Generate images via Gibbs sampling. Returns generated images."""
    p = rbm["W"].shape[0]
    if image_shape is None:
        side = int(np.sqrt(p))
        image_shape = (side, side) if side * side == p else (20, 16)

    v = (np.random.rand(n_images, p) > 0.5).astype(np.float64)

    for _ in range(n_gibbs):
        h_prob = entree_sortie_RBM(rbm, v)
        h_sample = (np.random.rand(*h_prob.shape) < h_prob).astype(np.float64)
        v_prob = sortie_entree_RBM(rbm, h_sample)
        v = (np.random.rand(*v_prob.shape) < v_prob).astype(np.float64)

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
            ax.imshow(v[i].reshape(image_shape), cmap="gray")
        ax.axis("off")
    plt.suptitle("RBM Generated Images")
    plt.tight_layout()
    plt.savefig("rbm_generated.png", dpi=150, bbox_inches="tight")
    plt.show()

    return v
