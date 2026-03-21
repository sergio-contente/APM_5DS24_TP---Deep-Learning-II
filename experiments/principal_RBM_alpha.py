"""Step 4a: Validate RBM on Binary AlphaDigits."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
IMG_DIR = os.path.join(os.path.dirname(__file__), "..", "images")
os.makedirs(IMG_DIR, exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt
from src import lire_alpha_digit, init_RBM, train_RBM, generer_image_RBM

# Load characters 0, 1, 2 (indices 0, 1, 2)
print("Loading Binary AlphaDigits (characters 0, 1, 2)...")
data = lire_alpha_digit([0, 1, 2])
print(f"Data shape: {data.shape}")  # (117, 320) = 3 chars * 39 samples

# Display some original images
fig, axes = plt.subplots(3, 5, figsize=(10, 6))
for i in range(3):
    for j in range(5):
        axes[i, j].imshow(data[i * 39 + j].reshape(20, 16), cmap="gray")
        axes[i, j].axis("off")
plt.suptitle("Original Binary AlphaDigits (chars 0, 1, 2)")
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/alpha_originals.png", dpi=150, bbox_inches="tight")
plt.show()

# --- Experiment 1: Effect of number of hidden units ---
print("\n=== Effect of hidden units ===")
for q in [50, 100, 200, 300]:
    print(f"\nTraining RBM 320 -> {q}")
    rbm = init_RBM(320, q)
    train_RBM(rbm, data, epochs=100, lr=0.1, batch_size=32)
    generer_image_RBM(rbm, n_gibbs=500, n_images=10, image_shape=(20, 16),
                      save_path=f"{IMG_DIR}/rbm_alpha_q{q}.png",
                      title=f"RBM Generated (q={q})")

# --- Experiment 2: Effect of character diversity ---
print("\n=== Effect of character diversity ===")
for n_chars in [3, 10, 36]:
    chars = list(range(n_chars))
    data_div = lire_alpha_digit(chars)
    print(f"\nTraining RBM on {n_chars} characters ({data_div.shape[0]} samples)")
    rbm = init_RBM(320, 200)
    train_RBM(rbm, data_div, epochs=100, lr=0.1, batch_size=32)
    generer_image_RBM(rbm, n_gibbs=500, n_images=10, image_shape=(20, 16),
                      save_path=f"{IMG_DIR}/rbm_alpha_nchars{n_chars}.png",
                      title=f"RBM Generated ({n_chars} chars, q=200)")

print("\nDone! Check generated images.")
