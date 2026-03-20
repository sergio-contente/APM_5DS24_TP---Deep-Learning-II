"""Step 4b: Validate DBN on Binary AlphaDigits."""
import numpy as np
import matplotlib.pyplot as plt
from utils import lire_alpha_digit
from dbn import init_DBN, train_DBN, generer_image_DBN

# Load characters 0, 1, 2
print("Loading Binary AlphaDigits (characters 0, 1, 2)...")
data = lire_alpha_digit([0, 1, 2])
print(f"Data shape: {data.shape}")

# --- Experiment 1: Single-layer DBN (equivalent to RBM) vs 2-layer DBN ---
print("\n=== DBN with 1 hidden layer (320 -> 200) ===")
dbn1 = init_DBN([320, 200])
train_DBN(dbn1, data, epochs=100, lr=0.1, batch_size=32)
generer_image_DBN(dbn1, n_gibbs=500, n_images=10, image_shape=(20, 16))
plt.savefig("dbn_alpha_1layer.png", dpi=150, bbox_inches="tight")

print("\n=== DBN with 2 hidden layers (320 -> 200 -> 100) ===")
dbn2 = init_DBN([320, 200, 100])
train_DBN(dbn2, data, epochs=100, lr=0.1, batch_size=32)
generer_image_DBN(dbn2, n_gibbs=500, n_images=10, image_shape=(20, 16))
plt.savefig("dbn_alpha_2layers.png", dpi=150, bbox_inches="tight")

print("\n=== DBN with 3 hidden layers (320 -> 200 -> 100 -> 50) ===")
dbn3 = init_DBN([320, 200, 100, 50])
train_DBN(dbn3, data, epochs=100, lr=0.1, batch_size=32)
generer_image_DBN(dbn3, n_gibbs=500, n_images=10, image_shape=(20, 16))
plt.savefig("dbn_alpha_3layers.png", dpi=150, bbox_inches="tight")

# --- Experiment 2: Effect of character diversity on DBN ---
print("\n=== DBN with varying number of characters ===")
for n_chars in [5, 10, 36]:
    chars = list(range(n_chars))
    data_div = lire_alpha_digit(chars)
    print(f"\nDBN on {n_chars} characters ({data_div.shape[0]} samples)")
    dbn = init_DBN([320, 200, 100])
    train_DBN(dbn, data_div, epochs=100, lr=0.1, batch_size=32)
    generer_image_DBN(dbn, n_gibbs=500, n_images=10, image_shape=(20, 16))
    plt.savefig(f"dbn_alpha_nchars{n_chars}.png", dpi=150, bbox_inches="tight")

print("\nDone! Check generated images.")
