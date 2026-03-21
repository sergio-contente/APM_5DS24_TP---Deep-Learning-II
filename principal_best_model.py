"""Train the best DNN configuration based on experimental results.

Analysis of Figures 1-3:
- Fig 1: Pre-training is essential for deep networks (>3 layers). With 2 layers,
  both methods perform similarly (~3%). No benefit from adding more layers.
  -> Best: 2 hidden layers.
- Fig 2: Pre-trained network improves with more neurons, reaching ~2.6% at 700.
  Random network stays ~2.8% regardless. Pre-trained + 700 neurons is the best.
  -> Best: 700 neurons per layer.
- Fig 3: With all 60k samples, both converge. Pre-training helps most with few data.
  -> Best: use all 60000 training samples.

Best configuration: [784, 700, 700, 10], pre-trained, all training data.
"""
import numpy as np
import matplotlib.pyplot as plt
import copy
from utils import load_mnist
from dnn import init_DNN, pretrain_DNN, retropropagation, test_DNN, entree_sortie_reseau

# ============================================================
# Load data
# ============================================================
print("Loading MNIST...")
X_train, Y_train, X_test, Y_test = load_mnist()

# ============================================================
# Train best model: [784, 700, 700, 10] pre-trained
# ============================================================
BEST_LAYERS = [784, 700, 700, 10]
print(f"\nBest configuration: {BEST_LAYERS}")
print("Justification:")
print("  - 2 hidden layers: no benefit from deeper networks (Fig 1)")
print("  - 700 neurons: best error rate for pre-trained (Fig 2)")
print("  - All 60k samples: maximizes performance (Fig 3)")
print("  - Pre-trained: consistently better or equal to random init")

dnn = init_DNN(BEST_LAYERS)

print("\n--- Pre-training (RBM, 100 epochs) ---")
pretrain_DNN(dnn, X_train, epochs=100, lr=0.1, batch_size=128)

print("\n--- Backpropagation (200 epochs) ---")
retropropagation(dnn, X_train, Y_train, epochs=200, lr=0.1, batch_size=128)

# ============================================================
# Evaluate on train and test
# ============================================================
err_train = test_DNN(dnn, X_train, Y_train)
err_test = test_DNN(dnn, X_test, Y_test)
print(f"\nBest model results:")
print(f"  Train error: {err_train*100:.2f}%")
print(f"  Test error:  {err_test*100:.2f}%")

# ============================================================
# Softmax output probabilities + images (combined figure)
# ============================================================
print("\nGenerating softmax visualization...")
n_samples = 10
sample_idx = np.random.choice(X_test.shape[0], n_samples, replace=False)

fig, axes = plt.subplots(2, n_samples, figsize=(2.5 * n_samples, 5),
                         gridspec_kw={"height_ratios": [1, 1.2]})

for i, idx in enumerate(sample_idx):
    activations = entree_sortie_reseau(dnn, X_test[idx:idx+1])
    probs = activations[-1][0]
    true_label = np.argmax(Y_test[idx])
    pred_label = np.argmax(probs)

    # Top row: image
    axes[0, i].imshow(X_test[idx].reshape(28, 28), cmap="gray")
    axes[0, i].set_title(f"True: {true_label}", fontsize=9)
    axes[0, i].axis("off")

    # Bottom row: softmax bar chart
    colors = ["green" if j == true_label else ("red" if j == pred_label and j != true_label else "gray")
              for j in range(10)]
    axes[1, i].bar(range(10), probs, color=colors)
    axes[1, i].set_title(f"Pred: {pred_label}", fontsize=9)
    axes[1, i].set_xticks(range(10))
    axes[1, i].set_ylim(0, 1)
    axes[1, i].tick_params(labelsize=7)

plt.suptitle(f"Best Model {BEST_LAYERS} — Test Error: {err_test*100:.2f}%\n"
             "(green = true label, red = wrong prediction)", fontsize=12)
plt.tight_layout()
plt.savefig("softmax_best_model.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\nSaved to softmax_best_model.png")
