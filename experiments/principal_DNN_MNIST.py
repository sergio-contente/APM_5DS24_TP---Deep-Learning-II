"""Step 5: Final study on MNIST - Compare pre-trained vs random DNN."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
IMG_DIR = os.path.join(os.path.dirname(__file__), "..", "images")
os.makedirs(IMG_DIR, exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt
import copy
from src import load_mnist, init_DNN, pretrain_DNN, retropropagation, test_DNN, entree_sortie_reseau

# ============================================================
# Hyperparameters
# ============================================================
RBM_EPOCHS = 100
BACKPROP_EPOCHS = 200
LR = 0.1
BATCH_SIZE = 128

# ============================================================
# Load and prepare MNIST
# ============================================================
print("Loading MNIST...")
X_train, Y_train, X_test, Y_test = load_mnist()
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Images are binarized: unique values = {np.unique(X_train[:10])}")


def run_experiment(layer_sizes, X_tr, Y_tr, X_te, Y_te,
                   rbm_epochs=RBM_EPOCHS, bp_epochs=BACKPROP_EPOCHS,
                   lr=LR, batch_size=BATCH_SIZE):
    """Run one comparison: pretrained DNN vs random DNN.
    Returns (err_pretrained_test, err_random_test, err_pretrained_train, err_random_train, dnn1, dnn2)."""
    print(f"\n{'='*60}")
    print(f"Architecture: {layer_sizes}, Train samples: {X_tr.shape[0]}")
    print(f"{'='*60}")

    # 1. Initialize DNN
    dnn1 = init_DNN(layer_sizes)
    # 2. Deep copy for the random network (identical starting weights)
    dnn2 = copy.deepcopy(dnn1)

    # 3. Pre-train DNN1
    print("\n--- Pre-training DNN1 (RBM) ---")
    pretrain_DNN(dnn1, X_tr, epochs=rbm_epochs, lr=lr, batch_size=batch_size, verbose=True)

    # 4. Train DNN1 with backpropagation
    print("\n--- Training DNN1 (pre-trained + backprop) ---")
    retropropagation(dnn1, X_tr, Y_tr, epochs=bp_epochs, lr=lr, batch_size=batch_size, verbose=True)

    # 5. Train DNN2 with backpropagation only (random init)
    print("\n--- Training DNN2 (random + backprop) ---")
    retropropagation(dnn2, X_tr, Y_tr, epochs=bp_epochs, lr=lr, batch_size=batch_size, verbose=True)

    # 6. Evaluate on train and test
    err1_test = test_DNN(dnn1, X_te, Y_te)
    err2_test = test_DNN(dnn2, X_te, Y_te)
    err1_train = test_DNN(dnn1, X_tr, Y_tr)
    err2_train = test_DNN(dnn2, X_tr, Y_tr)
    print(f"\nResults (test):  Pre-trained = {err1_test*100:.2f}%, Random = {err2_test*100:.2f}%")
    print(f"Results (train): Pre-trained = {err1_train*100:.2f}%, Random = {err2_train*100:.2f}%")

    return err1_test, err2_test, err1_train, err2_train, dnn1, dnn2


# ============================================================
# Figure 1: Error rate vs. Number of hidden layers
# ============================================================
print("\n" + "#" * 60)
print("FIGURE 1: Error rate vs. Number of hidden layers")
print("#" * 60)

n_layers_list = [2, 3, 4, 5]
errors_pretrained_fig1 = []
errors_random_fig1 = []

for n_layers in n_layers_list:
    layers = [784] + [200] * n_layers + [10]
    err_p, err_r, _, _, _, _ = run_experiment(layers, X_train, Y_train, X_test, Y_test)
    errors_pretrained_fig1.append(err_p * 100)
    errors_random_fig1.append(err_r * 100)

plt.figure(figsize=(8, 5))
plt.plot(n_layers_list, errors_pretrained_fig1, "b-o", label="Pre-trained + backprop")
plt.plot(n_layers_list, errors_random_fig1, "r-o", label="Random + backprop")
plt.xlabel("Number of hidden layers")
plt.ylabel("Classification error rate (%)")
plt.title("Error Rate vs. Number of Hidden Layers (200 neurons each)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(n_layers_list)
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/fig1_layers.png", dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# Figure 2: Error rate vs. Number of neurons per layer
# ============================================================
print("\n" + "#" * 60)
print("FIGURE 2: Error rate vs. Number of neurons per layer")
print("#" * 60)

neurons_list = [100, 300, 500, 700]
errors_pretrained_fig2 = []
errors_random_fig2 = []

for n_neurons in neurons_list:
    layers = [784, n_neurons, n_neurons, 10]
    err_p, err_r, _, _, _, _ = run_experiment(layers, X_train, Y_train, X_test, Y_test)
    errors_pretrained_fig2.append(err_p * 100)
    errors_random_fig2.append(err_r * 100)

plt.figure(figsize=(8, 5))
plt.plot(neurons_list, errors_pretrained_fig2, "b-o", label="Pre-trained + backprop")
plt.plot(neurons_list, errors_random_fig2, "r-o", label="Random + backprop")
plt.xlabel("Number of neurons per layer")
plt.ylabel("Classification error rate (%)")
plt.title("Error Rate vs. Number of Neurons per Layer (2 hidden layers)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(neurons_list)
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/fig2_neurons.png", dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# Figure 3: Error rate vs. Number of training samples
# ============================================================
print("\n" + "#" * 60)
print("FIGURE 3: Error rate vs. Number of training samples")
print("#" * 60)

n_samples_list = [1000, 3000, 7000, 10000, 30000, 60000]
errors_pretrained_fig3 = []
errors_random_fig3 = []

for n_samples in n_samples_list:
    if n_samples < X_train.shape[0]:
        idx = np.random.choice(X_train.shape[0], n_samples, replace=False)
        X_sub = X_train[idx]
        Y_sub = Y_train[idx]
    else:
        X_sub = X_train
        Y_sub = Y_train
    layers = [784, 200, 200, 10]
    err_p, err_r, _, _, _, _ = run_experiment(layers, X_sub, Y_sub, X_test, Y_test)
    errors_pretrained_fig3.append(err_p * 100)
    errors_random_fig3.append(err_r * 100)

plt.figure(figsize=(8, 5))
plt.plot(n_samples_list, errors_pretrained_fig3, "b-o", label="Pre-trained + backprop")
plt.plot(n_samples_list, errors_random_fig3, "r-o", label="Random + backprop")
plt.xlabel("Number of training samples")
plt.ylabel("Classification error rate (%)")
plt.title("Error Rate vs. Number of Training Samples ([784,200,200,10])")
plt.xscale("log")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/fig3_samples.png", dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY OF RESULTS")
print("=" * 60)
print("\nFigure 1 (layers):")
for nl, ep, er in zip(n_layers_list, errors_pretrained_fig1, errors_random_fig1):
    print(f"  {nl} layers: pretrained={ep:.2f}%, random={er:.2f}%")
print("\nFigure 2 (neurons):")
for nn, ep, er in zip(neurons_list, errors_pretrained_fig2, errors_random_fig2):
    print(f"  {nn} neurons: pretrained={ep:.2f}%, random={er:.2f}%")
print("\nFigure 3 (samples):")
for ns, ep, er in zip(n_samples_list, errors_pretrained_fig3, errors_random_fig3):
    print(f"  {ns} samples: pretrained={ep:.2f}%, random={er:.2f}%")
