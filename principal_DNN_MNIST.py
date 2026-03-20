"""Step 5: Final study on MNIST - Compare pre-trained vs random DNN."""
import numpy as np
import matplotlib.pyplot as plt
import copy
from utils import load_mnist
from dnn import init_DNN, pretrain_DNN, retropropagation, test_DNN, entree_sortie_reseau

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
    Returns (error_pretrained, error_random)."""
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

    # 6. Test both
    err1 = test_DNN(dnn1, X_te, Y_te)
    err2 = test_DNN(dnn2, X_te, Y_te)
    print(f"\nResults: Pre-trained error = {err1*100:.2f}%, Random error = {err2*100:.2f}%")

    return err1, err2, dnn1, dnn2


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
    err_p, err_r, _, _ = run_experiment(layers, X_train, Y_train, X_test, Y_test)
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
plt.savefig("fig1_layers.png", dpi=150, bbox_inches="tight")
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
    err_p, err_r, _, _ = run_experiment(layers, X_train, Y_train, X_test, Y_test)
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
plt.savefig("fig2_neurons.png", dpi=150, bbox_inches="tight")
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
    err_p, err_r, _, _ = run_experiment(layers, X_sub, Y_sub, X_test, Y_test)
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
plt.savefig("fig3_samples.png", dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# Softmax output probabilities visualization
# ============================================================
print("\n" + "#" * 60)
print("Softmax output probabilities for sample images")
print("#" * 60)

# Train a final model with best config
print("\nTraining final model [784, 300, 300, 10] with all data...")
_, _, dnn_best, _ = run_experiment([784, 300, 300, 10], X_train, Y_train, X_test, Y_test)

# Show softmax outputs for 10 test images
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
sample_idx = np.random.choice(X_test.shape[0], 10, replace=False)

for i, idx in enumerate(sample_idx):
    ax = axes[i // 5, i % 5]
    activations = entree_sortie_reseau(dnn_best, X_test[idx:idx+1])
    probs = activations[-1][0]
    true_label = np.argmax(Y_test[idx])
    pred_label = np.argmax(probs)

    ax.bar(range(10), probs, color=["green" if j == true_label else "gray" for j in range(10)])
    ax.set_title(f"True: {true_label}, Pred: {pred_label}", fontsize=9)
    ax.set_xticks(range(10))
    ax.set_ylim(0, 1)

plt.suptitle("Softmax Output Probabilities (green = true label)")
plt.tight_layout()
plt.savefig("softmax_probs.png", dpi=150, bbox_inches="tight")
plt.show()

# Show corresponding images
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, idx in enumerate(sample_idx):
    ax = axes[i // 5, i % 5]
    ax.imshow(X_test[idx].reshape(28, 28), cmap="gray")
    ax.set_title(f"Label: {np.argmax(Y_test[idx])}", fontsize=9)
    ax.axis("off")
plt.suptitle("Test Images")
plt.tight_layout()
plt.savefig("test_images.png", dpi=150, bbox_inches="tight")
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

final_err = test_DNN(dnn_best, X_test, Y_test)
print(f"\nBest model error rate: {final_err*100:.2f}%")
