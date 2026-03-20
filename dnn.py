import numpy as np
import copy
from rbm import sigmoid, init_RBM, entree_sortie_RBM
from dbn import init_DBN, train_DBN


def init_DNN(layer_sizes):
    """Initialize DNN: DBN for hidden layers + classification (softmax) layer.
    E.g. [784, 200, 200, 10] -> 2 hidden RBMs + 1 softmax RBM."""
    # Hidden layers (all except first and last define the DBN)
    dbn_sizes = layer_sizes[:-1]  # e.g. [784, 200, 200]
    dbn = init_DBN(dbn_sizes)

    # Add classification layer
    last_hidden = layer_sizes[-2]
    n_classes = layer_sizes[-1]
    softmax_rbm = init_RBM(last_hidden, n_classes)
    dbn.append(softmax_rbm)

    return dbn


def pretrain_DNN(dnn, data, epochs=100, lr=0.1, batch_size=128, verbose=True):
    """Pre-train hidden layers (excluding softmax) via greedy layer-wise RBM training."""
    hidden_rbms = dnn[:-1]  # All but the last (softmax) layer
    if verbose:
        print(f"Pre-training {len(hidden_rbms)} hidden layer(s)...")
    train_DBN(hidden_rbms, data, epochs=epochs, lr=lr, batch_size=batch_size, verbose=verbose)


def calcul_softmax(rbm, data):
    """Compute softmax: P(y|h) = exp(h*W+b) / sum(exp(h*W+b))."""
    logits = data @ rbm["W"] + rbm["b"]
    # Numerical stability
    logits -= np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def entree_sortie_reseau(dnn, data):
    """Full forward pass through DNN.
    Returns list of activations at each layer + final softmax probabilities.
    activations[0] = input data
    activations[i] = output of layer i (sigmoid for hidden, softmax for last)
    """
    activations = [data]
    current = data

    # Hidden layers (sigmoid)
    for i in range(len(dnn) - 1):
        current = entree_sortie_RBM(dnn[i], current)
        activations.append(current)

    # Output layer (softmax)
    probs = calcul_softmax(dnn[-1], current)
    activations.append(probs)

    return activations


def retropropagation(dnn, data, labels, epochs=200, lr=0.1, batch_size=128, verbose=True):
    """Train DNN via backpropagation with cross-entropy loss."""
    n = data.shape[0]
    losses = []

    for epoch in range(epochs):
        perm = np.random.permutation(n)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            X_batch = data[idx]
            Y_batch = labels[idx]
            bs = X_batch.shape[0]

            # Forward pass
            activations = entree_sortie_reseau(dnn, X_batch)

            # Cross-entropy loss
            probs = activations[-1]
            loss = -np.mean(np.sum(Y_batch * np.log(probs + 1e-10), axis=1))
            epoch_loss += loss
            n_batches += 1

            # Backward pass
            delta = probs - Y_batch  # (bs, n_classes)

            L = len(dnn)
            for l in range(L - 1, -1, -1):
                h_prev = activations[l]  # input to this layer

                # Compute gradient for weights and biases
                grad_W = (h_prev.T @ delta) / bs
                grad_b = np.mean(delta, axis=0)

                # Propagate delta backward BEFORE updating weights
                if l > 0:
                    h_l = activations[l]
                    delta = (delta @ dnn[l]["W"].T) * h_l * (1 - h_l)

                # Now update weights
                dnn[l]["W"] -= lr * grad_W
                dnn[l]["b"] -= lr * grad_b

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Backprop Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")

    return losses


def test_DNN(dnn, data_test, labels_test):
    """Evaluate DNN on test set. Returns error rate."""
    activations = entree_sortie_reseau(dnn, data_test)
    probs = activations[-1]
    predictions = np.argmax(probs, axis=1)
    true_labels = np.argmax(labels_test, axis=1)
    error_rate = np.mean(predictions != true_labels)
    return error_rate
