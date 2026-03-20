import numpy as np
import scipy.io as sio
import struct
import os

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
MNIST_DIR = os.path.join(DATA_DIR, "Fichiers MNIST-20260320")
ALPHA_PATH = os.path.join(DATA_DIR, "binaryalphadigs.mat")


def lire_alpha_digit(characters):
    """Load Binary AlphaDigits data for given character indices (0-35).
    Returns matrix (n_samples, 320) where each row is a flattened 20x16 image."""
    mat = sio.loadmat(ALPHA_PATH)
    dat = mat["dat"]  # shape (36, 39), each entry is a 20x16 array
    images = []
    for c in characters:
        for i in range(dat.shape[1]):
            img = dat[c, i].flatten().astype(np.float64)
            images.append(img)
    return np.array(images)


def _read_idx_images(path):
    with open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows * cols).astype(np.float64)


def _read_idx_labels(path):
    with open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data


def load_mnist():
    """Load MNIST, binarize images (threshold 127), one-hot encode labels.
    Returns (X_train, Y_train, X_test, Y_test)."""
    X_train = _read_idx_images(os.path.join(MNIST_DIR, "train-images-idx3-ubyte"))
    y_train = _read_idx_labels(os.path.join(MNIST_DIR, "train-labels-idx1-ubyte"))
    X_test = _read_idx_images(os.path.join(MNIST_DIR, "t10k-images-idx3-ubyte"))
    y_test = _read_idx_labels(os.path.join(MNIST_DIR, "t10k-labels-idx1-ubyte"))

    # Binarize
    X_train = (X_train >= 127).astype(np.float64)
    X_test = (X_test >= 127).astype(np.float64)

    # One-hot encode
    Y_train = np.zeros((len(y_train), 10))
    Y_train[np.arange(len(y_train)), y_train] = 1.0
    Y_test = np.zeros((len(y_test), 10))
    Y_test[np.arange(len(y_test)), y_test] = 1.0

    return X_train, Y_train, X_test, Y_test
