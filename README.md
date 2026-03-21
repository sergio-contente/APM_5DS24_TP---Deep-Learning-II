# APM_5DS24_TP — Deep Learning II

RBM, DBN and DNN from scratch (NumPy) for MNIST digit classification, with unsupervised pre-training via greedy layer-wise RBM training.

## Structure

- `rbm.py` — RBM: init, CD-1 training, Gibbs sampling image generation
- `dbn.py` — DBN: greedy layer-wise training, top-down image generation
- `dnn.py` — DNN: pre-training via DBN, backpropagation, softmax classification
- `utils.py` — Data loading (Binary AlphaDigits + MNIST binarized)

### Main scripts

- `principal_RBM_alpha.py` — RBM experiments on Binary AlphaDigits (varying hidden units and character diversity)
- `principal_DBN_alpha.py` — DBN experiments on Binary AlphaDigits (varying layers and character diversity)
- `principal_DNN_MNIST.py` — Full MNIST study: pre-trained vs random init, generates 3 figures (error vs layers, neurons, training samples)
- `bonus_comparison.ipynb` — Bonus: comparison of 5 generative models (RBM, DBN, VAE, GAN, DDPM) on MNIST with ~160k params each

## How to run

```bash
# Binary AlphaDigits experiments
python principal_RBM_alpha.py
python principal_DBN_alpha.py

# MNIST study (takes a while, ~14 network pairs)
python principal_DNN_MNIST.py

# Bonus (Jupyter notebook, recommended on Google Colab with GPU)
# Open bonus_comparison.ipynb in Colab and run all cells
```

## Dependencies

- NumPy, SciPy, Matplotlib (core code)
- PyTorch, torchvision (bonus notebook only)
