# APM_5DS24_TP — Deep Learning II

RBM, DBN and DNN from scratch (NumPy) for MNIST digit classification, with unsupervised pre-training via greedy layer-wise RBM training.

## Project Structure

```
dl2/
├── src/                              # Source modules
│   ├── __init__.py
│   ├── rbm.py                        # RBM: init, CD-1 training, Gibbs sampling
│   ├── dbn.py                        # DBN: greedy layer-wise training, generation
│   ├── dnn.py                        # DNN: pre-training, backprop, softmax, testing
│   └── utils.py                      # Data loading (AlphaDigits + MNIST)
├── experiments/                      # Runnable scripts
│   ├── principal_RBM_alpha.py        # RBM experiments on Binary AlphaDigits
│   ├── principal_DBN_alpha.py        # DBN experiments on Binary AlphaDigits
│   ├── principal_DNN_MNIST.py        # MNIST study: pre-trained vs random (3 figures)
│   ├── principal_best_model.py       # Best config from experimental analysis
│   └── principal_bonus.py            # Bonus: generative models comparison (local)
├── notebooks/                        # Jupyter notebooks
│   └── bonus_comparison.ipynb        # Bonus: RBM vs DBN vs VAE vs GAN vs DDPM
├── images/                           # Generated figures
├── data/                             # Datasets (not included in archive)
│   ├── binaryalphadigs.mat
│   └── Fichiers MNIST-20260320/
├── report/                           # Report and assignment
│   └── TP_DNN.pdf
├── requirements.txt
└── README.md
```

## How to run

```bash
# Binary AlphaDigits experiments
python experiments/principal_RBM_alpha.py
python experiments/principal_DBN_alpha.py

# MNIST study (generates 3 comparison figures)
python experiments/principal_DNN_MNIST.py

# Best model with softmax visualization
python experiments/principal_best_model.py

# Bonus (Jupyter notebook, recommended on Google Colab with GPU)
# Open notebooks/bonus_comparison.ipynb in Colab and run all cells
```

## Dependencies

- NumPy, SciPy, Matplotlib (core code)
- PyTorch, torchvision (bonus only)
