# Image Clustering on CIFAR-10: ResNet-50 vs CLIP Features

An unsupervised-learning experiment that compares how well K-Means and Gaussian Mixture
Models recover the 10 CIFAR-10 classes from three feature sets: ImageNet-pretrained
ResNet-50, OpenAI CLIP (ViT-B/32) image embeddings, and the two combined.

**Project notebook:** [Open the complete Jupyter notebook](https://github.com/sachin2967/Image-Clustering/blob/main/code_cifar10_updated.ipynb).
This is a reproducible notebook project rather than a hosted interactive application; see [Run](#run) for local execution.

## Method

1. Load CIFAR-10 (train + test = 60,000 images) with torchvision.
2. Extract features with ResNet-50 (`IMAGENET1K_V1`, final classifier removed) and with
   `openai/clip-vit-base-patch32` image embeddings.
3. Standardize, then reduce each set with PCA to the number of components that explain 95%
   of the variance (263).
4. Cluster with K-Means (`k=10`, `n_init=10`, seed 42) and a Gaussian Mixture Model.
5. Score agreement with the true labels using Cohen's kappa, and visualize with t-SNE.

## Results (as printed by the notebook)

| Features | K-Means kappa | GMM kappa |
| --- | ---: | ---: |
| ResNet-50 | 0.3025 | 0.1249 |
| **CLIP** | **0.8432** | **0.6081** |
| Combined | 0.5843 | 0.2706 |

CLIP embeddings separate CIFAR-10 classes far better than ResNet-50 features, and simply
concatenating the two did not help. These come from a single run with one seed; the
notebook does not report confidence intervals or other clustering metrics, and the labels
are used only to evaluate, not to train.

## Run

A GPU is recommended (the notebook was run on CUDA); it also runs on CPU, slowly.

```bash
git clone https://github.com/sachin2967/Image-Clustering.git
cd Image-Clustering
python -m venv .venv && source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter notebook code_cifar10_updated.ipynb
```

The CIFAR-10 data and the pretrained weights are downloaded automatically on first run.

## Files

| File | Purpose |
| --- | --- |
| `code_cifar10_updated.ipynb` | Full experiment: features, PCA, clustering, evaluation, t-SNE |
| `requirements.txt` | Python dependencies |

## License

MIT
