# Single File Variational Autoencoder (VAE) Example on MNIST

A Python script implements a Variational Autoencoder (VAE) using PyTorch to process MNIST handwritten digits. The model learns a latent representation and reconstructs images using a reparameterization trick. Additionally, Network Architecture Search (NAS) is performed to experiment with different latent space dimensions, helping understand the relation between VAE's performance and latent space size.

More details see [this blog post](https://www.gong.host/blog/2023/12/25/vae-from-scratch).

---

## Run

```sh
pip install torch torchvision neetbox
python vae.py
```

The script will download the MNIST dataset and train the VAE model. The training process will be visualized in browser.

---

## Visualization

Start training and open neetbox dashboard at `http://localhost:20202`.

![screenshot](https://www.gong.host/assets/images/image-20231228150104010-9b9d99eb108f2351f0a449964a136978.png)