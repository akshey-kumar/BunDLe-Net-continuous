# BunDLe-Net-continuous

Behavioural and Dynamic Learning Network (BunDLe Net) is an algorithm to learn meaningful coarse-grained representations from time-series data. It maps high-dimensional data to low-dimensional space while preserving both dynamical and behavioural information. It has been applied, but is not limited, to neuronal manifold learning. 

This is an implementation of the BunDLe-Net architecture for continuous-valued and multidimensional behaviours. In `notebooks/`, you can find BunDLe-Net deployed on rat and primate electrophysiological data which forms part of the results for the journal paper:[https://www.biorxiv.org/content/10.1101/2023.08.08.551978v2](https://www.biorxiv.org/content/10.1101/2023.08.08.551978v2)


After creating a new virtual environment, to install dependencies, you can run
```bash
python3 -m pip install -r requirements.txt
```

**BunDLe-Net embedding of *C.elegans* neuronal data in 3-D latent space**

![BunDLe-Net embedding of C.elegans neuronal data in 3-D latent space](https://github.com/akshey-kumar/BunDLe-Net/blob/main/figures/rotation_comparable_embeddings/rotation_BunDLeNet_worm_0.gif)
