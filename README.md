# BunDLe-Net-continuous

Behavioural and Dynamic Learning Network (BunDLe-Net) is an algorithm to learn meaningful coarse-grained representations from time-series data. It maps high-dimensional data to low-dimensional space while preserving both dynamical and behavioural information. It has been applied, but is not limited, to neuronal manifold learning. 

This is an implementation of the BunDLe-Net architecture for continuous-valued and multidimensional behaviours. In `notebooks/`, you can find BunDLe-Net deployed on rat and primate electrophysiological data which forms part of the results for the journal paper:[https://www.biorxiv.org/content/10.1101/2023.08.08.551978v2](https://www.biorxiv.org/content/10.1101/2023.08.08.551978v2)

For the discrete version of BunDLe-Net, please see [this repo](https://github.com/akshey-kumar/BunDLe-Net).

# Project Setup Instructions

Follow these steps to set up the virtual environment for this project.

## Prerequisites

Ensure that you have Python version 3.10.12 installed on your system.

## Setting Up the Virtual Environment

1. Create the Virtual Environment

   ```bash
   python3.10 -m venv .env
   ```
   

2. Activate the virtual environment and check the python version
    ```bash
    source .env/bin/activate
    python --version
    ```
   You should see,
    ```
    Python 3.10.12
    ```
3. Install the requirements
    ```bash
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt 
    ```
