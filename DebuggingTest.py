import os
import warnings

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import seaborn as sns
import xarray as xr

def stick_breaking(beta):
    portion_remaining = pt.concatenate([[1], pt.extra_ops.cumprod(1 - beta)[:-1]])
    return beta * portion_remaining

# Settings for Gaussian Mixture Model with dirichlet distribution
df = pd.read_csv('mixture.csv')
X = df['X'].values[:, None]
Y = df['Y'].values
k = 2 # Set k to small value for testing
RANDOM_SEED = 42

with pm.Model(coords={"component": np.arange(k)}) as Mixture_regression1:
    # Priors for weight parameter
    alpha = pm.Gamma("alpha", 1.0, 1.0)
    beta = pm.Beta("beta", 10.0, alpha, dims="component", initval=np.ones(k) / k)  # Generate beta values
    π = pm.Deterministic("π", stick_breaking(beta), dims="component")  # Component weights

    # Priors for unknown model parameters
    α = pm.Normal('α', mu=0, sigma=100, shape=(1, k))  # Intercept
    β = pm.Normal('β', mu=0, sigma=100, shape=(1, k))

    σ = pm.HalfCauchy('σ', 5, shape=k)  # Noise

    mu = α + β*X

    likelihood = pm.NormalMixture('likelihood', π, mu, sigma=σ, observed=Y)
    trace = pm.sample(
        tune=2500,
        init="advi",
        target_accept=0.975,
        random_seed=RANDOM_SEED
    )

