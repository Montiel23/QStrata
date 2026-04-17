import torch

# def realistic_homodyne_readout(mu, cov, detector_efficiency=0.9, electronic_noise=0.05):

#     # base physics value
#     val = mu[0] ** 2 + cov[0,0]

#     #apply efficiency
#     val = val * detector_efficiency

#     #apply electronic noise (random jitter)
#     noise = torch.randn_like(val) * electronic_noise

#     return val + noise

def realistic_homodyne_readout(mu, cov, mode, detector_efficiency=0.9, electronic_noise=0.05):
    "computes <x^2> for a specific mode with noise constraints"

    #mu indices are 2*i for X, 2*i + 1 for P
    #cov indices are [2*i, 2*i] for Var(X)

    mean_x = mu[2 * mode]
    var_x = cov[2 * mode, 2 * mode]

    # base physics: <X^2> = <X>^2 + Var(X)
    val = (mean_x ** 2) + var_x

    #apply hardware constraints
    val = val * detector_efficiency
    noise = torch.randn_like(val) * electronic_noise

    return val + noise

    
