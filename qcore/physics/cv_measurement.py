def realistic_homodyne_readout(mu, cov, detector_efficiency=0.9, electronic_noise=0.05):

    # base physics value
    val = mu[0] ** 2 + cov[0,0]

    #apply efficiency
    val = val * detector_efficiency

    #apply electronic noise (random jitter)
    noise = torch.randn_like(val) * electronic_noise

    return val + noise