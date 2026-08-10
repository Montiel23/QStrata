import torch
import numpy as np

# def realistic_homodyne_readout(mu, cov, mode, detector_efficiency=0.9, electronic_noise=0.05):
#     "computes <x^2> for a specific mode with noise constraints"

#     #mu indices are 2*i for X, 2*i + 1 for P
#     #cov indices are [2*i, 2*i] for Var(X)

#     mean_x = mu[2 * mode]
#     var_x = cov[2 * mode, 2 * mode]

#     # base physics: <X^2> = <X>^2 + Var(X)
#     val = (mean_x ** 2) + var_x

#     #apply hardware constraints
#     val = val * detector_efficiency
#     noise = torch.randn_like(val) * electronic_noise

#     return val + noise


def realistic_homodyne_readout(mu, cov, mode, angle=0.0, detector_efficiency=0.9, electronic_noise=0.05):
    # computes measurement for a quadrature at a given angle

    #extract raw quadratures
    x_idx = 2 * mode
    p_idx = 2 * mode + 1

    m_x, m_p = mu[x_idx], mu[p_idx]

    V_xx = cov[x_idx, x_idx]
    V_pp = cov[p_idx, p_idx]
    V_xp = cov[x_idx, p_idx]

    #project onto chosen angle theta
    #mean of quadrature q_theta = xcos(theta) + psin(theta)
    cos_t = np.cos(angle)
    sin_t = np.sin(angle)

    mean_theta = m_x * cos_t + m_p * sin_t

    #variance of quadrature: var(q_theta)
    var_theta = (cos_t**2 * V_xx +
                 sin_t**2 * V_pp +
                 2 * cos_t * sin_t + V_xp)
    
    # base physics signal = q_theta^2 + var(q_theta)
    #using the mean is better for classification
    #keeping the second moment matches previous logic
    val = (mean_theta ** 2) + var_theta

    #apply hardware constraints
    val = val * detector_efficiency
    noise = torch.randn_like(val) * electronic_noise

    return val + noise



# =====================================================================
# 3. DUAL HOMODYNE READOUT
# =====================================================================
def realistic_homodyne_readout_v2(
    mu,
    cov,
    mode,
    angle=0.0,
    detector_efficiency=0.9,
    electronic_noise=0.05,
):
    """Computes second-moment quadrature measurement <q_theta^2> with hardware noise."""
    device = mu.device
    x_idx = 2 * mode
    p_idx = 2 * mode + 1

    m_x, m_p = mu[..., x_idx], mu[..., p_idx]
    V_xx = cov[..., x_idx, x_idx]
    V_pp = cov[..., p_idx, p_idx]
    V_xp = cov[..., x_idx, p_idx]

    if not isinstance(angle, torch.Tensor):
        angle = torch.tensor(angle, device=device, dtype=torch.float32)

    cos_t = torch.cos(angle)
    sin_t = torch.sin(angle)

    mean_theta = m_x * cos_t + m_p * sin_t
    var_theta = (
        cos_t**2 * V_xx + sin_t**2 * V_pp + 2.0 * cos_t * sin_t * V_xp
    )

    val = (mean_theta**2) + var_theta
    val = val * detector_efficiency

    if mu.requires_grad or cov.requires_grad:
        noise = torch.randn_like(val) * electronic_noise
    else:
        noise = 0.0

    return val + noise