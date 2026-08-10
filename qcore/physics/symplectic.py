import torch


# =====================================================================
# 1. SYMPLECTIC GATE GENERATORS
# =====================================================================
def get_rotation_matrix(n_modes, mode, phi, device="cpu"):
    """R(phi) = exp(i * phi * a_dag * a)"""
    S = torch.eye(2 * n_modes, device=device)
    c, s = torch.cos(phi), torch.sin(phi)
    S[2 * mode, 2 * mode] = c
    S[2 * mode, 2 * mode + 1] = -s
    S[2 * mode + 1, 2 * mode] = s
    S[2 * mode + 1, 2 * mode + 1] = c
    return S


def get_beamsplitter_matrix(n_modes, m1, m2, theta, device="cpu"):
    """BS(theta) mixes two modes m1 and m2"""
    S = torch.eye(2 * n_modes, device=device)
    c, s = torch.cos(theta), torch.sin(theta)

    # Position X quadratures
    S[2 * m1, 2 * m1], S[2 * m1, 2 * m2] = c, s
    S[2 * m2, 2 * m1], S[2 * m2, 2 * m2] = -s, c

    # Momentum P quadratures
    S[2 * m1 + 1, 2 * m1 + 1], S[2 * m1 + 1, 2 * m2 + 1] = c, s
    S[2 * m2 + 1, 2 * m1 + 1], S[2 * m2 + 1, 2 * m2 + 1] = -s, c
    return S


def get_squeezing_matrix(n_modes, mode, r, device="cpu"):
    """S(r) scales quadratures: X -> X*e^-r, P -> P*e^r"""
    r = torch.as_tensor(r, device=device, dtype=torch.float32)
    S = torch.eye(2 * n_modes, device=device)

    S[2 * mode, 2 * mode] = torch.exp(-r)
    S[2 * mode + 1, 2 * mode + 1] = torch.exp(r)
    return S


def get_displacement_vector(n_modes, mode, alpha, hbar=2.0, device="cpu"):
    """D(alpha) translates phase-space displacement vector."""
    S = torch.zeros(2 * n_modes, device=device)
    scale = torch.sqrt(torch.tensor(2.0 * hbar, device=device))

    if torch.is_complex(alpha):
        S[2 * mode] = alpha.real * scale
        S[2 * mode + 1] = alpha.imag * scale
    else:
        S[2 * mode] = alpha * scale

    return S