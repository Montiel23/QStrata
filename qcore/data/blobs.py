import torch
def make_blobs(n=200):
    c0 = torch.randn(n//2, 2) * 0.5 + torch.tensor([-1.0, -1.0])
    c1 = torch.randn(n//2, 2) * 0.5 + torch.tensor([1.0, 1.0])

    X = torch.cat([c0, c1], dim=0)
    y = torch.cat([torch.zeros(n//2), torch.ones(n//2)])

    return X, y
