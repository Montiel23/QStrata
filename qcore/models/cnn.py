"""
qcore/models/cnn.py — Config-driven CNN builder.

Public API: build_model(config) -> nn.Module
"""

import torch.nn as nn


def build_model(config: dict) -> nn.Module:
    """
    Build a binary CNN from a flat config dict.

    Expected keys
    -------------
    conv_channels  : list[int]  — number of filters per conv block
    use_batchnorm  : bool       — insert BatchNorm2d after each Conv2d
    dropout        : float      — dropout rate before final Linear (0 = skip)
    pooling        : str        — "adaptive_avg" (only supported value)
    input_channels : int        — number of input image channels
    num_classes    : int        — number of output classes

    Returns
    -------
    nn.Sequential — model not yet moved to a device; caller calls .to(DEVICE).
    """
    conv_channels = config["conv_channels"]
    use_batchnorm = config["use_batchnorm"]
    dropout       = config["dropout"]
    pooling       = config["pooling"]
    in_channels   = config["input_channels"]
    num_classes   = config["num_classes"]

    if pooling != "adaptive_avg":
        raise ValueError(
            f"Unsupported pooling type '{pooling}'. "
            "Only 'adaptive_avg' is supported in this slice."
        )

    layers = []

    current_in = in_channels
    for out_channels in conv_channels:
        layers.append(nn.Conv2d(current_in, out_channels, kernel_size=3, padding=1))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        current_in = out_channels

    layers.append(nn.AdaptiveAvgPool2d((1, 1)))
    layers.append(nn.Flatten())

    if dropout > 0.0:
        layers.append(nn.Dropout(p=dropout))

    layers.append(nn.Linear(current_in, num_classes))

    return nn.Sequential(*layers)
