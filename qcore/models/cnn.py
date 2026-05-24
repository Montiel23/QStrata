"""
qcore/models/cnn.py — Config-driven CNN builder.

Public API:
  build_block(block_type, in_ch, out_ch) -> nn.Sequential
  build_model(config)                    -> nn.Module
"""

import torch.nn as nn


def build_block(block_type: str, in_ch: int, out_ch: int) -> nn.Sequential:
    """
    Build a single convolutional block.

    Parameters
    ----------
    block_type : str
        "standard"      — Conv2d → BatchNorm2d → ReLU
        "depthwise_sep" — depthwise Conv2d → BN → ReLU → pointwise Conv2d → BN → ReLU
    in_ch : int
        Number of input channels.
    out_ch : int
        Number of output channels.

    Returns
    -------
    nn.Sequential
    """
    if block_type == "standard":
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    if block_type == "depthwise_sep":
        return nn.Sequential(
            # Depthwise: spatial filtering, one filter per input channel
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(),
            # Pointwise: channel mixing, 1×1 projection to out_ch
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    raise ValueError(
        f"Unsupported block_type '{block_type}'. "
        "Supported: 'standard', 'depthwise_sep'."
    )


def build_model(config: dict) -> nn.Module:
    """
    Build a binary CNN from a flat config dict.

    Expected keys
    -------------
    conv_channels  : list[int]  — number of filters per conv block
    use_batchnorm  : bool       — controls BatchNorm in standard block (ignored
                                  for depthwise_sep which always uses BatchNorm)
    dropout        : float      — dropout rate before final Linear (0 = skip)
    pooling        : str        — "adaptive_avg" (only supported value)
    input_channels : int        — number of input image channels
    num_classes    : int        — number of output classes
    block_type     : str        — (optional) "standard" (default) or "depthwise_sep"

    Backward compatibility
    ----------------------
    When block_type is absent from config, behaviour is functionally identical
    to the original implementation (standard conv + BatchNorm driven by
    use_batchnorm).

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
    block_type    = config.get("block_type", "standard")

    if pooling != "adaptive_avg":
        raise ValueError(
            f"Unsupported pooling type '{pooling}'. "
            "Only 'adaptive_avg' is supported in this slice."
        )

    layers = []

    current_in = in_channels
    for out_channels in conv_channels:
        if block_type == "standard":
            # Preserve original behavior: BatchNorm conditional on use_batchnorm
            layers.append(nn.Conv2d(current_in, out_channels, kernel_size=3, padding=1))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
        else:
            layers.append(build_block(block_type, current_in, out_channels))
        current_in = out_channels

    layers.append(nn.AdaptiveAvgPool2d((1, 1)))
    layers.append(nn.Flatten())

    if dropout > 0.0:
        layers.append(nn.Dropout(p=dropout))

    layers.append(nn.Linear(current_in, num_classes))

    return nn.Sequential(*layers)
