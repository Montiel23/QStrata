"""
qcore/data/registry.py — Dataset selector / factory.

Usage:
    from qcore.data.registry import get_dataset
    dataset = get_dataset("pneumoniamnist", "train")
"""

from qcore.data.medmnist_dataset import MedMNISTDataset, SUPPORTED_DATASETS

SUPPORTED_SPLITS = ("train", "val", "test")


def get_dataset(name: str, split: str, **kwargs) -> MedMNISTDataset:
    """
    Return a MedMNISTDataset for the given dataset name and split.

    Parameters
    ----------
    name : str
        Dataset name. One of: pneumoniamnist, pathmnist.
    split : str
        One of: train, val, test.
    **kwargs
        Forwarded to MedMNISTDataset (e.g. download=False).

    Raises
    ------
    ValueError
        If name or split is not supported.
    """
    if name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset '{name}'. "
            f"Supported datasets: {SUPPORTED_DATASETS}"
        )
    if split not in SUPPORTED_SPLITS:
        raise ValueError(
            f"Unsupported split '{split}'. "
            f"Supported splits: {SUPPORTED_SPLITS}"
        )

    return MedMNISTDataset(
        dataset_name=name,
        split=split,
        download=kwargs.get("download", True),
    )
