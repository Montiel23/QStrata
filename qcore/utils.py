import torch
import datetime
import os

def kron(a, b):
    return torch.kron(a, b)


def bce(pred, target):
    eps = 1e-6
    return -(target * torch.log(pred+eps) + (1-target) * torch.log(1-pred+eps))


def make_run_dir_from_config(config):

    base_dir = os.path.join("results")

    name_parts = []

    name_parts.append(f"{config['n_qubits']}q")

    # if config.get("entangle", False):
    #     name_parts.append("ent")

    name_parts.append(f"depth{config['depth']}")
    name_parts.append(f"m{config['measure_wire']}")
    name_parts.append(f"a{config['alpha']}")
    name_parts.append(f"d{config['dataset']}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    run_name = "_".join(name_parts) + "_" + timestamp

    run_dir = os.path.join(base_dir, run_name)

    os.makedirs(run_dir, exist_ok=True)

    return run_dir