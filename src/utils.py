import os
import random
import numpy as np
import torch


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def save_checkpoint(
    model,
    optimizer,
    epoch,
    loss,
    path
):

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss
        },
        path
    )


def load_checkpoint(
    model,
    optimizer,
    path,
    device
):

    checkpoint = torch.load(
        path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    return checkpoint


def ensure_dir(path):

    os.makedirs(
        path,
        exist_ok=True
    )


def print_model_summary(model):

    params = count_parameters(
        model
    )

    print("=" * 50)
    print("Model Summary")
    print("=" * 50)
    print(
        f"Trainable Parameters: {params:,}"
    )
    print("=" * 50)