import torch
from torch.optim import Adam

from config import load_config
from dataloader import create_train_loader, create_val_loader
from dprnn import DPRNNTasNet
from trainer import Trainer


def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def main():

    config = load_config()

    device = get_device()

    print("Using device:", device)

    train_loader = create_train_loader(config)
    val_loader = create_val_loader(config)

    model = DPRNNTasNet(
        num_sources=config["max_sources"],
        encoder_channels=config["model"]["encoder_dim"],
        hidden_size=config["model"]["hidden_size"],
        kernel_size=config["model"]["encoder_kernel_size"],
        stride=config["model"]["encoder_stride"],
        num_blocks=config["model"]["num_blocks"],
        chunk_size=config["model"]["chunk_sizes"][0]
    )

    model = model.to(device)

    optimizer = Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        config=config
    )

    trainer.fit(
        config["epochs"]
    )


if __name__ == "__main__":
    main()