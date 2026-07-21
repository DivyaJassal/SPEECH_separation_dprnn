import torch

from config import load_config
from dataloader import create_train_loader
from dprnn import DPRNNTasNet


def main():

    config = load_config()

    train_loader = create_train_loader(
        config
    )

    batch = next(iter(train_loader))

    mixture = batch["mixture"]

    sources = batch["sources"]

    print("Input mixture:")
    print(mixture.shape)

    print("Target sources:")
    print(sources.shape)

    model = DPRNNTasNet(
    num_sources=config["max_sources"],
    encoder_channels=config["model"]["encoder_dim"],
    hidden_size=config["model"]["hidden_size"],
    kernel_size=config["model"]["encoder_kernel_size"],
    stride=config["model"]["encoder_stride"],
    num_blocks=config["model"]["num_blocks"],
    chunk_size=config["model"]["chunk_sizes"][0]
)

    output = model(
        mixture
    )

    print("Model output:")
    print(output.shape)


if __name__ == "__main__":
    main()