import torch
from losses import pit_si_sdr_loss


def main():

    estimated = torch.randn(
        6,4,64000
    )

    target = torch.randn(
        6,4,64000
    )

    loss = pit_si_sdr_loss(
        estimated,
        target
    )

    print(loss)


if __name__ == "__main__":
    main()