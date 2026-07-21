import os
import itertools

import torch
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

from config import load_config
from dataloader import create_test_loader
from dprnn import DPRNNTasNet
from losses import pairwise_si_sdr


def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_model(config, device):

    model = DPRNNTasNet(

        num_sources=config["max_sources"],

        encoder_channels=config["model"]["encoder_dim"],

        hidden_size=config["model"]["hidden_size"],

        kernel_size=config["model"]["encoder_kernel_size"],

        stride=config["model"]["encoder_stride"],

        num_blocks=config["model"]["num_blocks"],

        chunk_size=config["model"]["chunk_sizes"][0],

    ).to(device)

    checkpoint = torch.load(

        os.path.join(
            config["checkpoint_dir"],
            "best_dprnn.pt"
        ),

        map_location=device

    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model
def evaluate():

    config = load_config()

    device = get_device()

    print(f"Using device: {device}")

    test_loader = create_test_loader(config)

    model = load_model(
        config,
        device
    )

    results = []

    with torch.no_grad():

        for batch in tqdm(
            test_loader,
            desc="Evaluating"
        ):

            mixture = batch["mixture"].to(device)

            sources = batch["sources"].to(device)

            estimates = model(
                mixture
            )

            batch_size = estimates.shape[0]

            num_sources = estimates.shape[1]

            for b in range(batch_size):

                est = estimates[
                    b:b+1
                ]

                tgt = sources[
                    b:b+1
                ]

                mix = mixture[
                    b:b+1
                ]

                pairwise = pairwise_si_sdr(
                    est,
                    tgt
                )[0]
                perms = list(
                    itertools.permutations(
                        range(num_sources)
                    )
                )

                best_score = None

                best_perm = None

                for perm in perms:

                    score = 0

                    for i in range(num_sources):

                        score += pairwise[
                            i,
                            perm[i]
                        ]

                    if (
                        best_score is None
                        or score > best_score
                    ):

                        best_score = score

                        best_perm = perm

                for i in range(num_sources):

                    target_index = best_perm[i]

                    est_score = pairwise[
                        i,
                        target_index
                    ]

                    mix_score = pairwise_si_sdr(

                        mix.unsqueeze(1),

                        tgt[:, target_index].unsqueeze(1)

                    )[0,0,0]

                    results.append(

                        {

                            "speaker": target_index + 1,

                            "si_sdr": est_score.item(),

                            "si_sdri": (
                                est_score -
                                mix_score
                            ).item()

                        }

                    )
                    os.makedirs(

        "reports",

        exist_ok=True

    )

    df = pd.DataFrame(
        results
    )

    df.to_csv(

        "reports/results_test.csv",

        index=False

    )

    summary = (

        df.groupby("speaker")[
            [
                "si_sdr",
                "si_sdri"
            ]
        ]

        .mean()

        .reset_index()

    )

    overall_si_sdr = df["si_sdr"].mean()

    overall_si_sdri = df["si_sdri"].mean()

    plt.figure(
        figsize=(8,5)
    )

    plt.bar(

        summary["speaker"].astype(str),

        summary["si_sdri"]

    )

    plt.xlabel(
        "Speaker"
    )

    plt.ylabel(
        "SI-SDRi (dB)"
    )

    plt.title(
        "Speech Separation Performance"
    )

    plt.grid(True)

    plt.savefig(

        "reports/results_test.png",

        dpi=300,

        bbox_inches="tight"

    )
    print()

    print("=" * 60)

    print("Evaluation Complete")

    print("=" * 60)

    print(summary)

    print()

    print(
        f"Overall SI-SDR : {overall_si_sdr:.2f} dB"
    )

    print(
        f"Overall SI-SDRi: {overall_si_sdri:.2f} dB"
    )

    print()

    print(
        "CSV saved to reports/results_test.csv"
    )

    print(
        "Plot saved to reports/results_test.png"
    )


if __name__ == "__main__":

    evaluate()