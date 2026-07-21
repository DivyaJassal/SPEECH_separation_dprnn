import argparse
import os
import torch
import torchaudio

from config import load_config
from dprnn import DPRNNTasNet


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_model(device):

    cfg = load_config()

    model = DPRNNTasNet(
        num_sources=cfg["max_sources"],
        encoder_channels=cfg["model"]["encoder_dim"],
        hidden_size=cfg["model"]["hidden_size"],
        kernel_size=cfg["model"]["encoder_kernel_size"],
        stride=cfg["model"]["encoder_stride"],
        num_blocks=cfg["model"]["num_blocks"],
        chunk_size=cfg["model"]["chunk_sizes"][1]
    )

    checkpoint = torch.load(
        "../checkpoints/best_dprnn.pt",
        map_location=device
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    return model, cfg


def separate_audio(audio_path, output_dir):

    device = get_device()

    model, cfg = load_model(device)

    waveform, sr = torchaudio.load(audio_path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    waveform = waveform.to(device)

    with torch.no_grad():
        estimates = model(waveform)

    estimates = estimates.squeeze(0).cpu()

    os.makedirs(output_dir, exist_ok=True)

    for i, source in enumerate(estimates):

        path = os.path.join(
            output_dir,
            f"speaker_{i+1}.wav"
        )

        torchaudio.save(
            path,
            source.unsqueeze(0),
            sr
        )

        print(path)

    print("\nFinished!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--audio",
        required=True
    )

    parser.add_argument(
        "--output",
        default="outputs"
    )

    args = parser.parse_args()

    separate_audio(
        args.audio,
        args.output
    )