import json
import random
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset


class SpeechSeparationDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        sample_rate=16000,
        segment_seconds=4,
        max_sources=4,
        return_metadata=False,
    ):
        self.root = Path(root_dir) / split
        self.sample_rate = sample_rate
        self.segment_length = sample_rate * segment_seconds
        self.max_sources = max_sources
        self.return_metadata = return_metadata

        self.samples = sorted(self.root.glob("sample_*"))

        print(f"{split}: {len(self.samples)} samples found")

    def __len__(self):
        return len(self.samples)

    def random_crop(self, mixture, sources):
        total_length = mixture.shape[1]

        if total_length <= self.segment_length:
            pad = self.segment_length - total_length

            mixture = torch.nn.functional.pad(
                mixture,
                (0, pad)
            )

            sources = torch.nn.functional.pad(
                sources,
                (0, pad)
            )

            return mixture, sources

        start = random.randint(
            0,
            total_length - self.segment_length
        )

        end = start + self.segment_length

        return (
            mixture[:, start:end],
            sources[:, start:end]
        )

    def __getitem__(self, idx):
        sample_dir = self.samples[idx]

        mixture, sr = torchaudio.load(
            sample_dir / "mixture.flac"
        )

        # Optional: ensure sample rate is correct
        if sr != self.sample_rate:
            raise ValueError(
                f"Expected sample rate {self.sample_rate}, but got {sr}"
            )

        source_files = sorted(
            sample_dir.glob("source_*.flac")
        )

        source_list = []

        for file in source_files:
            audio, _ = torchaudio.load(file)
            source_list.append(audio)

        # Pad with silent sources until max_sources
        while len(source_list) < self.max_sources:
            source_list.append(
                torch.zeros_like(mixture)
            )

        sources = torch.cat(source_list, dim=0)

        with open(sample_dir / "metadata.json", "r") as f:
            metadata = json.load(f)

        mixture, sources = self.random_crop(
            mixture,
            sources
        )

        sample = {
            "mixture": mixture.float(),
            "sources": sources.float(),
            "num_sources": metadata["num_speakers"],
        }

        if self.return_metadata:
            sample["speaker_ids"] = metadata["speaker_ids"]
            sample["overlap_ratio"] = metadata["overlap_ratio"]
            sample["duration"] = metadata["mix_duration"]

        return sample