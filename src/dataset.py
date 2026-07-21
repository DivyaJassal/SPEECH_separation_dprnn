import json
import random
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset


class SpeechSeparationDataset(Dataset):

    def __init__(
        self,
        root_dir,
        split="train",
        sample_rate=16000,
        segment_seconds=4,
        max_sources=3,
        return_metadata=False,
        curriculum=False,
        curriculum_overlap=1.0,
        dynamic_chunk=False,
        chunk_choices=None,
    ):

        self.root = Path(root_dir) / split
        self.split = split
        self.sample_rate = sample_rate

        self.segment_seconds = segment_seconds
        self.segment_length = int(
            sample_rate * segment_seconds
        )

        self.max_sources = max_sources
        self.return_metadata = return_metadata

        self.curriculum = curriculum
        self.curriculum_overlap = curriculum_overlap

        self.dynamic_chunk = dynamic_chunk

        if chunk_choices is None:
            chunk_choices = [4]

        self.chunk_choices = chunk_choices

        self.samples = sorted(
            self.root.glob("sample_*")
        )

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples found inside {self.root}"
            )

        self.metadata_cache = []

        valid_samples = []

        for sample_dir in self.samples:

            metadata_file = sample_dir / "metadata.json"

            if not metadata_file.exists():
                continue

            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            if (
                self.curriculum
                and split == "train"
            ):

                overlap = metadata.get(
                    "overlap_ratio",
                    1.0
                )

                if overlap > self.curriculum_overlap:
                    continue

            valid_samples.append(sample_dir)
            self.metadata_cache.append(metadata)

        self.samples = valid_samples
        self.metadata_cache = self.metadata_cache[:2000]
        self.samples = self.samples[:2000]

        print(
            f"{split}: {len(self.samples)} usable samples found"
        )


    def __len__(self):
        return len(self.samples)


    def set_curriculum_overlap(self, overlap):

        self.curriculum_overlap = overlap

        if self.split != "train":
            return

        valid_samples = []
        valid_metadata = []

        for sample_dir in sorted(
            self.root.glob("sample_*")
        ):

            metadata_file = sample_dir / "metadata.json"

            if not metadata_file.exists():
                continue

            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            if metadata.get(
                "overlap_ratio",
                1.0
            ) <= overlap:

                valid_samples.append(sample_dir)
                valid_metadata.append(metadata)

        self.samples = valid_samples
        self.metadata_cache = valid_metadata


    def set_segment_seconds(self, seconds):

        self.segment_seconds = seconds

        self.segment_length = int(
            self.sample_rate * seconds
        )


    def update_dynamic_chunk(self):

        if (
            self.dynamic_chunk
            and self.split == "train"
        ):

            seconds = random.choice(
                self.chunk_choices
            )

            self.set_segment_seconds(
                seconds
            )


    def random_crop(
        self,
        mixture,
        sources
    ):

        total_length = mixture.shape[-1]

        if total_length <= self.segment_length:

            pad = self.segment_length - total_length

            mixture = F.pad(
                mixture,
                (0, pad)
            )

            sources = F.pad(
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
            mixture[..., start:end],
            sources[..., start:end]
        )


    def center_crop(
        self,
        mixture,
        sources
    ):

        total_length = mixture.shape[-1]

        if total_length <= self.segment_length:

            pad = self.segment_length - total_length

            mixture = F.pad(
                mixture,
                (0, pad)
            )

            sources = F.pad(
                sources,
                (0, pad)
            )

            return mixture, sources

        start = (
            total_length - self.segment_length
        ) // 2

        end = start + self.segment_length

        return (
            mixture[..., start:end],
            sources[..., start:end]
        )


    def load_sources(
        self,
        sample_dir,
        mixture
    ):

        source_files = sorted(
            sample_dir.glob("source_*.flac")
        )

        source_list = []

        for file in source_files:

            audio, sr = torchaudio.load(file)

            if sr != self.sample_rate:
                raise RuntimeError(
                    f"Unexpected sample rate {sr}"
                )

            source_list.append(audio)

        while len(source_list) < self.max_sources:

            source_list.append(
                torch.zeros_like(mixture)
            )

        source_list = source_list[
            :self.max_sources
        ]

        return torch.cat(
            source_list,
            dim=0
        )


    def load_mixture(
        self,
        sample_dir
    ):

        mixture, sr = torchaudio.load(
            sample_dir / "mixture.flac"
        )

        if sr != self.sample_rate:
            raise RuntimeError(
                f"Expected {self.sample_rate}, got {sr}"
            )

        return mixture


    def __getitem__(self, idx):

        self.update_dynamic_chunk()

        sample_dir = self.samples[idx]

        metadata = self.metadata_cache[idx]

        mixture = self.load_mixture(
            sample_dir
        )

        sources = self.load_sources(
            sample_dir,
            mixture
        )

        if self.split == "train":

            mixture, sources = self.random_crop(
                mixture,
                sources
            )

        else:

            mixture, sources = self.center_crop(
                mixture,
                sources
            )

        sample = {
            "mixture": mixture.float(),
            "sources": sources.float(),
            "num_sources": int(
                metadata.get(
                    "num_speakers",
                    self.max_sources
                )
            )
        }

        if self.return_metadata:

            sample["speaker_ids"] = metadata.get(
                "speaker_ids",
                []
            )

            sample["overlap_ratio"] = metadata.get(
                "overlap_ratio",
                0.0
            )

            sample["duration"] = metadata.get(
                "mix_duration",
                mixture.shape[-1] / self.sample_rate
            )

            sample["sample_name"] = sample_dir.name

        return sample
    