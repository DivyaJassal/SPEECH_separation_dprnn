from torch.utils.data import DataLoader

from dataset import SpeechSeparationDataset


def create_dataloader(
    root_dir,
    split,
    sample_rate,
    segment_seconds,
    max_sources,
    batch_size,
    shuffle=True,
    num_workers=0,
):

    dataset = SpeechSeparationDataset(
        root_dir=root_dir,
        split=split,
        sample_rate=sample_rate,
        segment_seconds=segment_seconds,
        max_sources=max_sources,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader