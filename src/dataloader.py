from torch.utils.data import DataLoader

from config import get
from dataset import SpeechSeparationDataset


def create_dataset(cfg, split="train"):

    return SpeechSeparationDataset(
    root_dir=get(cfg, "data_root"),
    split=split,
    sample_rate=get(cfg, "sample_rate"),
    segment_seconds=get(cfg, "segment_seconds"),
    max_sources=get(cfg, "max_sources"),
    return_metadata=False,
)


def create_dataloader(
    cfg,
    split="train",
):

    dataset = create_dataset(
        cfg,
        split=split,
    )

    if split == "train":

        batch_size = get(cfg, "train_batch_size")

        shuffle = get(cfg, "shuffle")

    else:

        batch_size = get(cfg, "val_batch_size")

        shuffle = False

    num_workers = get(cfg, "num_workers")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=(
            get(cfg, "drop_last")
            if split == "train"
            else False
        ),
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=(
            num_workers > 0
        ),
        prefetch_factor=(
            2 if num_workers > 0 else None
        ),
    )

    return loader


def create_train_loader(cfg):

    return create_dataloader(
        cfg,
        split="train",
    )


def create_val_loader(cfg):

    return create_dataloader(
        cfg,
        split="val",
    )


def create_test_loader(cfg):

    return create_dataloader(
        cfg,
        split="test",
    )