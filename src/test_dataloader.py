from config import load_config
from dataloader import create_dataloader

config = load_config()

loader = create_dataloader(
    root_dir="../conversational_dataset_v2",
    split="train",
    sample_rate=config["sample_rate"],
    segment_seconds=config["segment_seconds"],
    max_sources=config["max_sources"],
    batch_size=4,
)

batch = next(iter(loader))

print("Mixture Shape:")
print(batch["mixture"].shape)

print()

print("Sources Shape:")
print(batch["sources"].shape)

print()

print("Number of Speakers:")
print(batch["num_sources"])