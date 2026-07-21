from config import load_config
from dataset import SpeechSeparationDataset


def main():

    cfg = load_config()

    dataset = SpeechSeparationDataset(
        root_dir=cfg["data_root"],
        split="train",
        sample_rate=cfg["sample_rate"],
        segment_seconds=cfg["segment_seconds"],
        max_sources=cfg["max_sources"],
        return_metadata=True,
    )

    print("=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    print(f"Dataset Size : {len(dataset)}")
    print()

    sample = dataset[0]

    print("First Sample")
    print("-" * 70)
    print(f"Mixture Shape      : {sample['mixture'].shape}")
    print(f"Sources Shape      : {sample['sources'].shape}")
    print(f"Number of Speakers : {sample['num_sources']}")
    print(f"Speaker IDs        : {sample['speaker_ids']}")
    print(f"Overlap Ratio      : {sample['overlap_ratio']:.4f}")
    print(f"Duration           : {sample['duration']:.2f} sec")

    print()

    print("Tensor Statistics")
    print("-" * 70)
    print(f"Mixture Mean : {sample['mixture'].mean():.6f}")
    print(f"Mixture Std  : {sample['mixture'].std():.6f}")
    print(f"Sources Mean : {sample['sources'].mean():.6f}")
    print(f"Sources Std  : {sample['sources'].std():.6f}")

    print()

    print("Sanity Checks")
    print("-" * 70)

    assert sample["mixture"].shape[0] == 1
    assert sample["sources"].shape[0] == cfg["max_sources"]
    assert sample["mixture"].shape[-1] == sample["sources"].shape[-1]
    assert 2 <= sample["num_sources"] <= cfg["max_sources"]

    print("✓ Mixture shape correct")
    print("✓ Source padding correct")
    print("✓ Waveform lengths match")
    print("✓ Variable speaker count correct")

    print()
    print("=" * 70)
    print("Dataset test passed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()