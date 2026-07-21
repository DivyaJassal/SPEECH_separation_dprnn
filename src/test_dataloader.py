from config import load_config
from dataloader import create_train_loader


def main():

    cfg = load_config()

    loader = create_train_loader(cfg)

    print("=" * 70)
    print("DATALOADER SUMMARY")
    print("=" * 70)

    print(f"Number of Batches : {len(loader)}")
    print()

    batch = next(iter(loader))

    print("Batch Shapes")
    print("-" * 70)
    print(f"Mixture : {batch['mixture'].shape}")
    print(f"Sources : {batch['sources'].shape}")
    print(f"Num Speakers : {batch['num_sources']}")
    print()

    print("Tensor Statistics")
    print("-" * 70)
    print(f"Mixture Mean : {batch['mixture'].mean():.6f}")
    print(f"Mixture Std  : {batch['mixture'].std():.6f}")
    print(f"Sources Mean : {batch['sources'].mean():.6f}")
    print(f"Sources Std  : {batch['sources'].std():.6f}")
    print()

    B = cfg["train_batch_size"]
    T = cfg["segment_seconds"] * cfg["sample_rate"]

    assert batch["mixture"].shape == (B, 1, T)
    assert batch["sources"].shape == (
        B,
        cfg["max_sources"],
        T,
    )

    print("✓ Mixture batch shape correct")
    print("✓ Source batch shape correct")
    print("✓ DataLoader working correctly")

    print("=" * 70)
    print("DataLoader test passed!")
    print("=" * 70)


if __name__ == "__main__":
    main()