import os
import torch
from tqdm import tqdm

from losses import pit_si_sdr_loss


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        config
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.config = config

        self.best_loss = float("inf")

        self.checkpoint_dir = config.get(
            "checkpoint_dir",
            "checkpoints"
        )

        os.makedirs(
            self.checkpoint_dir,
            exist_ok=True
        )

    def train_epoch(self):

        self.model.train()

        total_loss = 0

        loop = tqdm(
            self.train_loader,
            desc="Training"
        )

        for batch in loop:

            mixture = batch["mixture"].to(self.device)
            sources = batch["sources"].to(self.device)

            self.optimizer.zero_grad()

            estimates = self.model(
                mixture
            )

            loss = pit_si_sdr_loss(
                estimates,
                sources
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                5.0
            )

            self.optimizer.step()

            total_loss += loss.item()

            loop.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        return total_loss / len(self.train_loader)

    def validate(self):

        self.model.eval()

        total_loss = 0

        with torch.no_grad():

            loop = tqdm(
                self.val_loader,
                desc="Validation"
            )

            for batch in loop:

                mixture = batch["mixture"].to(self.device)
                sources = batch["sources"].to(self.device)

                estimates = self.model(
                    mixture
                )

                loss = pit_si_sdr_loss(
                    estimates,
                    sources
                )

                total_loss += loss.item()

                loop.set_postfix(
                    loss=f"{loss.item():.4f}"
                )

        return total_loss / len(self.val_loader)

    def fit(self, epochs):

        for epoch in range(epochs):

            print(f"\n========== Epoch {epoch + 1}/{epochs} ==========")

            # -------------------
            # Training
            # -------------------
            train_loss = self.train_epoch()

            # -------------------
            # Save latest checkpoint
            # -------------------
            last_path = os.path.join(
                self.checkpoint_dir,
                "last_dprnn.pt"
            )

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "train_loss": train_loss,
                },
                last_path
            )

            print(f"Latest checkpoint saved -> {last_path}")

            # -------------------
            # Validation
            # -------------------
            try:

                val_loss = self.validate()

                print(f"Train Loss : {train_loss:.4f}")
                print(f"Val Loss   : {val_loss:.4f}")

                if val_loss < self.best_loss:

                    self.best_loss = val_loss

                    best_path = os.path.join(
                        self.checkpoint_dir,
                        "best_dprnn.pt"
                    )

                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                        },
                        best_path
                    )

                    print(f"Best model saved -> {best_path}")

            except Exception as e:

                print("\nValidation failed!")
                print(e)
                print("Training checkpoint has already been saved.")
                break

        print("\nTraining Finished.")