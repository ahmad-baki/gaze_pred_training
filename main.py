
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Simple CNN for illustration
class SimpleCNN(nn.Module):
    def __init__(self, channels, dropout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 14 * 14, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.fc(self.conv(x))
# import your dataset and model definitions here
# from my_dataset import MyDataset
# from my_model import MyModel

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """
    Launches a WandB sweep with Bayesian hyperparameter optimization,
    then runs training trials via wandb.agent.
    """
    # Convert Hydra sweep config to a plain dict
    sweep_cfg = OmegaConf.to_container(cfg.sweep, resolve=True)

    # Create the sweep in WandB
    sweep_id = wandb.sweep(
        sweep=sweep_cfg,
        project=cfg.wandb.project,
        entity=cfg.wandb.entity
    )
    print(f"Created sweep: {sweep_id}")

    # Start the agent; each agent run calls `train(cfg)`
    wandb.agent(sweep_id, function=lambda: train(cfg))


def train(cfg: DictConfig):
    """
    Single training run for a given set of hyperparameters.
    """
    # Initialize a new W&B run; pass full Hydra cfg so defaults are tracked
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    config = run.config  # this includes hyperparameters from the sweep

    # TODO: Prepare data loaders (replace placeholders with your code)
    train_loader = DataLoader(
        # MyDataset(split="train"),
        batch_size=config.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        # MyDataset(split="val"),
        batch_size=config.batch_size
    )

    # Build model, optimizer, and loss
    model = SimpleCNN(dropout=config.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(config.training.epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        # Validation step
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                outputs = model(x)
                total_val_loss += criterion(outputs, y).item()
        avg_val_loss = total_val_loss / len(val_loader)

        # Log metrics to WandB
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss
        })

    run.finish()


if __name__ == "__main__":
    main()
