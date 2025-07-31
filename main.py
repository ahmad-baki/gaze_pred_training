from typing import Union
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.gaze_predictor import GazePredictor
# from data import MyDataset  # Dein Dataset-Import


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Sweep-Definition in ein plain Dict umwandeln
    sweep_cfg = OmegaConf.to_container(
        cfg.sweep,
        resolve=True,
        throw_on_missing=True
    )
    # Sweep anlegen
    sweep_id = wandb.sweep(
        sweep=sweep_cfg,
        project=cfg.wandb.project,
        entity=cfg.wandb.entity
    )
    print(f"Created sweep: {sweep_id}")

    # WandB-Agent startet und ruft für jeden Trial train(cfg) auf
    wandb.agent(sweep_id, function=train, count=1)


def train(cfg: DictConfig):
    # Neue W&B-Run initialisieren und den vollen Hydra-Cfg loggen
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True) # type: ignore
    )
    # Hier holen wir uns nur die hyperparametrischen Werte aus wandb.config
    config = wandb.config

    # DataLoader mit Sweep-Batch-Size
    train_loader: DataLoader = DataLoader(
        MyDataset(split="train"),
        batch_size=config.batch_size,
        shuffle=True
    )
    val_loader: DataLoader = DataLoader(
        MyDataset(split="val"),
        batch_size=config.batch_size
    )

    # Modell: feste Architektur-Teile aus cfg, Dropout aus Sweep
    model: GazePredictor = GazePredictor(
        hidden_dims=cfg.model.hidden_dims,
        dropout=config.dropout,
        repo=cfg.model.feature_extractor_repo,
        dino_model_name=cfg.model.feature_extractor_name
    ).to(device)

    optimizer: torch.optim.Optimizer
    if config.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")

    # Loss-Funktion anhand des gesampelten Sweep-Parameters
    criterion: Union[nn.MSELoss, nn.CrossEntropyLoss]
    if config.loss == "mse":
        criterion = nn.MSELoss()
    elif config.loss == "ce":
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss function: {config.loss}")

    # Trainingsschleife mit gesampelten Epochs
    for epoch in range(config.training.epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                total_val_loss += criterion(model(x), y).item()
        avg_val_loss = total_val_loss / len(val_loader)

        # Metriken loggen
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss
        })

    run.finish()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
