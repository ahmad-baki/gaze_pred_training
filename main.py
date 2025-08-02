from typing import Union
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split, DataLoader
from alive_progress import alive_bar

from helper.dataset import PreprocessedGazeDataset
from models.gaze_predictor import GazePredictor

# from data import MyDataset  # Dein Dataset-Import


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# @hydra.main(config_path="conf", config_name="config")
# def run_main(cfg: DictConfig):
#     print("Running main with config:", cfg)
#     # Sweep-Definition in ein plain Dict umwandeln
#     sweep_cfg = OmegaConf.to_container(
#         cfg.sweep,
#         resolve=True,
#         throw_on_missing=True
#     )
#     # Sweep anlegen
#     sweep_id = wandb.sweep(
#         sweep=sweep_cfg,
#         project=cfg.wandb.project,
#         entity=cfg.wandb.entity
#     )
#     print(f"Created sweep: {sweep_id}")

#     # WandB-Agent startet und ruft für jeden Trial train(cfg) auf
#     wandb.agent(sweep_id, function=lambda: train(cfg), count=1)


@hydra.main(config_path="conf", config_name="config")
def train(cfg):
    # Neue W&B-Run initialisieren und den vollen Hydra-Cfg loggen
    print("Running training")
    run = wandb.init(
        # config=cfg,
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True) # type: ignore
    )
    train_loader, val_loader = get_data_loaders(cfg, batch_size=cfg.batch_size)
    print("Data loaders created.")
    # Modell: feste Architektur-Teile aus cfg, Dropout aus Sweep
    model: GazePredictor = GazePredictor(
        hidden_dims=cfg.model.hidden_dims,
        dropout=cfg.dropout,
        repo=cfg.model.feature_extractor.repo,
        dino_model_name=cfg.model.feature_extractor.name
    ).to(device)
    print("Model created.")

    optimizer: torch.optim.Optimizer
    if cfg.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    elif cfg.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")

    # Loss-Funktion anhand des gesampelten Sweep-Parameters
    criterion: Union[nn.MSELoss, nn.CrossEntropyLoss]
    if cfg.loss_function == "mse":
        criterion = nn.MSELoss()
    elif cfg.loss_function == "ce":
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss function: {cfg.loss}")
    print("Optimizer and loss function set.")
    # Trainingsschleife mit gesampelten Epochs
    print(f"Starting training for {cfg.epochs} epochs...")
    with alive_bar(cfg.epochs, title="Training progress") as bar:
         # Training und Validierung
        for epoch in range(cfg.epochs):
            bar.text = f"Epoch {epoch + 1}/{cfg.epochs}"
            bar()

            # WandB-Log für die aktuelle Epoche
            model.train()
            total_loss = 0.0
            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                outputs = model(x)
                # print shapes
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_train_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{cfg.epochs}, Train Loss: {avg_train_loss:.4f}")
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    # print shapes
                    x = x.to(device)
                    y = y.to(device)
                    total_val_loss += criterion(model(x), y).item()
            avg_val_loss = total_val_loss / len(val_loader)
            print(f"Epoch {epoch + 1}/{cfg.epochs}, Val Loss: {avg_val_loss:.4f}")

            # Metriken loggen
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss
            })
    print(f"Finished training for {cfg.epochs} epochs.")
    run.finish()

def get_data_loaders(config, batch_size):
    dataset = PreprocessedGazeDataset(
        image_dir=config.dataset.image_dir,
        label_dir=config.dataset.label_dir
        # transform=config.dataset.transform
    )

    # split sizes
    total_size = len(dataset)
    train_size = int(config.dataset.train_split * total_size)
    val_size   = total_size - train_size

    # do the split (with a fixed seed for reproducibility)
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # DataLoader für Training und Validierung
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    return train_loader,val_loader


if __name__ == "__main__":
    train()  # pylint: disable=no-value-for-parameter
