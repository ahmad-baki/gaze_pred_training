import csv
import os
from time import time
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split, DataLoader
from alive_progress import alive_bar

from helper.dataset import PreprocessedGazeDatasetWorkspace
from models.gaze_predictor import GazePredictor


# from data import MyDataset  # Dein Dataset-Import


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = torch.amp.GradScaler(device=device)

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
def train(cfg: DictConfig):
    # ──────── SETUP ─────────────────────────
    setup_start = time()
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    csv_dir_path = f"/home/ka/ka_anthropomatik/ka_eb5961/gaze_pred_training/time_logs/{run.id}"
    os.makedirs(csv_dir_path, exist_ok=True)
    print(f"Created directory for time logs: {csv_dir_path}")


    train_loader, val_loader = get_data_loaders(cfg, batch_size=cfg.batch_size)

    model = GazePredictor(
        feature_dims=cfg.model.feature_extractor.feature_dims,
        hidden_dims=cfg.model.hidden_dims,
        dropout=cfg.dropout,
        repo=cfg.model.feature_extractor.repo,
        dino_model_name=cfg.model.feature_extractor.name
    ).to(device)

    optimizer = {
        "adam": torch.optim.Adam,
        "sgd":  torch.optim.SGD
    }[cfg.optimizer](model.parameters(), lr=cfg.learning_rate)

    criterion = {
        "mse": nn.MSELoss(),
        "ce":  nn.CrossEntropyLoss()
    }[cfg.loss_function]

    setup_time = time() - setup_start

    # ──────── TRAINING ───────────────────────
    csv_path = os.path.join(csv_dir_path, 'timings.csv')
    file_exists = os.path.isfile(csv_path)
    csv_file = open(csv_path, 'a', newline='')
    csv_writer = csv.writer(csv_file)
    if not file_exists:
        csv_writer.writerow(['epoch', 'epoch_time', 'data_loading_time',
                             'model_running_time', 'update_step_time'])

    # ── TRAINING LOOP ────────────────────
    sum_data_loading = sum_model_running = sum_update = 0.0
    training_start = time()

    with alive_bar(cfg.epochs, title="Training") as bar:
        for epoch in range(cfg.epochs):
            bar.text = f"Epoch {epoch+1}/{cfg.epochs}"
            bar()

            # Per-epoch timers
            epoch_start = time()
            epoch_dl = epoch_mr = epoch_us = 0.0

            model.train()
            for x, y in train_loader:
                # Data loading timer
                dl_start = time()
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                dl_end = time()
                epoch_dl += (dl_end - dl_start)
                sum_data_loading += (dl_end - dl_start)

                # Model forward + loss timer
                mr_start = time()
                outputs = model(x)
                loss = criterion(outputs, y)
                mr_end = time()
                epoch_mr += (mr_end - mr_start)
                sum_model_running += (mr_end - mr_start)

                # Backward + update timer
                us_start = time()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                us_end = time()
                epoch_us += (us_end - us_start)
                sum_update += (us_end - us_start)

            model.eval()
            with torch.no_grad():
                for x, y in val_loader:
                    # Data loading timer
                    dl_start = time()
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    dl_end = time()
                    epoch_dl += (dl_end - dl_start)
                    sum_data_loading += (dl_end - dl_start)
                    # Model forward + loss timer
                    mr_start = time()
                    outputs = model(x)
                    loss = criterion(outputs, y)
                    mr_end = time()
                    epoch_mr += (mr_end - mr_start)
                    sum_model_running += (mr_end - mr_start)


            epoch_time = time() - epoch_start

            # Log to CSV
            csv_writer.writerow([
                epoch + 1,
                f"{epoch_time:.4f}",
                f"{epoch_dl:.4f}",
                f"{epoch_mr:.4f}",
                f"{epoch_us:.4f}",
            ])

            # Log to W&B
            wandb.log({
                "epoch": epoch,
                "epoch_time": epoch_time,
                "data_loading_time": epoch_dl,
                "model_running_time": epoch_mr,
                "update_step_time": epoch_us,
            })
            print(f"Epoch {epoch+1} time: {epoch_time:.2f}s "
                  f"(dl: {epoch_dl:.2f}s, mr: {epoch_mr:.2f}s, us: {epoch_us:.2f}s)")

    print("Training complete.")
    # ──────── FINALIZE ──────────────────────
    training_time = time() - training_start
    csv_file.close()

    # ──────── SUMMARY ────────────────────────
    total_time = setup_time + training_time
    totals = {
        "setup_time":         f"{100 * setup_time / total_time:.4f}%",
        "training_time":      f"{100 * training_time / total_time:.4f}%",
        "data_loading_time":  f"{100 * sum_data_loading / total_time:.4f}%",
        "model_running_time": f"{100 * sum_model_running / total_time:.4f}%",
        "update_step_time":   f"{100 * sum_update / total_time:.4f}%",
        "total_time":         f"{total_time:.4f}s"
    }
    totals_csv = os.path.join(csv_dir_path, 'totals.csv')
    file_exists = os.path.isfile(totals_csv)

    # Open in append mode, write header if new, then write totals
    with open(totals_csv, 'a', newline='') as f_tot:
        print(f"Writing totals to {totals_csv}")
        writer = csv.writer(f_tot)
        if not file_exists:
            writer.writerow([
                'setup_time',
                'training_time',
                'data_loading_time',
                'model_running_time',
                'update_step_time',
                'total_time'
            ])
        writer.writerow([
            totals['setup_time'],
            totals['training_time'],
            totals['data_loading_time'],
            totals['model_running_time'],
            totals['update_step_time'],
            totals['total_time'],
        ])

    run.finish()


def get_data_loaders(config, batch_size):
    # dataset = PreprocessedGazeDatasetWorkspace(
    #     image_dir=config.dataset.image_dir,
    #     label_dir=config.dataset.label_dir
    #     # transform=config.dataset.transform
    # )

    dataset = PreprocessedGazeDatasetWorkspace(
        image_dir=config.dataset.dir,
        task=config.dataset.task,
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
        num_workers=8,             # bump up
        pin_memory=True,
        persistent_workers=True,   # workers stick around
        prefetch_factor=4          # how many batches each worker preloads
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,             # bump up
        pin_memory=True,
        persistent_workers=True,   # workers stick around
        prefetch_factor=4          # how many batches each worker preloads
    )
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    return train_loader,val_loader


if __name__ == "__main__":
    train()  # pylint: disable=no-value-for-parameter
