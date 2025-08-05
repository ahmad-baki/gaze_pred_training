import csv
import math
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

from helper.preprocessed_gaze_dataset_workspace import PreprocessedGazeDatasetWorkspace
from models.gaze_predictor import GazePredictorModel


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
    save_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    os.makedirs(save_dir, exist_ok=True)
    print(f"Created directory for time logs: {save_dir}")


    train_loader, val_loader = get_data_loaders(cfg, batch_size=cfg.batch_size)

    model = hydra.utils.instantiate(cfg.model, dropout=cfg.dropout).to(device)
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    criterion = hydra.utils.instantiate(cfg.loss)

    setup_time = time() - setup_start

    # ──────── TRAINING ───────────────────────
    csv_path = os.path.join(save_dir, 'timings.csv')
    file_exists = os.path.isfile(csv_path)
    csv_file = open(csv_path, 'a', newline='')
    csv_writer = csv.writer(csv_file)
    if not file_exists:
        csv_writer.writerow(['epoch', 'epoch_time', 'data_loading_time',
                             'model_running_time', 'update_step_time'])

    # ── TRAINING LOOP ────────────────────
    sum_data_loading = sum_model_running = sum_update = 0.0
    training_start = time()

    best_val_loss = math.inf
    best_model_save_path = ""
    with alive_bar(cfg.epochs, title="Training") as bar:
        for epoch in range(cfg.epochs):
            bar.text = f"Epoch {epoch+1}/{cfg.epochs}"
            bar()

            # Per-epoch timers
            epoch_start = time()
            epoch_dl = epoch_mr = epoch_us = 0.0

            # -- Track Training Accuracy --
            train_correct = 0
            train_total = 0

            # -- Tracking Training Loss --
            sum_train_loss = 0.0
            train_batches = 0
            model.train()
            for x, y in train_loader:
                train_batches += 1

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
                target_idx = y.view(y.size(0), -1).argmax(dim=1)
                loss = criterion(outputs, target_idx)
                mr_end = time()
                epoch_mr += (mr_end - mr_start)
                sum_model_running += (mr_end - mr_start)

                # Track accuracy
                pred_idx = outputs.argmax(dim=1)
                train_correct += (pred_idx == target_idx).sum().item()
                train_total += y.size(0)
                sum_train_loss += loss.item()

                # Backward + update timer
                us_start = time()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                us_end = time()
                epoch_us += (us_end - us_start)
                sum_update += (us_end - us_start)

            # -- Track Validation Accuracy --
            val_correct = 0
            val_total = 0

            # -- Track Validation Loss --
            sum_validation_loss = 0.0
            val_batches = 0
            model.eval()
            with torch.no_grad():
                for x, y in val_loader:
                    # Data loading timer
                    dl_start = time()
                    val_batches += 1
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

                    # Track accuracy
                    pred_idx = outputs.view(outputs.size(0), -1).argmax(dim=1)
                    target_idx = y.view(y.size(0), -1).argmax(dim=1)
                    val_correct += (pred_idx == target_idx).sum().item()
                    val_total += y.size(0)

                    sum_validation_loss += loss.item()

            epoch_time = time() - epoch_start

            # -- Log epoch results --
            avg_train_acc = train_correct / train_total if train_total > 0 else 0
            avg_val_acc = val_correct / val_total if val_total > 0 else 0

            avg_train_loss = sum_train_loss / train_batches
            avg_val_loss = sum_validation_loss / val_batches

            if epoch % cfg.model_save_interval == 0 and best_val_loss > avg_val_loss:
                model_save_path = os.path.join(save_dir, f'model_epoch_{epoch+1}.pth')
                model.save(model_save_path)
                best_val_loss = avg_val_loss
                os.remove(best_model_save_path) if best_model_save_path else ""
                best_model_save_path = model_save_path
                print(f"Saved model checkpoint to {model_save_path}")

            # Log to CSV
            # csv_writer.writerow([
            #     epoch + 1,
            #     f"{epoch_time:.4f}",
            #     f"{epoch_dl:.4f}",
            #     f"{epoch_mr:.4f}",
            #     f"{epoch_us:.4f}",
            # ])

            # Log to W&B
            wandb.log({
                "epoch": epoch,
                "epoch_time": epoch_time,
                "data_loading_time": epoch_dl,
                "model_running_time": epoch_mr,
                "update_step_time": epoch_us,
                "avg_train_loss": avg_train_loss,
                "avg_val_loss": avg_val_loss,
                "avg_train_acc": avg_train_acc,
                "avg_val_acc": avg_val_acc,
            })
            print(f"Epoch {epoch+1} time: {epoch_time:.2f}s "
                  f"(dl: {epoch_dl:.2f}s, mr: {epoch_mr:.2f}s, us: {epoch_us:.2f}s)")
            print(f"Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}")
            print(f"Avg Train Acc: {avg_train_acc:.4f}, Avg Val Acc: {avg_val_acc:.4f}")

    torch.save(model.state_dict(), save_dir + '/model.pth')
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
    totals_csv = os.path.join(save_dir, 'totals.csv')
    file_exists = os.path.isfile(totals_csv)

    # Open in append mode, write header if new, then write totals
    # with open(totals_csv, 'a', newline='') as f_tot:
    #     print(f"Writing totals to {totals_csv}")
    #     writer = csv.writer(f_tot)
    #     if not file_exists:
    #         writer.writerow([
    #             'setup_time',
    #             'training_time',
    #             'data_loading_time',
    #             'model_running_time',
    #             'update_step_time',
    #             'total_time'
    #         ])
    #     writer.writerow([
    #         totals['setup_time'],
    #         totals['training_time'],
    #         totals['data_loading_time'],
    #         totals['model_running_time'],
    #         totals['update_step_time'],
    #         totals['total_time'],
    #     ])

    run.finish()


def get_data_loaders(config, batch_size):
    # dataset = PreprocessedGazeDatasetWorkspace(
    #     image_dir=config.dataset.image_dir,
    #     label_dir=config.dataset.label_dir
    #     # transform=config.dataset.transform
    # )

    dataset = PreprocessedGazeDatasetWorkspace(
        dir=config.dataset.dir,
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
