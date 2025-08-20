import contextlib
import csv
import math
import os
from time import time
import hydra
from matplotlib.pyplot import hist
from omegaconf import DictConfig, OmegaConf
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split, DataLoader
from alive_progress import alive_bar

from helper.preprocessed_gaze_dataset_workspace import PreprocessedGazeDatasetWorkspace
from models.gaze_predictor import GazePredictorModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = torch.amp.GradScaler(device=device)
autocast = torch.autocast(device_type="cuda", dtype=torch.float16) if torch.cuda.is_available() else contextlib.nullcontext()

from typing import List, Tuple
import torch
from torch import Tensor
from torch.utils.data import Subset, DataLoader
import random

def _label_index_from_y(y: Tensor) -> int:
    # y may be (C,), (1,G,G), or (C,1,1). We just flatten and argmax.
    return int(y.view(-1).argmax().item())

def stratified_train_val_split(
    dataset,
    train_ratio: float,
    num_classes: int = 256,
    seed: int = 42,
    ensure_val_min1: bool = True
) -> Tuple[Subset, Subset, torch.Tensor]:
    """
    Build a stratified train/val split so each present class appears in train.

    Args:
        dataset: your Dataset yielding (x, y) with y one-hot-ish.
        train_ratio: fraction for train (e.g., 0.8).
        num_classes: total classes.
        seed: RNG seed for reproducibility.
        ensure_val_min1: if True, classes with >=2 samples get at least 1 in val.

    Returns:
        (train_subset, val_subset , class_frequencies)
    """
    # 1) Collect label per sample (O(N)). If this is slow, add a fast label accessor in your dataset.
    all_labels: List[int] = []
    for i in range(len(dataset)):
        _, y = dataset[i]
        all_labels.append(_label_index_from_y(y))

    # 2) Group indices per class
    idx_per_class: List[List[int]] = [[] for _ in range(num_classes)]
    for i, c in enumerate(all_labels):
        idx_per_class[c].append(i)

    # 3) Per-class split
    rng = random.Random(seed)
    train_idx: List[int] = []
    val_idx: List[int] = []
    for c, idxs in enumerate(idx_per_class):
        if not idxs:
            continue  # class absent altogether
        rng.shuffle(idxs)
        n = len(idxs)
        # at least 1 in train if class exists
        k = max(1, int(round(train_ratio * n)))
        if ensure_val_min1 and n >= 2:
            # ensure at least 1 goes to val too
            k = min(k, n - 1)
        train_idx.extend(idxs[:k])
        val_idx.extend(idxs[k:])

    # 4) Build subsets
    class_frequencies = torch.zeros(num_classes, dtype=torch.long)
    for c, idxs in enumerate(idx_per_class):
        class_frequencies[c] = len(idxs)
    return Subset(dataset, train_idx), Subset(dataset, val_idx), class_frequencies

# Usage inside get_data_loaders:
# train_dataset, val_dataset, class_frequencies = stratified_train_val_split(dataset, cfg.dataset.train_split, num_classes=256)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, ...)
# val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, ...)


def accuracy_topk(outputs, targets, k=3):
    """
    outputs: Tensor of shape (B, C) before softmax
    targets: Tensor of shape (B,) containing class indices [0..C-1]
    returns: percentage of samples whose target is in top‐k preds
    """
    # get the top k predicted class indices
    topk_vals, topk_inds = outputs.topk(k, dim=1, largest=True, sorted=True)
    # targets.view(-1,1) is (B,1); compare to each of the k preds
    correct = topk_inds.eq(targets.view(-1, 1)).any(dim=1).float()
    return correct.mean().item()

def _label_index_from_y(y: Tensor) -> int:
    # y may be (C,), (1,G,G), or (C,1,1). We just flatten and argmax.
    return int(y.view(-1).argmax().item())

def labels_for_subset(subset: Subset) -> torch.Tensor:
    """Return a length-|subset| tensor of class indices for the given subset."""
    labels = torch.empty(len(subset), dtype=torch.long)
    for i in range(len(subset)):
        _, y = subset[i]
        labels[i] = _label_index_from_y(y)
    return labels

import torch

def sum_euclidean_distance_index(P: torch.Tensor, Q: torch.Tensor, grid_size: int = 16, center: bool = True) -> torch.Tensor:
    """
    Sum of Euclidean distances between positions from two indexed batches.

    P, Q: tensors shaped (B), where each value is the index of the cell in the grid.
    grid_size: side length (default 16).
    center: if True, measure from cell centers (+0.5) instead of corners.

    Returns: scalar tensor (sum over batch).
    """
    # Convert flat index -> (x, y)
    yP = torch.div(P, grid_size, rounding_mode='floor')
    xP = P - yP * grid_size
    yQ = torch.div(Q, grid_size, rounding_mode='floor')
    xQ = Q - yQ * grid_size

    if center:
        xP = xP.float() + 0.5
        yP = yP.float() + 0.5
        xQ = xQ.float() + 0.5
        yQ = yQ.float() + 0.5
    else:
        xP = xP.float()
        yP = yP.float()
        xQ = xQ.float()
        yQ = yQ.float()

    d = torch.hypot(xP - xQ, yP - yQ)  # (B,)
    return d.sum()


def get_data_loaders(config, batch_size):
    # Create dataset; note that 'task' can be a string or a list of tasks
    print(config.dataset.tasks)
    dataset = PreprocessedGazeDatasetWorkspace(
        dir=config.dataset.dir,
        tasks=config.dataset.tasks,
    )

    # split sizes
    # total_size = len(dataset)
    # train_size = int(config.dataset.train_split * total_size)
    # val_size   = total_size - train_size

    # First way: random
    # train_dataset, val_dataset = random_split(
    #     dataset,
    #     [train_size, val_size],
    #     generator=torch.Generator().manual_seed(42)
    # )

    # Second way: stratified so that every class is guaranteed to be represented
    train_dataset, val_dataset, _ = stratified_train_val_split(dataset, config.dataset.train_split, num_classes=256)

    # ── Build per-sample weights for the train subset ─────────────────────
    train_labels = labels_for_subset(train_dataset)           # (N_train,)
    counts = torch.bincount(train_labels, minlength=256).clamp_min(1).float()


    if config.sampler == "random_sampler":
        sampler = torch.utils.data.RandomSampler(train_dataset)
    else:  # config.sampler == "weighted_random_sampler"
        # α controls how strong the balancing is: 0.0 = none, 1.0 = full inverse-freq
        alpha = config.alpha
        class_weights = counts.pow(-alpha)                        # (256,)
        generator = torch.Generator()
        generator.manual_seed(42)
        sample_weights = class_weights[train_labels]              # (N_train,)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),   # one "epoch" ~ same size as dataset
            replacement=True,
            generator=generator
        )

    # Random Weighted Sampler (reproducible if you pass a generator with a seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    return train_loader, val_loader


@hydra.main(config_path="conf", config_name="best_config")
def train(cfg: DictConfig):
    print(cfg)
    # region setup
    # ──────── SETUP ─────────────────────────
    setup_start = time()
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    save_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    if not os.path.exists(os.path.join(save_dir, "checkpoints")):
        os.makedirs(os.path.join(save_dir, "checkpoints"))

    print("Creating dataloader...")
    train_loader, val_loader = get_data_loaders(cfg, batch_size=cfg.batch_size)
    print("Created Dataloader")
    # ─── Compute per‐cell class weights ─────────────────────────────────────
    # assumes each y is one‐hot of shape [B, 1, G, G] or [B, G*G]
    print("Computing class weights...")
    all_labels = []
    for x, y in train_loader:
        idx = y.view(y.size(0), -1).argmax(dim=1)
        all_labels.append(idx)
    all_labels = torch.cat(all_labels)
    counts = torch.bincount(all_labels, minlength=16*16).float()
    freqs = counts / counts.sum()
    class_weights = (1.0 / (freqs + 1e-6))
    class_weights /= class_weights.mean()
    class_weights = class_weights.to(device)
    print("Computed class weights")
    # model = hydra.utils.instantiate(cfg.model, dropout=cfg.dropout).to(device)
    # optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    # criterion = hydra.utils.instantiate(cfg.loss)
    print("Instantiating objects")
    model = hydra.utils.instantiate(cfg.model, dropout=cfg.dropout).to(device)

    # ─── Optimizer factory ───────────────────────────────────────────────
    optim_type = cfg.optimizer.type.lower()
    if optim_type in ("adam", "adamw"):
        Optim = torch.optim.AdamW if optim_type == "adamw" else torch.optim.Adam
        optimizer = Optim(
            model.parameters(),
            lr=cfg.lr,
            betas=(cfg.optimizer.betas0, cfg.optimizer.betas1),
            weight_decay=cfg.weight_decay,
        )
    elif optim_type == "sgd":
        # ensure you add `momentum` to your config if you want to sweep it
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=cfg.optimizer.momentum,
            weight_decay=cfg.weight_decay,
        )
    elif optim_type == "ranger":
        from ranger import Ranger  
        optimizer = Ranger(
            model.parameters(),
            lr=cfg.lr,
            betas=(cfg.optimizer.betas0, cfg.optimizer.betas1),
            weight_decay=cfg.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer type: {cfg.optimizer.type}")

    # ─── Loss with label smoothing & weights ─────────────────────────────
    # criterion = torch.nn.CrossEntropyLoss(
    #     weight=class_weights,
    #     label_smoothing=cfg.label_smoothing,
    # )
    criterion = torch.nn.CrossEntropyLoss()
    # ─── Scheduler factory ────────────────────────────────────────────────
    sched_cfg = cfg.scheduler
    if sched_cfg.type == "steplr":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_cfg.step_size,
            gamma=sched_cfg.gamma
        )
    elif sched_cfg.type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.epochs
        )
    elif sched_cfg.type == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=sched_cfg.max_lr,
            total_steps=cfg.epochs * len(train_loader)
        )
    else:  # reduceonplateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=sched_cfg.patience,
            factor=sched_cfg.gamma
        )
    print("Finished setting up training.")
    setup_time = time() - setup_start

    # ──────── TRAINING ───────────────────────
    # csv_path = os.path.join(save_dir, 'timings.csv')
    # file_exists = os.path.isfile(csv_path)
    # csv_file = open(csv_path, 'a', newline='')
    # csv_writer = csv.writer(csv_file)
    # if not file_exists:
    #     csv_writer.writerow(['epoch', 'epoch_time', 'data_loading_time',
    #                          'model_running_time', 'update_step_time'])

    # ── TRAINING LOOP ────────────────────
    sum_data_loading = sum_model_running = sum_update = 0.0
    # training_start = time()

    best_val_dist = math.inf
    best_model_save_path = ""

    # endregion
    
    print("Starting training...")
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
            train_topk_acc_sum = 0.0

            # -- Tracking Training Loss --
            sum_train_loss = 0.0
            train_batches = 0
            train_distance = 0.0  
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
                with autocast:
                    outputs = model(x)
                    target_idx = y.view(y.size(0), -1).argmax(dim=1)
                    loss = criterion(outputs, target_idx)
                mr_end = time()
               

                # Backward + update timer
                us_start = time()
                scaler.scale(loss).backward()
                # ─── gradient clipping ────────────────────────────
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    cfg.gradient_clip_max_norm
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if sched_cfg.type == "onecycle":
                    scheduler.step()

                us_end = time()
                epoch_us += (us_end - us_start)
                sum_update += (us_end - us_start)

                epoch_mr += (mr_end - mr_start)
                sum_model_running += (mr_end - mr_start)

                # Track accuracy
                pred_idx = outputs.argmax(dim=1)
                # Top-1
                train_correct += (pred_idx == target_idx).sum().item()
                # Top-k
                batch_topk_acc = accuracy_topk(outputs, target_idx, cfg.top_k)
                train_topk_acc_sum += batch_topk_acc * y.size(0)
                train_distance += sum_euclidean_distance_index(pred_idx, target_idx)  # <-- Compute train batch distance
                train_total += y.size(0)
                sum_train_loss += loss.item()

            # -- Track Validation Accuracy --
            
            val_correct = 0
            val_total = 0
            val_topk_acc_sum = 0.0
            val_distance = 0.0

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
                    with autocast:
                        outputs = model(x)
                        target_idx = y.view(y.size(0), -1).argmax(dim=1)
                        loss = criterion(outputs, target_idx)
                    mr_end = time()
                    epoch_mr += (mr_end - mr_start)
                    sum_model_running += (mr_end - mr_start)

                    # Track accuracy
                    pred_idx = outputs.argmax(dim=1)
                    # Top-1
                    val_correct += (pred_idx == target_idx).sum().item()
                    # Top-k
                    batch_topk_acc = accuracy_topk(outputs, target_idx, cfg.top_k)
                    # calculate the eucledian distance if interpreted as position in 16x16 grid
                    val_distance += sum_euclidean_distance_index(pred_idx, target_idx)
                    val_topk_acc_sum += batch_topk_acc * y.size(0)
                    val_total += y.size(0)

                    sum_validation_loss += loss.item()


            # ─── acc and loss calc ─────────────────────────────────
            avg_train_acc = train_correct / train_total if train_total > 0 else 0
            avg_val_acc = val_correct / val_total if val_total > 0 else 0
            avg_train_topk_acc = train_topk_acc_sum / train_total if train_total > 0 else 0
            avg_val_topk_acc = val_topk_acc_sum / val_total if val_total > 0 else 0
            avg_train_distance = train_distance / train_total if train_total > 0 else 0
            avg_val_distance = val_distance / val_total if val_total > 0 else 0

            avg_train_loss = sum_train_loss / train_total if train_total > 0 else 0
            avg_val_loss = sum_validation_loss / val_total if val_total > 0 else 0

            # ─── step scheduler ─────────────────────────────────
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            elif sched_cfg.type != "onecycle":
                scheduler.step()

            epoch_time = time() - epoch_start

            # -- Log epoch results --1

            # save checkpoint
            if epoch % cfg.model_save_interval == 0:
                model_save_path = os.path.join(save_dir, "checkpoints", f'model_epoch_{epoch+1}.pth')
                model.save(model_save_path)

            # save best model
            if best_val_dist > avg_val_distance:
                model_save_path = os.path.join(save_dir, f'model_epoch_{epoch+1}.pth')
                model.save(model_save_path)
                best_val_dist = avg_val_distance
                os.remove(best_model_save_path) if best_model_save_path else ""
                best_model_save_path = model_save_path
                print(f"Saved best model to {model_save_path}")

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
                "avg_train_loss": avg_train_loss,
                "avg_val_loss": avg_val_loss,
                "avg_train_acc": avg_train_acc,
                "avg_val_acc": avg_val_acc,
                f"avg_train_top{cfg.top_k}_acc": avg_train_topk_acc,
                f"avg_val_top{cfg.top_k}_acc": avg_val_topk_acc,
                "avg_train_distance": avg_train_distance,
                "avg_val_distance": avg_val_distance,
                "epoch": epoch,
                "epoch_time": epoch_time,
                "data_loading_time": epoch_dl,
                "model_running_time": epoch_mr,
                "update_step_time": epoch_us,
            })
            print(f"Epoch {epoch+1} time: {epoch_time:.2f}s "
                  f"(dl: {epoch_dl:.2f}s, mr: {epoch_mr:.2f}s, us: {epoch_us:.2f}s)")
            print(f"Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}")
            print(f"Avg Train Acc: {avg_train_acc:.4f}, Avg Val Acc: {avg_val_acc:.4f}")
            print(f"Avg Train Top-{cfg.top_k} Acc: {avg_train_topk_acc:.4f}, Avg Val Top-{cfg.top_k} Acc: {avg_val_topk_acc:.4f}")
            print(f"Avg Train Distance: {avg_train_distance:.4f}, Avg Val Distance: {avg_val_distance:.4f}")

    torch.save(model.state_dict(), save_dir + '/model.pth')
    print("Training complete.")
    run.finish()

if __name__ == "__main__":
    train()  # pylint: disable=no-value-for-parameter