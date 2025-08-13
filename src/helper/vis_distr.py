import os
from pathlib import Path
import sys
from typing import Tuple

import argparse
import cv2
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
from preprocessed_gaze_dataset_workspace import PreprocessedGazeDatasetWorkspace
from hydra import compose, initialize
from torch.utils.data import DataLoader


def overlay_grid_and_highlight_pred(
    img: np.ndarray,
    values: torch.Tensor,
    nx: int = 16,
    ny: int = 16,
    grid_color: Tuple[int,int,int] = (0, 255, 0),
    grid_thickness: int = 1,
    overlay_alpha: float = 0.5,
    overlay_text: bool = True
) -> 'np.ndarray':
    """
    Overlays an nx×ny grid on the image and colors each cell based on values array.
    Cells with values above median are colored green, cells below median are colored red.

    Parameters
    ----------
    img : numpy.ndarray
        An already-loaded BGR numpy array.
    values : numpy.ndarray, shape (nx*ny,)
        Array with values between 0 and 1 for each grid cell.
    nx, ny : int
        Number of columns and rows in the grid.
    grid_color : (B, G, R)
        Color of the grid lines.
    grid_thickness : int
        Thickness of the grid lines.
    overlay_alpha : float in [0,1]
        Opacity of the cell overlays.

    Returns
    -------
    numpy.ndarray
        The annotated BGR image with heatmap overlay.
    """
    # 1. Load image if needed
    if isinstance(img, str):
        print(f"Loading image from path: {img}")
        loaded_img = cv2.imread(img)
        if loaded_img is None:
            raise FileNotFoundError(f"Could not load image at '{img}'")
        img = loaded_img
    
    out = img.copy()
    h, w = out.shape[:2]
    print(f"Image shape: {out.shape}, grid: {nx}x{ny}")

    # 2. Validate input
    if len(values) != nx * ny:
        print(f"Values array shape: {values.shape}")
        raise ValueError(f"Values array must have length {nx * ny}, got {len(values)}")
    
    # 3. Compute per-cell size
    dx = w // nx
    dy = h // ny

    # 4. Calculate median for color threshold and find max value index
    median_value = float(np.median(values))
    print(f"Median value for coloring: {median_value}")
    max_idx = int(np.argmax(values))
    max_value = float(values[max_idx])
    min_idx = int(np.argmin(values))
    min_value = float(values[min_idx])
    rng  = max(max_value - min_value, sys.float_info.epsilon)
    sum_val = float(values.detach().sum().item())
    print(f"Min value index: {min_idx}, value: {min_value}")
    print(f"Max value index: {max_idx}, value: {max_value}")
    print(f"Sum of values: {sum_val}")

    # 5. Create overlay for all cells
    overlay = out.copy()
    
    # 6. Color each cell based on its value
    for i in range(nx * ny):
        # Calculate grid position
        row = i // nx
        col = i % nx
        x0, y0 = col * dx, row * dy
        x1, y1 = x0 + dx, y0 + dy
        
        # Determine color based on value relative to median
        value = float(values[i].item())
        # region OLD_MEDIAN_COLORING
        # if value >= median_value:
        #     # Green for values >= median
        #     color_perc = 1 - (max_value - value)
        #     cell_color = (0, color_perc * 255, 0)  # BGR format, int
        # else:
        #     # Red for values < median
        #     color_perc = 1 - (value - min_value)
        #     cell_color = (0, 0, color_perc * 255)  # BGR format, int
        # endregion

        value_norm = (value - min_value) / rng          # 0..1
        g = int(255 * value_norm)
        r = int(255 * (1 - value_norm))
        cell_color = (0, g, r) # BGR format, int

        # Fill the cell with the determined color
        # print shape
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color=cell_color, thickness=-1)

        if overlay_text:
            text = f"{value/sum_val:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            thickness = 1
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = x0 + (dx - text_size[0]) // 2
            text_y = y0 + (dy + text_size[1]) // 2
            # Use black or white text depending on cell color brightness
            # brightness = sum(cell_color) / 3
            text_color = (0,0,0) #(255,255,255) if brightness < 128 else (0,0,0)
            cv2.putText(overlay, text, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)
    
    # 7. Blend overlay with original image
    cv2.addWeighted(overlay, overlay_alpha, out, 1 - overlay_alpha, 0, out)
    
    # 8. Highlight the cell with highest probability with black rectangle border
    max_row = max_idx // nx
    max_col = max_idx % nx
    max_x0, max_y0 = max_col * dx, max_row * dy
    max_x1, max_y1 = max_x0 + dx, max_y0 + dy
    cv2.rectangle(out, (max_x0, max_y0), (max_x1, max_y1), color=(0, 0, 0), thickness=3)

    # 9. Draw grid lines on top
    for i in range(1, nx):
        x = i * dx
        cv2.line(out, (x, 0), (x, h), color=grid_color, thickness=grid_thickness)
    for j in range(1, ny):
        y = j * dy
        cv2.line(out, (0, y), (w, y), color=grid_color, thickness=grid_thickness)

    return out

def class_counts(loader: DataLoader[Tuple[torch.Tensor, torch.Tensor]], num_classes: int = 256) -> torch.Tensor:
    """
    Count labels per class.

    Args:
        loader: yields (x, y) where y is one-hot or heatmap-like, e.g. (B, C) or (B, 1, G, G) with C=G*G.
        num_classes: total number of classes (C).

    Returns:
        Tensor of shape (num_classes,) with dtype=torch.long containing counts per class.
    """
    counts: torch.Tensor = torch.zeros(num_classes, dtype=torch.long)
    for _, y in loader:
        y_flat: torch.Tensor = y.view(y.size(0), -1)
        idx: torch.Tensor = y_flat.argmax(dim=1).to(torch.long).cpu()
        counts += torch.bincount(idx, minlength=num_classes)
    return counts

def main():
    parser = argparse.ArgumentParser(
        description="Visualize gaze distribution for tasks using dataset."
    )
    parser.add_argument("--task", "-t", type=str, default=None,
                        help="Task name to process. If not provided, processes all tasks defined in the config.")
    parser.add_argument("--output-dir", "-o", type=str, 
                        default="/home/ka/ka_anthropomatik/ka_eb5961/gaze_pred_training/graphics",
                        help="Directory to save the output image.")
    args = parser.parse_args()

    with initialize(version_base=None, config_path="../conf", job_name="gaze_and_model_gif"):
        cfg = compose(config_name="config")
    
    if args.task:
        tasks = [args.task]
        cfg.dataset.tasks = tasks
    else:
        tasks = cfg.dataset.tasks

    for task in tasks:
        dataset = PreprocessedGazeDatasetWorkspace(
            dir=cfg.dataset.dir,
            tasks=[task],
        )
        print(f"Dataset length for task {task}: {len(dataset)}")
        print("Creating DataLoader...")
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)
        print(f"Counting classes for task {task}...")
        counts = class_counts(loader, num_classes=16*16)
        print(f"Raw class counts for task {task}: {counts}")

        counts = counts.float() / counts.sum()
        img_sample = dataset[0][0].numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
        img_sample = cv2.resize(img_sample, (1024, 1024))
        img_sample = np.clip(img_sample, 0.0, 1.0)
        img_sample = (img_sample * 255).astype(np.uint8)
        img_sample = img_sample[:, :, ::-1]  # RGB -> BGR
        img_sample = cv2.rotate(img_sample, cv2.ROTATE_180)
        print(f"Sample image shape for task {task}: {img_sample.shape}")
        print("Overlaying grid and highlighting prediction...")
        img_with_grid = overlay_grid_and_highlight_pred(
            img=img_sample,
            values=counts,
            nx=16,
            ny=16,
            grid_color=(0, 255, 0),
            overlay_alpha=0.5
        )
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{task}_gaze_distribution.png"
        print(f"Saving image to {output_path}")
        cv2.imwrite(str(output_path), img_with_grid)
        print(f"Grid overlay for task {task} saved to {output_path}")

if __name__ == "__main__":
    main()
