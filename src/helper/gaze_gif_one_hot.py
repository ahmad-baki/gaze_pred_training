import json
from typing import Tuple
import sys
import os
import torchvision.transforms as T


# Add the parent directory (src) to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.gaze_predictor import GazePredictorModel

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import cv2
import argparse
from alive_progress import alive_bar
import imageio
from PIL import Image
from hydra import compose, initialize
from omegaconf import OmegaConf
import copy

def rescale_img(img: np.ndarray) -> np.ndarray:
    target_width = int(img.shape[1] / 16) * 16
    target_height = int(img.shape[0] / 16) * 16
    return cv2.resize(img, (target_width, target_height))

def overlay_grid_and_highlight_true(
    img: np.ndarray,
    pos_tuple: Tuple[int, int],
    nx: int = 16,
    ny: int = 16,
    grid_color: Tuple[int,int,int] = (0, 255, 0),
    grid_thickness: int = 1,
    highlight_color: Tuple[int,int,int] = (0, 0, 255),
    highlight_thickness: int = 3,
    fill_highlight: bool = True,
    overlay_alpha: float = 0.4
) -> 'np.ndarray':
    """
    Overlays an nx×ny grid on the image and highlights the cell indicated by a one-hot torch.Tensor.

    Parameters
    ----------
    img : numpy.ndarray
        An already-loaded BGR numpy array.
    one_hot : torch.Tensor, shape (nx*ny,)
        One-hot tensor with exactly one element == 1 indicating which cell to highlight.
    nx, ny : int
        Number of columns and rows in the grid.
    grid_color : (B, G, R)
        Color of the grid lines.
    grid_thickness : int
        Thickness of the grid lines.
    highlight_color : (B, G, R)
        Color for highlighting the active cell.
    highlight_thickness : int
        Thickness of the highlight border (ignored if fill_highlight=True).
    fill_highlight : bool
        If True, fill the cell with a semi-transparent overlay; otherwise draw only the border.
    overlay_alpha : float in [0,1]
        Opacity of the filled highlight (only used if fill_highlight=True).

    Returns
    -------
    numpy.ndarray
        The annotated BGR image.
    """
    # 1. Load image if needed
    if isinstance(img, str):
        img = cv2.imread(img)
        if img is None:
            raise FileNotFoundError(f"Could not load image at '{img}'")
    out = img.copy()
    h, w = out.shape[:2]

    # 2. Compute per-cell size
    dx = w // nx
    dy = h // ny

    # 3. Draw grid lines
    for i in range(1, nx):
        x = i * dx
        cv2.line(out, (x, 0), (x, h), color=grid_color, thickness=grid_thickness)
    for j in range(1, ny):
        y = j * dy
        cv2.line(out, (0, y), (w, y), color=grid_color, thickness=grid_thickness)

    # 4. Locate active cell 
    row, col = pos_tuple
    x0, y0 = col * dx, row * dy
    x1, y1 = x0 + dx, y0 + dy

    # 5. Highlight that cell
    if fill_highlight:
        overlay = out.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color=highlight_color, thickness=-1)
        cv2.addWeighted(overlay, overlay_alpha, out, 1 - overlay_alpha, 0, out)
    else:
        cv2.rectangle(out, (x0, y0), (x1, y1), color=highlight_color, thickness=highlight_thickness)

    return out


def process_gaze_gif(source_dir, target_dir, task, skip_amount=10):
    """
    Process gaze data and images to create GIF files.
    
    Args:
        source_dir (str): Source directory containing the data
        target_dir (str): Target directory for output GIFs
        task (str): Task name to process ('all' for all tasks)
        skip_amount (int): Number of frames to skip between each processed frame
    """
    if not os.path.exists(source_dir):
        print(f'[ERROR] directory {source_dir} does not exist, exiting')
        return False

    if not os.path.exists(target_dir):
        print(f'[INFO] target directory {target_dir} does not exist, creating it')
        os.makedirs(target_dir)

    task_dir = os.listdir(source_dir) if task == 'all' else [task]
    if len(task_dir) == 0:
        print(f'[ERROR] no task directories found in {source_dir}, exiting')
        return False

    with initialize(version_base=None, config_path="../conf", job_name="gaze_and_model_gif"):
        cfg = compose(config_name="config")

    print(type(cfg))
    print(OmegaConf.to_yaml(cfg))

    traj_src_dir = []
    traj_target_dir = []

    for task_name in task_dir:
        task_path = os.path.join(source_dir, task_name)
        if not os.path.isdir(task_path):
            print(f'[WARN] {task_path} is not a directory, skipping')
            continue
        
        for pred_pos_idx, traj_folder in enumerate(os.listdir(task_path)):
            traj_dir = os.path.join(task_path, traj_folder, "sensors", "continuous_device_")
            if not os.path.isdir(traj_dir):
                print(f'[WARN] {traj_dir} is not a directory, skipping')
                continue

            target_gif_path = os.path.join(target_dir, task_name, f"{traj_folder}.gif")
            if os.path.exists(target_gif_path):
                print(f'[WARN] target file {target_gif_path} already exists, skipping')
                continue
            traj_target_dir.append(target_gif_path)
            traj_src_dir.append(traj_dir)

    print(f'[INFO] found {len(traj_src_dir)} trajectories to process')
    
    with alive_bar(len(traj_src_dir), title='Processing trajectories') as bar:
        for traj_dir, target_gif_path in zip(traj_src_dir, traj_target_dir):
            bar.text(f'Processing {traj_dir} -> {target_gif_path}')
            bar()

            files = os.listdir(traj_dir)
            image_files = [os.path.join(traj_dir, f) for f in files if f.endswith('.png')]
            image_files = sorted(image_files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            gaze_file = os.path.join(traj_dir, "gaze_vectors.npy")

            print(f'[INFO] {traj_dir}: found {len(image_files)} image files and found following gaze-file: {gaze_file}')

            # create gif from images and gaze points
            target_folder = os.path.dirname(target_gif_path)
            if not os.path.exists(target_folder):
                print(f'[INFO] target folder {target_folder} does not exist, creating it')
                os.makedirs(target_folder)
                
            # load numpy
            gaze_pos_onehot = np.load(gaze_file) # shape: (N, H, W)
            with imageio.get_writer(target_gif_path, mode='I') as writer:
                palette = None
                for i in range(0, len(image_files), skip_amount):

                    image_file_path = image_files[i]
                    image: np.ndarray = cv2.imread(image_file_path)
                    image = rescale_img(image)                        
                    # rotate image 180 degrees
                    # image_rotated = cv2.rotate(copy.deepcopy(image), cv2.ROTATE_180)
                    image_rotated = cv2.rotate(image, cv2.ROTATE_180)


                    # 1) True img and label
                    # gaze_file_path = gaze_files[i]
                    # gaze_pos_rel = json.load(open(gaze_file_path, 'r'))
                    # gaze_pos_abs = (gaze_pos_rel['x'] * image_rotated.shape[1],
                    #                 (1 - gaze_pos_rel['y']) * image_rotated.shape[0])

                    gaze_pos_onehot_i = np.flipud(gaze_pos_onehot[i].reshape(16, 16))
                    idx = int(gaze_pos_onehot_i.argmax())
                    row = idx // 16
                    col = idx % 16

                    # draw gaze point
                    # cv2.circle(image_rotated, (int(gaze_pos_abs[0]), int(gaze_pos_abs[1])), 5, (0, 0, 255), -1)
                    # image_converted = cv2.cvtColor(image_rotated, cv2.COLOR_BGR2RGB)

                    # Overlay grid and highlight the true cell
                    true_pos_tuple = (int(row), int(col))
                    image_label_overlay = overlay_grid_and_highlight_true(
                        img=image_rotated,
                        pos_tuple=true_pos_tuple,
                        nx=16, ny=16,
                        grid_color=(0, 255, 0),
                        highlight_color=(0, 0, 255),
                        fill_highlight=True,
                        overlay_alpha=0.4
                    )

                    writer.append_data(image_label_overlay)  # type: ignore
            print(f'[INFO] saved GIF to {target_gif_path}')
    return True


def main():
    """Main function with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Generate GIF files from gaze data and images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--source-dir', '-s',
        type=str,
        default="/pfs/work9/workspace/scratch/ka_eb5961-holo2gaze/new_frame/processed_cropped/3d",
        help='Source directory containing the data'
    )
    
    parser.add_argument(
        '--target-dir', '-t',
        type=str,
        default="/pfs/work9/workspace/scratch/ka_eb5961-holo2gaze/new_frame/gaze_pos_gif/cucumber_in_pot",
        help='Target directory for output GIFs'
    )
    
    parser.add_argument(
        '--task',
        type=str,
        default='cucumber_in_pot',
        help='Task name to process (use "all" to process all tasks)'
    )
    
    parser.add_argument(
        '--skip-amount', '-k',
        type=int,
        default=10,
        help='Number of frames to skip between each processed frame'
    )
    
    args = parser.parse_args()
    
    print(f"[INFO] Starting gaze GIF generation with parameters:")
    print(f"  Source directory: {args.source_dir}")
    print(f"  Target directory: {args.target_dir}")
    print(f"  Task: {args.task}")
    print(f"  Skip amount: {args.skip_amount}")
    
    success = process_gaze_gif(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        task=args.task,
        skip_amount=args.skip_amount
    )
    
    if success:
        print("[INFO] Processing completed successfully!")
    else:
        print("[ERROR] Processing failed!")
        exit(1)


if __name__ == "__main__":
    main()