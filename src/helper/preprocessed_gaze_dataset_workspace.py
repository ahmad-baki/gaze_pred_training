import os
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class PreprocessedGazeDatasetWorkspace(Dataset):
    """
    Loads:
      - pre-resized RGB images from `image_dir`
      - pre-computed gaze maps from `label_dir/gaze_vectors.npy`
    and returns (image_tensor, gaze_map_tensor).
    """
    SUB_PATH = "sensors/continuous_device_"
    def __init__(self,
                 dir: Path,
                 transform=None,
                 task: str | None = None
                 ):
        self.dir = Path(dir)

        self.image_files = []
        self.gaze_array = np.empty((0, 16, 16), dtype=np.float32)  # shape: (N, H, W)

        if task is not None:
            # if a specific task is given, only load that one
            task_dir = self.dir / task
            if not task_dir.exists():
                raise ValueError(f"Task directory {task_dir} does not exist.")
            self.get_data(task_dir)
        else:
            for task_iter in self.dir.iterdir():
                if not task_iter.is_dir():
                    continue
                self.get_data(task_iter)
        

        # optional image transform (e.g. ToTensor, Normalize)
        self.transform = transform or T.ToTensor()

    def get_data(self, task_dir: Path):
        # iterate over trajectories
        for trajectory in task_dir.iterdir():
            if not trajectory.is_dir():
                continue
            
            path = trajectory / self.SUB_PATH
            self.image_files += sorted([
                    path / fn for fn in os.listdir(path)
                    if fn.lower().endswith(('.png', '.jpg', '.jpeg'))
                    ])

                # load all gaze maps at once
            gaze_path = path / 'gaze_vectors.npy'
            assert gaze_path.exists(), \
                    f"Gaze vectors file not found: {gaze_path}"
            self.gaze_array = np.append(self.gaze_array, np.load(gaze_path), axis=0)  # shape: (N, H, W)

            assert len(self.gaze_array) == len(self.image_files), \
                    f"Mismatch: {len(self.image_files)} images vs {len(self.gaze_array)} gaze maps"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # load image
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)  # e.g. (3, H, W)

        # load precomputed gaze map
        gaze_map = self.gaze_array[idx]    # (H, W)
        gaze_tensor = torch.from_numpy(gaze_map).float()  # (H, W)
        # reshape to (H*W,) (flatten) if needed
        gaze_tensor = gaze_tensor.view(-1)
        # if you need a channel dim: uncomment
        # gaze_tensor = gaze_tensor.unsqueeze(0)  # (1, H, W)

        return img_tensor, gaze_tensor


# ——— USAGE EXAMPLE ———
DIR     = '/pfs/work9/workspace/scratch/ka_eb5961-holo2gaze/old_frame/raw/2d'

if __name__ == '__main__':

    # define any extra transforms here
    transform = T.Compose([
        T.ToTensor(),
    ])

    dataset = PreprocessedGazeDatasetWorkspace(
        dir=Path(DIR),
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # quick check
    for imgs, gazes in dataloader:
        print("Images:", imgs.shape)  # (B, 3, H_img, W_img) = (B, 3, 224, 224) 
        print("Gazes:", gazes.shape) # (B, H_gaze * W_gaze)  = (B, 16 * 16)
        break
