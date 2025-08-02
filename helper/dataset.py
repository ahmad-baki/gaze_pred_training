import os
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class PreprocessedGazeDataset(Dataset):
    """
    Loads:
      - pre-resized RGB images from `image_dir`
      - pre-computed gaze maps from `label_dir/gaze_vectors.npy`
    and returns (image_tensor, gaze_map_tensor).
    """
    def __init__(self,
                 image_dir: Path,
                 label_dir: Path,
                 transform=None):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)

        # collect and sort image files
        self.image_files = sorted([
            fn for fn in os.listdir(self.image_dir)
            if fn.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        # load all gaze maps at once
        gaze_path = self.label_dir / 'gaze_vectors.npy'
        self.gaze_array = np.load(gaze_path)  # shape: (N, H, W)
        assert len(self.gaze_array) == len(self.image_files), \
            f"Mismatch: {len(self.image_files)} images vs {len(self.gaze_array)} gaze maps"

        # optional image transform (e.g. ToTensor, Normalize)
        self.transform = transform or T.ToTensor()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # load image
        img_name = self.image_files[idx]
        img = Image.open(self.image_dir / img_name).convert('RGB')
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
IMG_DIR     = "/home/ka/ka_anthropomatik/ka_eb5961/gaze_pred_training/test/input/processed"
LABEL_DIR   = "/home/ka/ka_anthropomatik/ka_eb5961/gaze_pred_training/test/input/processed"


if __name__ == '__main__':

    # define any extra transforms here
    transform = T.Compose([
        T.ToTensor(),
    ])

    dataset = PreprocessedGazeDataset(
        image_dir=Path(IMG_DIR),
        label_dir=Path(LABEL_DIR),
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
