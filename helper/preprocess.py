import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T

def resize_image(inp_img: Image.Image, width: int, height: int) -> Image.Image:
    """
    inp_img: a PIL Image
    returns: PIL Image resized to (width, height)
    """
    transform = T.Resize((height, width))
    return transform(inp_img)

def resize_images_in_directory(src_dir: Path, tar_dir: Path, tar_width: int, tar_height: int):
    """
    Reads all .png/.jpg from src_dir, resizes them to (tar_width, tar_height),
    and writes them to tar_dir with the same filenames.
    """
    tar_dir.mkdir(parents=True, exist_ok=True)

    for filename in os.listdir(src_dir):
        if not (filename.lower().endswith('.png') or filename.lower().endswith('.jpg')):
            continue

        img_path = src_dir / filename
        img = Image.open(img_path).convert('RGB')
        resized_img = resize_image(img, tar_width, tar_height)
        # save via matplotlib to preserve original format & range
        plt.imsave(tar_dir / filename, np.array(resized_img))

def onehot_vector_gaze_dataset(
    src_dir: Path,
    tar_dir: Path,
    tar_width: int,
    tar_height: int,
    flip_vert: bool = True
):
    """
    Input: src_dir with JSON files, each:
      { "x": float, "y": float }
    Output: tar_dir/gaze_vectors.npy containing an array of 2D one-hot gaze maps.
    """
    tar_dir.mkdir(parents=True, exist_ok=True)
    gaze_data = []

    for filename in os.listdir(src_dir):
        if not filename.lower().endswith('.json'):
            continue

        with open(src_dir / filename, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # create blank map
        gaze_map = np.zeros((tar_height, tar_width), dtype=np.uint8)

        # compute pixel coords
        x_px = int(data['x'] * tar_width)
        # optionally flip vertical coordinate
        y_px = int((1 - data['y']) * tar_height) if flip_vert else int(data['y'] * tar_height)

        # clamp coords to valid range
        x_px = np.clip(x_px, 0, tar_width - 1)
        y_px = np.clip(y_px, 0, tar_height - 1)

        gaze_map[y_px, x_px] = 1
        gaze_data.append(gaze_map)

    gaze_array = np.stack(gaze_data, axis=0)
    np.save(tar_dir / 'gaze_vectors.npy', gaze_array)



TARGET_IMG_HEIGHT = 224
TARGET_IMG_WIDTH = 224
TARGET_GAZE_HEIGHT = 16
TARGET_GAZE_WIDTH = 16
SOURCE_IMG_DIR = Path('/home/ka/ka_anthropomatik/ka_eb5961/gaze_pred_training/test/input/raw')
TARGET_IMG_DIR = Path('/home/ka/ka_anthropomatik/ka_eb5961/gaze_pred_training/test/input/processed')
SOURCE_GAZE_DIR = Path('/home/ka/ka_anthropomatik/ka_eb5961/gaze_pred_training/test/input/raw')
TARGET_GAZE_DIR = Path('/home/ka/ka_anthropomatik/ka_eb5961/gaze_pred_training/test/input/processed')

if __name__ == "__main__":
    print("Starting preprocessing...")
    # Resize images
    resize_images_in_directory(
        src_dir=SOURCE_IMG_DIR,
        tar_dir=TARGET_IMG_DIR,
        tar_width=TARGET_IMG_WIDTH,
        tar_height=TARGET_IMG_HEIGHT
    )
    print("Preprocessing completed: Images resized.")
    # Create one-hot encoded gaze vectors
    onehot_vector_gaze_dataset(
        src_dir=SOURCE_GAZE_DIR,
        tar_dir=TARGET_GAZE_DIR,
        tar_width=TARGET_GAZE_WIDTH,
        tar_height=TARGET_GAZE_HEIGHT,
        flip_vert=True  # Set to False if you don't want to flip the y-coordinate
    )
    print("Gaze vectors created.")
    print("Preprocessing completed.")


