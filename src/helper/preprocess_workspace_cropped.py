import json
import os
from pathlib import Path
import shutil

from alive_progress import alive_bar
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T

def resize_image(inp_img: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    inp_img: a numpy array
    returns: numpy array resized to (width, height)
    """
    resize_img = cv2.resize(inp_img, (width, height))
    return resize_img

def resize_images_in_directory(src_dir: Path, tar_dir: Path, tar_width: int, tar_height: int):
    """
    Reads all .png/.jpg from src_dir, resizes them to (tar_width, tar_height),
    and writes them to tar_dir with the same filenames.
    """
    tar_dir.mkdir(parents=True, exist_ok=True)
    x0, x1 = 160, 520          # 520 = 160 + 360

    for filename in os.listdir(src_dir):
        if filename.lower().endswith('.txt'):
            try:
                depth_data = np.loadtxt(src_dir / filename, dtype=np.float64)
                cropped_depth = depth_data.reshape(360, 640, 3)[:, x0:x1, :]
                depth_data = resize_image(cropped_depth, tar_width, tar_height)
                np.savetxt(tar_dir / filename, depth_data.reshape(-1, 3), fmt='%d')
            except Exception as e:
                print(f"Error processing {src_dir / filename}: {e}")
                return False
        elif filename.lower().endswith('.png'):
            # Check if the file already exists in the target directory
            if os.path.exists(tar_dir / filename):
                print(f"Skipping {tar_dir / filename}, already processed.")
                continue

            img_path = src_dir / filename
            try:
                img = Image.open(img_path).convert('RGB')
                img_npy = np.array(img)
                img_cropped = img_npy[:, x0:x1, :]
                cropped_img = resize_image(img_cropped, tar_width, tar_height)
                plt.imsave(tar_dir / filename, cropped_img)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                return False
        elif not filename.lower().endswith('.json'):
            # copy it to target
            shutil.copy2(src_dir / filename, tar_dir / filename)
    return True

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
      Assumes x, y are in [0, 1] range and that origin is top-left. If origin is bottom-left, set flip_vert=False.
    Output: tar_dir/gaze_vectors.npy containing an array of 2D one-hot gaze maps.
    """
    tar_dir.mkdir(parents=True, exist_ok=True)
    gaze_data = []
    
    # Check if the gaze vector file already exists
    if os.path.exists(tar_dir / 'gaze_vectors.npy'):
        print(f"Skipping {tar_dir / 'gaze_vectors.npy'}, already processed.")
        return None
    
    for filename in os.listdir(src_dir):
        if not filename.lower().endswith('.json'):
            continue
        
        try:
            # Load JSON data
            with open(src_dir / filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in {src_dir / filename}: {e}")
            return False

        if 'x' not in data or 'y' not in data:
            print(f"Invalid data in {src_dir / filename}, missing 'x' or 'y': {data}")
            return False
        
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

    if len(gaze_data) == 0:
        print(f"No valid gaze data found in {src_dir}. Skipping...")
        return False
    gaze_array = np.stack(gaze_data, axis=0)
    np.save(tar_dir / 'gaze_vectors.npy', gaze_array)
    return True

TARGET_IMG_HEIGHT = 224
TARGET_IMG_WIDTH = 224
TARGET_GAZE_HEIGHT = 16
TARGET_GAZE_WIDTH = 16


SOURCE_PAR_DIR = Path('/pfs/work9/workspace/scratch/ka_eb5961-holo2gaze/old_frame/data/3d')
SUB_DIR = Path('sensors/continuous_device_')
TARGET_DIR = Path('/pfs/work9/workspace/scratch/ka_eb5961-holo2gaze/old_frame/processed_cropped/3d')

if __name__ == "__main__":
    print("Starting preprocessing...")
    # for each folder in SOURCE_DIR, resize images and create one-hot encoded gaze vectors

    faulty_trajectories = []
    for task in SOURCE_PAR_DIR.iterdir():

        if not task.is_dir():
            print(f"Skipping {SOURCE_PAR_DIR / task.name}, not a directory.")
            continue

        # iterate over trajectories
        for trajectory in task.iterdir():
            if not trajectory.is_dir():
                print(f"Skipping {task / trajectory.name}, not a directory.")
                continue

            print(f"Processing {task / trajectory.name} ...")
            # Resize images
            src_dir = SOURCE_PAR_DIR / task.name / trajectory.name / SUB_DIR
            target_dir = TARGET_DIR / task.name / trajectory.name / SUB_DIR
            if not src_dir.exists():
                print(f"Source directory {src_dir} does not exist, skipping...")
                faulty_trajectories.append(src_dir)
                continue
            if not target_dir.exists():
                target_dir.mkdir(parents=True, exist_ok=True)


            # Create one-hot encoded gaze vectors
            success = onehot_vector_gaze_dataset(
                src_dir=src_dir,
                tar_dir=target_dir,
                tar_width=TARGET_GAZE_WIDTH,
                tar_height=TARGET_GAZE_HEIGHT,
                flip_vert=True  # Set to False if you don't want to flip the y-coordinate
            )
            
            if success is None:
                print(f"Skipping {trajectory.name}, gaze vectors already exist.")
                continue

            if not success:
                faulty_trajectories.append(src_dir)
                continue
            print(f"Created gaze vectors for {trajectory.name}.") 


            # Resize images
            success = resize_images_in_directory(
                src_dir=src_dir,
                tar_dir=target_dir,
                tar_width=TARGET_IMG_WIDTH,
                tar_height=TARGET_IMG_HEIGHT
            )
            if not success:
                faulty_trajectories.append(src_dir)
                continue
            print(f"Resized images in {task / trajectory.name}.")
    if faulty_trajectories:
        print("Some trajectories failed to process:")
        for traj in faulty_trajectories:
            print(f"- {traj}")
            # delete target directory if exists
            if (TARGET_DIR / traj.name).exists():
                print(f"Deleting faulty target directory: {TARGET_DIR / traj.name}")
                shutil.rmtree(TARGET_DIR / traj.name)
    else:
        print("All trajectories processed successfully.")
    print("Preprocessing completed.")
