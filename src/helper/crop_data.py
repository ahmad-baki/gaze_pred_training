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

# --- Source/crop constants ---
SRC_WIDTH = 640
SRC_HEIGHT = 360
CROP_X0 = 200
CROP_X1 = 560               # 200 + 360
CROP_WIDTH = CROP_X1 - CROP_X0  # 360

# def resize_image(inp_img: np.ndarray, width: int, height: int) -> np.ndarray:
#     """
#     inp_img: a numpy array
#     returns: numpy array resized to (width, height)
#     """
#     return cv2.resize(inp_img, (width, height))

def detect_need_crop(src_dir: Path) -> bool:
    """
    Decide if the folder needs cropping (images are 640-wide vs. already 360-wide).
    - If any PNG width > CROP_WIDTH => needs cropping.
    - Else, check depth .txt row count: 360*640 => needs cropping, 360*360 => no crop.
    - Default: assume cropping needed (safer).
    """
    for filename in os.listdir(src_dir):
        if filename.lower().endswith('.png'):
            try:
                with Image.open(src_dir / filename) as im:
                    w, h = im.size
                return w > CROP_WIDTH
            except Exception:
                pass

    for filename in os.listdir(src_dir):
        if filename.lower().endswith('.txt'):
            try:
                depth_data = np.loadtxt(src_dir / filename, dtype=np.float64)
                n = depth_data.shape[0]
                if n == (SRC_HEIGHT * CROP_WIDTH):
                    return False  # already cropped
                elif n == (SRC_HEIGHT * SRC_WIDTH):
                    return True   # needs cropping
            except Exception:
                pass

    return True

def crop_images_in_directory(
    src_dir: Path, tar_dir: Path, tar_width: int, tar_height: int
) -> tuple[bool, bool]:
    """
    Processes .png and .txt from src_dir, crops horizontally to [CROP_X0:CROP_X1] iff needed,
    ALWAYS resizes to (tar_width, tar_height), and writes to tar_dir (overwrites).

    returns (success, need_crop)
    """
    tar_dir.mkdir(parents=True, exist_ok=True)
    need_crop = detect_need_crop(src_dir)

    for filename in os.listdir(src_dir):
        src_path = src_dir / filename
        dst_path = tar_dir / filename

        # Copy other files (not json/txt/png) unchanged
        if not (filename.lower().endswith('.png') or filename.lower().endswith('.txt') or filename.lower().endswith('.json')):
            try:
                shutil.copy2(src_path, dst_path)
            except Exception as e:
                print(f"Error copying {src_path}: {e}")
                return False, need_crop
            continue

        # Depth text
        if filename.lower().endswith('.txt'):
            try:
                depth_data = np.loadtxt(src_path, dtype=np.float64)
                # Expect flat N x 3 where N in {360*640, 360*360}
                if depth_data.ndim != 2 or depth_data.shape[1] != 3:
                    raise ValueError(f"Unexpected depth format in {src_path}, shape: {depth_data.shape}")

                # Reshape to image
                n = depth_data.shape[0]
                if n == SRC_HEIGHT * SRC_WIDTH:
                    depth_img = depth_data.reshape(SRC_HEIGHT, SRC_WIDTH, 3)
                elif n == SRC_HEIGHT * CROP_WIDTH:
                    depth_img = depth_data.reshape(SRC_HEIGHT, CROP_WIDTH, 3)
                else:
                    raise ValueError(f"Unexpected depth length {n}, expected {SRC_HEIGHT*SRC_WIDTH} or {SRC_HEIGHT*CROP_WIDTH}")

                # Crop iff needed
                if need_crop and depth_img.shape[1] == SRC_WIDTH:
                    depth_img = depth_img[:, CROP_X0:CROP_X1, :]

                # # ALWAYS resize
                # depth_resized = resize_image(depth_img, tar_width, tar_height)

                # Overwrite
                np.savetxt(dst_path, depth_img.reshape(-1, 3), fmt='%d')
            except Exception as e:
                print(f"Error processing depth {src_path}: {e}")
                return False, need_crop
            continue

        # Images
        if filename.lower().endswith('.png'):
            try:
                with Image.open(src_path).convert('RGB') as img_pil:
                    img_npy = np.array(img_pil)

                # Crop iff needed
                if need_crop and img_npy.shape[1] == SRC_WIDTH:
                    img_npy = img_npy[:, CROP_X0:CROP_X1, :]

                # ALWAYS resize
                # resized = resize_image(img_npy, tar_width, tar_height)

                # Overwrite
                plt.imsave(dst_path, img_npy)
            except Exception as e:
                print(f"Error processing image {src_path}: {e}")
                return False, need_crop

        else:
            shutil.copy2(src_dir / filename, tar_dir / filename)
        # JSON is handled in gaze function
    return True, need_crop

def onehot_vector_gaze_dataset(
    src_dir: Path,
    tar_dir: Path,
    tar_width: int,
    tar_height: int,
    flip_vert: bool = True,
    will_crop: bool = True,
) -> bool | None:
    """
    Input JSON per frame: { "x": float, "y": float } in [0,1], normalized to the *original* (640x360).
    If will_crop, remap x into the cropped window: x' = (x*SRC_WIDTH - CROP_X0) / CROP_WIDTH, then clamp to [0,1].
    Output: tar_dir/gaze_vectors.npy as (N, tar_height, tar_width) one-hot maps.
    """
    tar_dir.mkdir(parents=True, exist_ok=True)
    out_path = tar_dir / 'gaze_vectors.npy'

    # If already exists, keep (comment this out if you want to always regenerate)
    if os.path.exists(out_path):
        print(f"Skipping {out_path}, already processed.")
        return None


    for filename in os.listdir(src_dir):
        if not filename.lower().endswith('.json'):
            continue

        try:
            with open(src_dir / filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in {src_dir / filename}: {e}")
            return False

        if 'x' not in data or 'y' not in data:
            print(f"Invalid data in {src_dir / filename}, missing 'x' or 'y': {data}")
            return False

        x_norm = float(data['x'])
        y_norm = float(data['y'])

        # Adjust x for cropping if applied to images/depth
        if will_crop:
            x_px_full = x_norm * SRC_WIDTH
            x_px_cropped = x_px_full - (SRC_WIDTH - CROP_X1)
            x_norm_effective = float(np.clip(x_px_cropped / CROP_WIDTH, 0.0, 1.0))
        else:
            x_norm_effective = x_norm

        new_data = {
            'x': x_norm_effective,
            'y': y_norm
        }

        # save as json-file
        with open(tar_dir / filename, 'w', encoding='utf-8') as f:
            json.dump(new_data, f)
    return True


TARGET_IMG_HEIGHT = 224
TARGET_IMG_WIDTH = 224
TARGET_GAZE_HEIGHT = 16
TARGET_GAZE_WIDTH = 16

SOURCE_PAR_DIR = Path('/pfs/work9/workspace/scratch/ka_eb5961-holo2gaze/new_frame/data/3d')
SUB_DIR = Path('sensors/continuous_device_')
TARGET_DIR = Path('/pfs/work9/workspace/scratch/ka_eb5961-holo2gaze/new_frame/data_cropped/3d')

if __name__ == "__main__":
    print("Starting preprocessing...")

    faulty_trajectories = []
    for task in SOURCE_PAR_DIR.iterdir():
        if not task.is_dir():
            print(f"Skipping {SOURCE_PAR_DIR / task.name}, not a directory.")
            continue

        for trajectory in task.iterdir():
            if not trajectory.is_dir():
                print(f"Skipping {task / trajectory.name}, not a directory.")
                continue

            print(f"Processing {task / trajectory.name} ...")
            src_dir = SOURCE_PAR_DIR / task.name / trajectory.name / SUB_DIR
            target_dir = TARGET_DIR / task.name / trajectory.name / SUB_DIR
            if not src_dir.exists():
                print(f"Source directory {src_dir} does not exist, skipping...")
                faulty_trajectories.append(src_dir)
                continue
            target_dir.mkdir(parents=True, exist_ok=True)

            # 1) ALWAYS resize/crop images & depth (overwrites). Also learn if cropping was needed.
            success, need_crop = crop_images_in_directory(
                src_dir=src_dir,
                tar_dir=target_dir,
                tar_width=TARGET_IMG_WIDTH,
                tar_height=TARGET_IMG_HEIGHT
            )
            if not success:
                faulty_trajectories.append(src_dir)
                continue
            print(f"Resized images in {task / trajectory.name}. Cropping applied: {need_crop}")

            # 2) Build gaze vectors with x adjusted iff cropping applied
            success = onehot_vector_gaze_dataset(
                src_dir=src_dir,
                tar_dir=target_dir,
                tar_width=TARGET_GAZE_WIDTH,
                tar_height=TARGET_GAZE_HEIGHT,
                flip_vert=True,
                will_crop=need_crop,
            )

            if success is None:
                print(f"Skipping {trajectory.name}, gaze vectors already exist.")
                continue
            if not success:
                faulty_trajectories.append(src_dir)
                continue
            print(f"Created gaze vectors for {trajectory.name}.")

            # copy all other folder/files in SOURCE_PAR_DIR / task.name / trajectory.name over
            for item in (SOURCE_PAR_DIR / task.name / trajectory.name).iterdir():
                if item.name == "sensors":
                    continue

                if item.is_file():
                    print(f"Copying file {item.name} to {target_dir}")
                    shutil.copy2(item, TARGET_DIR / task.name / trajectory.name / item.name)
                elif item.is_dir():
                    print(f"Copying directory {item.name} to {target_dir}")
                    shutil.copytree(item, TARGET_DIR / task.name / trajectory.name / item.name, dirs_exist_ok=True)

    if faulty_trajectories:
        print("Some trajectories failed to process:")
        for traj in faulty_trajectories:
            print(f"- {traj}")
            if (TARGET_DIR / traj.name).exists():
                print(f"Deleting faulty target directory: {TARGET_DIR / traj.name}")
                shutil.rmtree(TARGET_DIR / traj.name)
    else:
        print("All trajectories processed successfully.")
    print("Preprocessing completed.")
