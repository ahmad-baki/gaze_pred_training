import json
import cv2
import os
import argparse
from alive_progress import alive_bar
import imageio
from PIL import Image 


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

    task_dir = os.listdir(source_dir) if task == '' else [task]
    if len(task_dir) == 0:
        print(f'[ERROR] no task directories found in {source_dir}, exiting')
        return False

    traj_src_dir = []
    traj_target_dir = []

    for task_name in task_dir:
        task_path = os.path.join(source_dir, task_name)
        if not os.path.isdir(task_path):
            print(f'[WARN] {task_path} is not a directory, skipping')
            continue
        
        for idx, traj_folder in enumerate(os.listdir(task_path)):
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

            files = sorted(os.listdir(traj_dir))

            # get union of image and gaze files that have the same name, ignore the extension
            paired_files = []
            for f in files:
                if f.endswith('.png'):
                    gaze_file = f.replace('.png', '.json')
                    if gaze_file in files:
                        paired_files.append(f.replace('.png', ''))
                
            # sort paired files by their float value in the name
            paired_files = sorted(paired_files, key=lambda x: int(x))

            image_files = [os.path.join(traj_dir, f"{f}.png") for f in paired_files]
            gaze_files = [os.path.join(traj_dir, f"{f}.json") for f in paired_files]

            print(f'[INFO] found {len(image_files)} image files and {len(gaze_files)} gaze files'
                f' in {traj_dir}')

            # create gif from images and gaze points
            target_folder = os.path.dirname(target_gif_path)
            if not os.path.exists(target_folder):
                print(f'[INFO] target folder {target_folder} does not exist, creating it')
                os.makedirs(target_folder)
                
            with imageio.get_writer(target_gif_path, mode='I') as writer:
                palette = None
                for i in range(0, len(image_files), skip_amount):
                    image_file_path = image_files[i]
                    gaze_file_path = gaze_files[i]
                    image = cv2.imread(image_file_path)
                    # rotate image 180 degrees
                    image = cv2.rotate(image, cv2.ROTATE_180)

                    gaze_pos_rel = json.load(open(gaze_file_path, 'r'))
                    gaze_pos_abs = (gaze_pos_rel['x'] * image.shape[1],
                                    (1 - gaze_pos_rel['y']) * image.shape[0])

                    # draw gaze point
                    cv2.circle(image, (int(gaze_pos_abs[0]), int(gaze_pos_abs[1])), 5, (0, 0, 255), -1)
                    image_coverted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    # # Displaying the converted image
                    # pil_image = Image.fromarray(image_coverted)
                    # if i == 0: # Quantize the first frame and get the palette
                    #     pil_image = pil_image.quantize(colors=254, method=Image.Quantize.MAXCOVERAGE)
                    #     palette = pil_image
                    # else: # Quantize all other frames to the same palette
                    #     pil_image = pil_image.quantize(colors=254, palette=palette, method=Image.Quantize.MAXCOVERAGE)
                    writer.append_data(image_coverted)
    
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
        default="/pfs/work9/workspace/scratch/ka_eb5961-holo2gaze/new_frame/data_cropped/3d",
        help='Source directory containing the data'
    )
    
    parser.add_argument(
        '--target-dir', '-t',
        type=str,
        default="/pfs/work9/workspace/scratch/ka_eb5961-holo2gaze/new_frame/gaze_pos_gif_point/",
        help='Target directory for output GIFs'
    )
    
    parser.add_argument(
        '--task',
        type=str,
        default='',
        help='Task name to process (use empty string to process all tasks)'
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


# Legacy commented code for reference:
        # while (curr_idx < len(paired_files) and break_flag is not True):
        #     image_file_path = image_files[curr_idx]
        #     image = cv2.imread(image_file_path)
        #     # rotate image 180 degrees
        #     image = cv2.rotate(image, cv2.ROTATE_180)

        #     gaze_file_path = gaze_files[curr_idx]
        #     gaze_pos_rel = json.load(open(gaze_file_path, 'r'))
        #     gaze_pos_abs = (gaze_pos_rel['x'] * image.shape[1],
        #                     gaze_pos_rel['y'] * image.shape[0])

        #     # draw gaze point
        #     cv2.circle(image, (int(gaze_pos_abs[0]), int(gaze_pos_abs[1])), 5, (0, 255, 0), -1)