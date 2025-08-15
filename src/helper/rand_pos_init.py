#!/usr/bin/env python3
"""
Load an image once, then (re)draw two random X marks on it every time you press SPACE.
Quit with Q or ESC.

Usage:
  python place_random_x_live.py /path/to/image.jpg --size 30 --thickness 3 --seed 42
"""

import argparse
import cv2
import numpy as np
from typing import Tuple

def draw_x(img: np.ndarray, center: Tuple[int, int], size: int = 20,
           color: Tuple[int, int, int] = (0, 0, 255), thickness: int = 2) -> None:
    """Draw an 'X' centered at (x, y) on the given image (in-place)."""
    x, y = center
    pt1 = (x - size, y - size)
    pt2 = (x + size, y + size)
    pt3 = (x - size, y + size)
    pt4 = (x + size, y - size)
    cv2.line(img, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)
    cv2.line(img, pt3, pt4, color, thickness, lineType=cv2.LINE_AA)

def random_points(h: int, w: int, margin: int, rng: np.random.Generator,
                  min_sep: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Two random points inside [margin, w-margin)×[margin, h-margin) with a minimal separation."""
    x1 = int(rng.integers(margin, w - margin))
    y1 = int(rng.integers(margin, h - margin))
    p1 = (x1, y1)

    for _ in range(100):
        x2 = int(rng.integers(margin, w - margin))
        y2 = int(rng.integers(margin, h - margin))
        if (x2 - x1) ** 2 + (y2 - y1) ** 2 >= min_sep ** 2:
            return p1, (x2, y2)

    # Fallback if separation condition failed repeatedly
    return p1, (x2, y2)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str, help="Path to the image file")
    parser.add_argument("--size", type=int, default=20, help="Half-size of each X in pixels")
    parser.add_argument("--thickness", type=int, default=2, help="Line thickness of the X")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    args = parser.parse_args()

    # Read the image ONCE at the beginning
    base = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if base is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    h, w = base.shape[:2]
    margin = max(args.size + args.thickness, 1)
    if w <= 2 * margin or h <= 2 * margin:
        raise ValueError("Image is too small for the chosen size/thickness.")

    rng = np.random.default_rng(args.seed)
    window_name = "Random X Marks (SPACE = new, Q/ESC = quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    def refresh() -> None:
        p1, p2 = random_points(h, w, margin, rng, min_sep=args.size)
        frame = base.copy()  # draw on a fresh copy each time
        draw_x(frame, p1, size=args.size, thickness=args.thickness)
        draw_x(frame, p2, size=args.size, thickness=args.thickness)
        cv2.imshow(window_name, frame)

    # Show once, then react to key presses
    refresh()
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord('q')):  # ESC or 'q'
            break
        if key == 32:  # SPACE
            refresh()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
