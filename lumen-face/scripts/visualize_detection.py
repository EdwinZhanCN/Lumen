#!/usr/bin/env python3
"""Simple script to draw bounding boxes and landmarks on images.

This is a standalone utility that doesn't depend on the lumen-face package.
Just provide the image path and bbox/landmark coordinates.

Usage:
    python scripts/visualize_detection.py <input_image> <output_image> [options]

Example:
    # Draw a single bbox
    python scripts/visualize_detection.py photo.jpg result.jpg --bbox 290.4 215.4 515.4 526.6

    # Draw multiple bboxes
    python scripts/visualize_detection.py photo.jpg result.jpg \
        --bbox 290.4 215.4 515.4 526.6 \
        --bbox 100.0 50.0 200.0 180.0

    # Draw bbox with landmarks (5 points: left_eye, right_eye, nose, left_mouth, right_mouth)
    python scripts/visualize_detection.py photo.jpg result.jpg \
        --bbox 290.4 215.4 515.4 526.6 \
        --landmarks 350.2 320.5 420.8 318.3 385.0 380.2 360.5 450.1 410.2 448.7

    # With confidence score
    python scripts/visualize_detection.py photo.jpg result.jpg \
        --bbox 290.4 215.4 515.4 526.6 \
        --score 0.95
"""

import argparse
import sys
from pathlib import Path
from typing import cast

import cv2
import numpy as np


def draw_bbox(
    image: np.ndarray,
    bbox: tuple[float, float, float, float],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    label: str | None = None,
) -> np.ndarray:
    """Draw a bounding box on the image.

    Args:
        image: Input image in BGR format
        bbox: Bounding box coordinates (x1, y1, x2, y2)
        color: Box color in BGR format
        thickness: Line thickness
        label: Optional text label to display above the box

    Returns:
        Image with drawn bounding box
    """
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    if label:
        # Get text size to create background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )

        # Draw background rectangle for text
        cv2.rectangle(
            image,
            (x1, y1 - text_height - baseline - 10),
            (x1 + text_width + 10, y1),
            color,
            -1,
        )

        # Draw text
        cv2.putText(
            image,
            label,
            (x1 + 5, y1 - 5),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
        )

    return image


def draw_landmarks(
    image: np.ndarray,
    landmarks: list[tuple[float, float]],
    colors: list[tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Draw facial landmarks on the image.

    Args:
        image: Input image in BGR format
        landmarks: List of (x, y) landmark coordinates
        colors: Optional list of colors for each landmark

    Returns:
        Image with drawn landmarks
    """
    # Default colors for 5-point landmarks (InsightFace format)
    if colors is None:
        colors = [
            (255, 0, 0),  # Blue - left eye
            (0, 0, 255),  # Red - right eye
            (0, 255, 255),  # Yellow - nose
            (255, 0, 255),  # Magenta - left mouth
            (255, 255, 0),  # Cyan - right mouth
        ]

    for idx, (x, y) in enumerate(landmarks):
        x, y = int(x), int(y)
        color = colors[idx] if idx < len(colors) else (255, 255, 255)

        # Draw filled circle
        cv2.circle(image, (x, y), 3, color, -1)
        # Draw outer circle
        cv2.circle(image, (x, y), 5, color, 1)

    return image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Draw bounding boxes and landmarks on images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single bbox
  python scripts/visualize_detection.py photo.jpg result.jpg --bbox 290.4 215.4 515.4 526.6

  # Bbox with confidence score
  python scripts/visualize_detection.py photo.jpg result.jpg \\
      --bbox 290.4 215.4 515.4 526.6 --score 0.95

  # Bbox with landmarks
  python scripts/visualize_detection.py photo.jpg result.jpg \\
      --bbox 290.4 215.4 515.4 526.6 \\
      --landmarks 350.2 320.5 420.8 318.3 385.0 380.2 360.5 450.1 410.2 448.7

  # Multiple faces
  python scripts/visualize_detection.py photo.jpg result.jpg \\
      --bbox 290.4 215.4 515.4 526.6 --score 0.95 \\
      --bbox 100.0 50.0 200.0 180.0 --score 0.87
        """,
    )

    parser.add_argument("input_image", type=str, help="Path to input image")
    parser.add_argument("output_image", type=str, help="Path to save output image")
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        action="append",
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Bounding box coordinates (x1 y1 x2 y2). Can be specified multiple times.",
    )
    parser.add_argument(
        "--landmarks",
        nargs="+",
        type=float,
        action="append",
        help="Landmark coordinates (x1 y1 x2 y2 x3 y3 ...). Can be specified multiple times (one per bbox).",
    )
    parser.add_argument(
        "--score",
        type=float,
        action="append",
        help="Confidence score for the bbox. Can be specified multiple times (one per bbox).",
    )
    parser.add_argument(
        "--color",
        type=str,
        default="0,255,0",
        help="Box color in BGR format (default: 0,255,0 = green)",
    )
    parser.add_argument(
        "--thickness", type=int, default=2, help="Line thickness (default: 2)"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Don't display the image, just save it",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Validate input
    input_path = Path(args.input_image)
    if not input_path.exists():
        print(f"Error: Input image not found: {input_path}")
        sys.exit(1)

    if not args.bbox:
        print("Error: At least one --bbox must be specified")
        print("\nUsage example:")
        print(
            "  python scripts/visualize_detection.py photo.jpg result.jpg --bbox 290.4 215.4 515.4 526.6"
        )
        sys.exit(1)

    # Parse color
    try:
        color_parts = [int(c) for c in args.color.split(",")]
        if len(color_parts) != 3:
            raise ValueError
        box_color = tuple(color_parts)
    except ValueError:
        print(f"Error: Invalid color format: {args.color}")
        print("Use format: B,G,R (e.g., 0,255,0 for green)")
        sys.exit(1)

    # Load image
    print(f"Loading image: {input_path}")
    image = cv2.imread(str(input_path))
    if image is None:
        print(f"Error: Failed to load image: {input_path}")
        sys.exit(1)

    print(f"Image size: {image.shape[1]}x{image.shape[0]}")

    # Draw each bbox
    output = image.copy()
    num_bboxes = len(args.bbox)
    num_landmarks = len(args.landmarks) if args.landmarks else 0
    num_scores = len(args.score) if args.score else 0

    print(f"\nDrawing {num_bboxes} bounding box(es)...")

    for idx, bbox in enumerate(args.bbox):
        # Prepare label
        label_parts = [f"Face {idx + 1}"]
        if args.score and idx < num_scores:
            label_parts.append(f"{args.score[idx]:.3f}")
        label = " - ".join(label_parts) if label_parts else None

        # Draw bbox
        print(
            f"  Box {idx + 1}: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]",
            end="",
        )
        if args.score and idx < num_scores:
            print(f" (score: {args.score[idx]:.3f})", end="")
        print()

        output = draw_bbox(
            output, bbox, cast(tuple[int, int, int], box_color), args.thickness, label
        )

        # Draw landmarks if provided
        if args.landmarks and idx < num_landmarks:
            lm_coords = args.landmarks[idx]
            if len(lm_coords) % 2 != 0:
                print(
                    f"    Warning: Landmarks {idx + 1} has odd number of coordinates, skipping"
                )
                continue

            landmarks = [
                (lm_coords[i], lm_coords[i + 1]) for i in range(0, len(lm_coords), 2)
            ]
            print(f"    Drawing {len(landmarks)} landmarks")
            output = draw_landmarks(output, landmarks)

    # Save output
    output_path = Path(args.output_image)
    cv2.imwrite(str(output_path), output)
    print(f"\nSaved visualization to: {output_path}")

    # Display image if requested
    if not args.no_display:
        try:
            print("Displaying image (press any key to close)...")
            cv2.imshow("Face Detection Visualization", output)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Could not display image (headless environment?): {e}")


if __name__ == "__main__":
    main()
