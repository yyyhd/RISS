import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create aligned LE|RE images for RISS."
    )
    parser.add_argument(
        "--le_dir",
        required=True,
        help="Directory containing low-energy PNG images.",
    )
    parser.add_argument(
        "--re_dir",
        required=True,
        help="Directory containing real recombined PNG images.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory for saving aligned PNG images.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Output width before LE|RE concatenation.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Output height before LE|RE concatenation.",
    )
    parser.add_argument(
        "--skip_otsu_crop",
        action="store_true",
        help="Skip Otsu breast-region cropping for preprocessed images.",
    )
    return parser.parse_args()


def load_grayscale(path):
    """Load PNG while retaining the original bit depth."""
    image = np.asarray(Image.open(path))

    if image.ndim == 3:
        image = image[..., :3].astype(np.float32)
        image = (
            0.299 * image[..., 0]
            + 0.587 * image[..., 1]
            + 0.114 * image[..., 2]
        )

    if image.ndim != 2:
        raise ValueError(f"Unsupported image shape {image.shape}: {path}")

    return image.astype(np.float32)


def normalize_uint8(image):
    """Scale image intensities to 0–255."""
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    image = np.clip(image, 0, None)

    maximum = float(image.max())
    if maximum <= 0:
        return np.zeros(image.shape, dtype=np.uint8)

    image = image / maximum * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def obtain_breast_bbox(le_image):
    """Find the largest foreground region using Otsu thresholding."""
    _, mask = cv2.threshold(
        le_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    number, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    if number <= 1:
        return 0, 0, le_image.shape[1], le_image.shape[0]

    # Ignore label 0 because it represents the background.
    largest_label = 1 + np.argmax(
        stats[1:, cv2.CC_STAT_AREA]
    )

    x = stats[largest_label, cv2.CC_STAT_LEFT]
    y = stats[largest_label, cv2.CC_STAT_TOP]
    width = stats[largest_label, cv2.CC_STAT_WIDTH]
    height = stats[largest_label, cv2.CC_STAT_HEIGHT]

    return x, y, x + width, y + height


def resize_image(image, width, height):
    return cv2.resize(
        image,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )


def collect_png_files(folder):
    files = {
        path.stem: path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    }

    if not files:
        raise FileNotFoundError(f"No PNG images found in: {folder}")

    return files


def main():
    args = parse_args()

    le_dir = Path(args.le_dir)
    re_dir = Path(args.re_dir)
    output_dir = Path(args.output_dir)

    if not le_dir.is_dir():
        raise NotADirectoryError(f"LE directory does not exist: {le_dir}")
    if not re_dir.is_dir():
        raise NotADirectoryError(f"RE directory does not exist: {re_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    le_files = collect_png_files(le_dir)
    re_files = collect_png_files(re_dir)

    missing_re = sorted(set(le_files) - set(re_files))
    missing_le = sorted(set(re_files) - set(le_files))

    if missing_re or missing_le:
        raise ValueError(
            "LE and RE filenames do not match.\n"
            f"Missing RE images: {missing_re[:10]}\n"
            f"Missing LE images: {missing_le[:10]}"
        )

    patient_ids = sorted(le_files)
    total = len(patient_ids)

    for index, patient_id in enumerate(patient_ids, start=1):
        le_image = normalize_uint8(
            load_grayscale(le_files[patient_id])
        )
        re_image = normalize_uint8(
            load_grayscale(re_files[patient_id])
        )

        if le_image.shape != re_image.shape:
            raise ValueError(
                f"Image-size mismatch for {patient_id}: "
                f"LE={le_image.shape}, RE={re_image.shape}"
            )

        if not args.skip_otsu_crop:
            x1, y1, x2, y2 = obtain_breast_bbox(le_image)
            le_image = le_image[y1:y2, x1:x2]
            re_image = re_image[y1:y2, x1:x2]

        le_image = resize_image(
            le_image, args.width, args.height
        )
        re_image = resize_image(
            re_image, args.width, args.height
        )

        zero_channel = np.zeros_like(le_image)

        # Input A: LE image in the red channel.
        input_a = np.stack(
            [le_image, zero_channel, zero_channel], axis=-1
        )

        # Target B: real RE image in the green channel.
        target_b = np.stack(
            [zero_channel, re_image, zero_channel], axis=-1
        )

        # Pix2pix/ResViT aligned format: A|B.
        aligned_image = np.concatenate(
            [input_a, target_b], axis=1
        )

        output_path = output_dir / f"{patient_id}.png"
        Image.fromarray(aligned_image).save(output_path)

        print(f"[{index}/{total}] Saved: {output_path.name}")

    print(f"Completed. {total} image pairs saved to {output_dir}")


if __name__ == "__main__":
    main()
