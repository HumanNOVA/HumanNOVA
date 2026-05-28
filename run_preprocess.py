import argparse
import os

import numpy as np
import rembg
from PIL import Image

from visualenc.base.utils import remove_background, resize_foreground


DEFAULT_RGBA_FOREGROUND_RATIO = 0.8
DEFAULT_OUTPUT_SIZE = 1024
DEFAULT_ALPHA_THRESHOLD = 0.8
DEFAULT_BACKGROUND_VALUE = 0.5


def build_rgba_image(
    image_path,
    rembg_session,
    foreground_ratio=DEFAULT_RGBA_FOREGROUND_RATIO,
    background_value=DEFAULT_BACKGROUND_VALUE,
):
    image = remove_background(Image.open(image_path), rembg_session)
    image = resize_foreground(image, foreground_ratio)

    image_np = np.array(image).astype(np.float32) / 255.0
    rgb = image_np[:, :, :3]
    alpha = image_np[:, :, 3:4]
    premultiplied_rgb = rgb * alpha + (1.0 - alpha) * background_value

    image_rgba = np.concatenate([premultiplied_rgb, alpha], axis=-1)
    image_rgba = (image_rgba * 255.0).astype(np.uint8)
    return Image.fromarray(image_rgba, mode="RGBA")


def rgba_to_rgb_image(
    rgba_image,
    alpha_threshold=DEFAULT_ALPHA_THRESHOLD,
    background_value=DEFAULT_BACKGROUND_VALUE,
):
    image_np = np.array(rgba_image).astype(np.float32) / 255.0
    rgb = image_np[:, :, :3]
    alpha = image_np[:, :, 3:4]
    foreground_mask = alpha > alpha_threshold
    image_rgb = rgb * foreground_mask + (1.0 - foreground_mask) * background_value
    image_rgb = (image_rgb * 255.0).astype(np.uint8)
    return Image.fromarray(image_rgb, mode="RGB")


def process_image(
    image_path,
    rembg_session,
    rgba_output_path=None,
    rgb_output_path=None,
    foreground_ratio=DEFAULT_RGBA_FOREGROUND_RATIO,
    output_size=DEFAULT_OUTPUT_SIZE,
    alpha_threshold=DEFAULT_ALPHA_THRESHOLD,
    background_value=DEFAULT_BACKGROUND_VALUE,
):
    rgba_image = build_rgba_image(
        image_path,
        rembg_session,
        foreground_ratio=foreground_ratio,
        background_value=background_value,
    ).resize((output_size, output_size))

    if rgba_output_path is not None:
        rgba_image.save(rgba_output_path)

    rgb_image = rgba_to_rgb_image(
        rgba_image,
        alpha_threshold=alpha_threshold,
        background_value=background_value,
    ).resize((output_size, output_size))

    if rgb_output_path is not None:
        rgb_image.save(rgb_output_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate RGBA images and feed them directly into the legacy RGB preprocessing step."
    )
    parser.add_argument("input_dir")
    parser.add_argument("rgba_output_dir")
    parser.add_argument("rgb_output_dir")
    parser.add_argument("--foreground-ratio", type=float, default=DEFAULT_RGBA_FOREGROUND_RATIO)
    parser.add_argument("--output-size", type=int, default=DEFAULT_OUTPUT_SIZE)
    parser.add_argument("--alpha-threshold", type=float, default=DEFAULT_ALPHA_THRESHOLD)
    parser.add_argument("--background-value", type=float, default=DEFAULT_BACKGROUND_VALUE)
    parser.add_argument(
        "--skip-rgba-save",
        action="store_true",
        help="Run the RGBA stage in memory and only save the final RGB output.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rembg_session = rembg.new_session()

    os.makedirs(args.rgb_output_dir, exist_ok=True)
    if not args.skip_rgba_save:
        os.makedirs(args.rgba_output_dir, exist_ok=True)

    for img_name in sorted(os.listdir(args.input_dir)):
        image_path = os.path.join(args.input_dir, img_name)
        if not os.path.isfile(image_path):
            continue

        stem, _ = os.path.splitext(img_name)
        rgba_output_path = None
        if not args.skip_rgba_save:
            rgba_output_path = os.path.join(args.rgba_output_dir, stem + ".png")
        rgb_output_path = os.path.join(args.rgb_output_dir, stem + ".png")

        process_image(
            image_path,
            rembg_session,
            rgba_output_path=rgba_output_path,
            rgb_output_path=rgb_output_path,
            foreground_ratio=args.foreground_ratio,
            output_size=args.output_size,
            alpha_threshold=args.alpha_threshold,
            background_value=args.background_value,
        )


if __name__ == "__main__":
    main()
