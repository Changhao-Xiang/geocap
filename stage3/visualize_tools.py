import os

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from common.args import feat_recog_args
from stage3.initial_chamber import ProloculusDetector
from stage3.utils import resize_img
from stage3.volution_counter import VolutionCounter


def visualize_volutions(
    img_paths,
    output_dir,
    feat_recog_args,
    show_initial_chamber=True,
    show_volution_lines=True,
    show_volution_numbers=True,
    save_format="png",
    dpi=300,
):
    """
    Visualize volutions and initial chamber for a batch of images.

    Parameters:
    img_paths (list): List of paths to input images
    output_dir (str): Directory to save visualization results
    feat_recog_args: Arguments for VolutionCounter
    show_initial_chamber (bool): Whether to visualize the initial chamber
    show_volution_lines (bool): Whether to visualize volution lines
    show_volution_numbers (bool): Whether to show volution numbers
    save_format (str): Format to save images ('png', 'jpg', etc.)
    dpi (int): DPI for saved images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize VolutionCounter
    counter = VolutionCounter(feat_recog_args)
    # Visualize initial chamber
    # proloculus_detector = ProloculusDetector(center_prior_jsonl="dataset/annotated-openai-gpt-5.2-25.jsonl")
    proloculus_detector = ProloculusDetector()
    for img_path in img_paths:
        # Get filename without extension
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        # Read and preprocess image
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        orig_h, orig_w = img.shape[:2]
        orig_img_rgb = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
        img_rgb = resize_img(orig_img_rgb)
        h, w = img_rgb.shape[:2]
        orig_img_gray = cv2.cvtColor(orig_img_rgb, cv2.COLOR_RGB2GRAY)

        # Detect initial chamber
        initial_chamber = proloculus_detector.detect_initial_chamber(img_path)

        if show_initial_chamber and hasattr(proloculus_detector, "img_center_block"):
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(proloculus_detector.img_center_block, cmap="gray")
            if initial_chamber is not None:
                x, y, r = initial_chamber
                block_x = x - proloculus_detector.block_origin_x
                block_y = y - proloculus_detector.block_origin_y
                circle = Circle((block_x, block_y), 0.5 * r, fill=False, edgecolor="red", linewidth=2)
                ax.add_patch(circle)
            ax.axis("off")

            block_output_path = os.path.join(output_dir, f"{img_name}_block.{save_format}")
            plt.tight_layout()
            plt.savefig(block_output_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)

        volutions_dict = {}
        if show_volution_lines:
            # Count volutions
            if initial_chamber is None:
                center = (w // 2, h // 2)
            else:
                center = tuple(initial_chamber[:2])

            volutions_dict, _ = counter.count_volutions(img_path, center=center)

            for idx, volution in volutions_dict.items():
                for i, point in enumerate(volution):
                    volution[i] = (int(point[0] * orig_w), int(point[1] * orig_h))
                # Remove duplicate x values by keeping only one point per x coordinate
                unique_x_points = {}
                for point in volution:
                    x = point[0]
                    if x not in unique_x_points:
                        unique_x_points[x] = point
                volution = list(unique_x_points.values())
                volutions_dict[idx] = volution

        # Save original image without any markings
        # original_output_path = os.path.join(output_dir, f"{img_name}_original.{save_format}")
        # plt.figure(figsize=(12, 8))
        # plt.imshow(orig_img_rgb)
        # plt.axis("off")
        # plt.tight_layout()
        # plt.savefig(original_output_path, dpi=dpi, bbox_inches="tight")
        # plt.close()

        # Get binarized image from counter
        binary_img = cv2.adaptiveThreshold(
            orig_img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 2
        )
        binary_img_rgb = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2RGB)

        def save_visualization(base_img_rgb, suffix, cmap=None):
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.imshow(base_img_rgb, cmap=cmap)

            if show_volution_lines:
                for _, points in volutions_dict.items():
                    x_points = [p[0] for p in points]
                    y_points = [p[1] for p in points]

                    ax.plot(x_points, y_points, "-", color="red", linewidth=2)

            if show_initial_chamber and initial_chamber is not None:
                x, y, r = initial_chamber
                circle = Circle((x, y), 0.5 * r, fill=False, edgecolor="red", linewidth=2)
                ax.add_patch(circle)

            ax.axis("off")

            output_path = os.path.join(output_dir, f"{img_name}_{suffix}.{save_format}")
            plt.tight_layout()
            plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)

        save_visualization(orig_img_rgb, "original")
        save_visualization(binary_img_rgb, "binary", cmap="gray")


def batch_visualize(
    input_dir,
    output_dir,
    feat_recog_args,
    file_extensions=(".jpg", ".jpeg", ".png", ".tif", ".tiff"),
    **kwargs,
):
    """
    Process all images in a directory and visualize volutions.

    Parameters:
    input_dir (str): Directory containing input images
    output_dir (str): Directory to save visualization results
    feat_recog_args: Arguments for VolutionCounter
    file_extensions (tuple): File extensions to process
    **kwargs: Additional arguments for visualize_volutions
    """
    # Get all image files in the input directory
    img_paths = []
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(file_extensions):
            img_paths.append(os.path.join(input_dir, filename))

    if not img_paths:
        print(f"No images found in {input_dir} with extensions {file_extensions}")
        return

    print(f"Found {len(img_paths)} images to process")
    visualize_volutions(img_paths, output_dir, feat_recog_args, **kwargs)
    print(f"Visualization complete. Results saved to {output_dir}")


def main():
    input_dir = "dataset/vis"  # 改成你的图片目录
    output_dir = "dataset/vis_initial_chamber"
    batch_visualize(
        input_dir,
        output_dir,
        feat_recog_args,
        show_initial_chamber=True,
        show_volution_lines=False,
    )

if __name__ == "__main__":
    main()
