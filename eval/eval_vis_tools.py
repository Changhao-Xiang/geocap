import argparse
import json
import os
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from common.args import logger
from stage3.get_angles_and_slope import get_angles_and_slope
from stage3.recognize import recognize_feature
from stage3.utils import get_circle_points


class VisToolOutputGenerator:
    def __init__(self, image_dir: str) -> None:
        self.image_path_root = image_dir
        self.all_images = os.listdir(self.image_path_root)
        self.loaded_llm = False

    def extract_valid_images(self, data_dict: dict):
        # Extract info of Axial image of Holotype specimen
        self.images = []
        for info in data_dict.values():
            images = info["images"]
            for image_dict in images:
                section_type = image_dict["section_type"]
                pixel_mm = image_dict["pixel/mm"]
                if (
                    "axial" in section_type.lower()
                    # and "holotype" in specimen_type.lower()
                    and pixel_mm > 0
                    and image_dict["image"] in self.all_images
                ):
                    image = image_dict["image"]
                    self.images.append({"image": image, "pixel_mm": pixel_mm, "desc": info["desc"]})

    def recognize_features(self, image_info: dict) -> dict[str, Any]:
        img_path = os.path.join(self.image_path_root, image_info["image"])
        self.img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        h, w = self.img.shape[:2]

        new_image_info = image_info
        new_image_info["img_height"] = h
        new_image_info["img_width"] = w

        # Overall shape
        img_countour = self.img[:, :, 3]
        contours, _ = cv2.findContours(img_countour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = max(contours, key=cv2.contourArea)

        # Get bounding box coordinates
        x_min = min(point[0][0] for point in contour)
        y_min = min(point[0][1] for point in contour)
        x_max = max(point[0][0] for point in contour)
        y_max = max(point[0][1] for point in contour)

        # Calculate length and width/diameter (in mm. ) and ratio
        new_image_info["length"] = (x_max - x_min) * image_info["pixel_mm"]
        new_image_info["width"] = (y_max - y_min) * image_info["pixel_mm"]
        if new_image_info["width"] == 0:
            print(image_info)
        new_image_info["ratio"] = new_image_info["length"] / new_image_info["width"]

        # Classify size by length
        if new_image_info["length"] < 1:
            new_image_info["size"] = "minute"
        elif new_image_info["length"] < 3:
            new_image_info["size"] = "small"
        elif new_image_info["length"] < 6:
            new_image_info["size"] = "medium"
        elif new_image_info["length"] < 10:
            new_image_info["size"] = "large"
        elif new_image_info["length"] < 20:
            new_image_info["size"] = "mega"
        else:
            new_image_info["size"] = "gigantic"

        # Get equator, slope and poles
        angles_and_scores = get_angles_and_slope(img_path)
        left_angle, right_angle, upper_angle, lower_angle = angles_and_scores[:4]
        equator_angle = (upper_angle + lower_angle) / 2
        if equator_angle < 2.95:
            equator = "convex"
        elif equator_angle < 3.15:
            equator = "straight"
        else:
            equator = "concave"
        new_image_info["equator"] = equator

        poles_angle = (left_angle + right_angle) / 2
        if poles_angle < 2.35:
            poles = "pointed"
        else:
            poles = "blunted"
        new_image_info["poles"] = poles

        slopes_score = np.sum(angles_and_scores[4:])
        if slopes_score < -2.3:
            slopes = "convex"
        elif slopes_score < -0.8:
            slopes = "straight"
        else:
            slopes = "concave"
        new_image_info["lateral_slopes"] = slopes

        shape_type = "ellipsoidal" if slopes == "convex" else "fusiform"
        ellipsoidal_classes = {
            "prolate spherical": [0.9, 0.98],
            "spherical": [0.98, 1.05],
            "sub-spherical": [1.05, 1.11],
            "ellipsoidal": [1.11, 3],
            "elongate ellipsoidal": [3, 6],
            "cylindrical": [6, 999],
        }
        fusiform_classes = {
            "lentoid": [0, 0.75],
            "rhombus": [0.75, 1.3],
            "inflated fusiform": [1.3, 1.9],
            "fusiform": [1.9, 3.5],
            "elongate fusiform": [3.5, 999],
        }
        ratio2shape = ellipsoidal_classes if shape_type == "ellipsoidal" else fusiform_classes
        for shape, ratio_range in ratio2shape.items():
            if ratio_range[0] < new_image_info["ratio"] <= ratio_range[1]:
                new_image_info["shape"] = shape
                break

        # Recognize fossil features
        volutions_dict, thickness_dict, initial_chamber, tunnel_angles = recognize_feature(img_path)

        # Process numerical info
        num_positive_keys = len([k for k in volutions_dict.keys() if k > 0])
        num_negative_keys = len([k for k in volutions_dict.keys() if k < 0])
        larger_key, smaller_key = max(num_positive_keys, num_negative_keys), min(
            num_positive_keys, num_negative_keys
        )
        if larger_key > smaller_key + 1:
            new_image_info["num_volutions"] = larger_key
            if num_positive_keys > num_negative_keys:
                volutions_dict = {k: v for k, v in volutions_dict.items() if k > 0}
            else:
                volutions_dict = {k: v for k, v in volutions_dict.items() if k < 0}
        else:
            new_image_info["num_volutions"] = (num_positive_keys + num_negative_keys) / 2

        # Calculate average height between adjacent volutions
        volution_heights = {}
        for idx, points in volutions_dict.items():
            if idx > 0 and idx - 1 in volutions_dict:
                next_points = volutions_dict[idx - 1]
            elif idx < 0 and idx + 1 in volutions_dict:
                next_points = volutions_dict[idx + 1]
            elif idx == 1 and initial_chamber is not None:
                initial_chamber_upper = get_circle_points(
                    center=initial_chamber[:2], radius=initial_chamber[2] // 2, angle_range=[225, 315]
                )
                next_points = initial_chamber_upper
            elif idx == -1 and initial_chamber is not None:
                initial_chamber_lower = get_circle_points(
                    center=initial_chamber[:2], radius=initial_chamber[2] // 2, angle_range=[45, 135]
                )
                next_points = initial_chamber_lower
            else:
                continue
            y_mean = np.mean([point[1] for point in points])
            next_y_mean = np.mean([point[1] for point in next_points])
            if abs(idx) not in volution_heights:
                volution_heights[abs(idx)] = abs(y_mean - next_y_mean)
            else:
                volution_heights[abs(idx)] = (volution_heights[abs(idx)] + abs(y_mean - next_y_mean)) / 2
        # Sort volution_heights by key in ascending order
        volution_heights = dict(sorted(volution_heights.items(), key=lambda item: item[0]))
        new_image_info["volution_heights"] = volution_heights

        # Calculate average thickness and thickness per volutions
        avg_thickness = np.mean([thickness for thickness in thickness_dict.values()])
        new_image_info["avg_thickness"] = avg_thickness * h
        thickness_per_vol = {}
        for idx, thickness in thickness_dict.items():
            if abs(idx) not in thickness_per_vol:
                thickness_per_vol[abs(idx)] = thickness * h
            else:
                thickness_per_vol[abs(idx)] = (thickness_per_vol[abs(idx)] + thickness * h) / 2
        thickness_per_vol = dict(sorted(thickness_per_vol.items(), key=lambda item: item[0]))
        new_image_info["thickness_per_vol"] = thickness_per_vol

        if initial_chamber is not None:
            # Convert to diameter
            new_image_info["size_init_chamber"] = initial_chamber[2] * image_info["pixel_mm"] * 1000

        if tunnel_angles:
            tunnel_angles = self.post_process_tunnel_angles(
                tunnel_angles, int(new_image_info["num_volutions"])
            )
            new_image_info["tunnel_angles"] = tunnel_angles
            new_image_info["tunnel_angle"] = int(sum(tunnel_angles.values()) / len(tunnel_angles))

        return new_image_info

    def post_process_tunnel_angles(
        self, tunnel_angles: dict[int, int], num_volutions: int, low_thres: int = 15, high_thres: int = 55
    ) -> dict[int, int]:
        for i in range(1, num_volutions + 1):
            if i not in tunnel_angles:
                tunnel_angles[i] = 25

        in_range_angles = [angle for angle in tunnel_angles.values() if low_thres < angle < high_thres]
        avg_angles = int(sum(in_range_angles) / len(in_range_angles)) if in_range_angles else 25
        for i, angle in tunnel_angles.items():
            if angle < low_thres or angle > high_thres:
                tunnel_angles[i] = avg_angles

        tunnel_angles = dict(sorted(tunnel_angles.items(), key=lambda x: x[0]))
        min_angle = min(tunnel_angles.values())
        max_angle = max(tunnel_angles.values())
        total_volutions = max(tunnel_angles.keys())

        if max_angle > min_angle and total_volutions > 1:
            avg_increase = (max_angle - min_angle) / (total_volutions - 1)
            base_angle = min(tunnel_angles.values())
            for i in tunnel_angles.keys():
                expected_angle = base_angle + (i - 1) * avg_increase
                tunnel_angles[i] = int(0.5 * tunnel_angles[i] + 0.5 * expected_angle)

        return tunnel_angles


def generate_vis_tools_output(image_info_json: str, image_dir: str, output_path: str):
    # Create a data generator instance
    evaluator = VisToolOutputGenerator(image_dir)

    # Load data from a JSON file containing fossil information
    with open(image_info_json, "r") as f:
        data_dict = json.load(f)

    evaluator.extract_valid_images(data_dict)

    # Process each image and extract required information
    output_info = []
    logger.info("Processing images and extracting information")
    for image_info in tqdm(evaluator.images):
        # Recognize features from the image
        processed_info = evaluator.recognize_features(image_info)

        # Create a dictionary with the required fields
        fossil_info = {
            "image_path": image_info["image"],
            "size": processed_info.get("size", ""),
            "shape": processed_info.get("shape", ""),
            "equator": processed_info.get("equator", ""),
            "lateral_slopes": processed_info.get("lateral_slopes", ""),
            "poles": processed_info.get("poles", ""),
            "length": f"{processed_info['length']:.3f} mm",
            "width": f"{processed_info['width']:.3f} mm",
            "ratio": f"{processed_info['ratio']:.3f}",
            # keep axis_shape as empty string if not explicitly inferred
            "axis_shape": "",
            "number_of_volutions": f"{processed_info.get('num_volutions', '')}",
            "coil_tightness": "",
            "height_of_volution": "",
            "thickness_of_spirotheca": "",
            "septa": "",
            "proloculus": "",
            "tunnel_angles": "",
            "tunnel_shape": "",
            "chomata": "",
            "axial_filling": "",
        }

        # Process thickness of spirotheca
        if "thickness_per_vol" in processed_info:
            avg_thickness_microns = int(processed_info["avg_thickness"] * processed_info["pixel_mm"] * 1000)
            thickness_microns = [
                str(int(thickness * processed_info["pixel_mm"] * 1000))
                for thickness in processed_info["thickness_per_vol"].values()
            ]
            fossil_info["thickness_of_spirotheca"] = (
                f"average: {avg_thickness_microns} microns; by volutions: {', '.join(thickness_microns)} microns"
            )

        # Process heights of volutions
        if "volution_heights" in processed_info:
            heights_microns = [
                str(int(height * processed_info["pixel_mm"] * 1000))
                for height in processed_info["volution_heights"].values()
            ]
            fossil_info["height_of_volution"] = ", ".join(heights_microns) + " microns"

        # Process proloculus (initial chamber)
        if "size_init_chamber" in processed_info:
            fossil_info["proloculus"] = f"{int(processed_info['size_init_chamber'])} microns"

        # Process tunnel angles
        if "tunnel_angles" in processed_info:
            angles = []
            for _, angle in processed_info["tunnel_angles"].items():
                angles.append(str(angle))
            if "tunnel_angle" in processed_info:
                fossil_info["tunnel_angles"] = (
                    f"average: {processed_info['tunnel_angle']} degrees; by volutions: "
                    + ", ".join(angles)
                    + " degrees"
                )
            else:
                fossil_info["tunnel_angles"] = ", ".join(angles) + " degrees"

        output_info.append(fossil_info)

    # Save the extracted information to a JSON file
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_info, f, indent=4)

    logger.info(f"Extracted information saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate visual tools extraction output.")
    parser.add_argument(
        "--image_info_json",
        type=str,
        default="dataset/common/filtered_data.json",
        help="JSON file containing image information.",
    )
    parser.add_argument(
        "--image_dir", type=str, default="dataset/common/filtered_images", help="Directory containing images."
    )
    parser.add_argument(
        "--output_path", type=str, default="./extracted_tool_results.json", help="Output JSON path."
    )
    args = parser.parse_args()

    generate_vis_tools_output(
        image_info_json=args.image_info_json, image_dir=args.image_dir, output_path=args.output_path
    )
    # eval_vis_tools()


if __name__ == "__main__":
    main()
