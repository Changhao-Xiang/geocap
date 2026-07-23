from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import os
import sys
from typing import Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
from tqdm import tqdm


CSV_FIELDS = [
    "image",
    "length",
    "width",
    "ratio",
    "size",
    "shape",
    "left_angle_deg",
    "right_angle_deg",
    "upper_angle_deg",
    "lower_angle_deg",
    "equator_angle_deg",
    "equator",
    "poles_angle_deg",
    "poles",
    "slopes_score_lu",
    "slopes_score_ru",
    "slopes_score_ld",
    "slopes_score_rd",
    "slopes_score",
    "slopes",
    "num_volutions",
    "heights_micron",
    "avg_thickness",
    "thickness_micron",
    "initial_chamber_x",
    "initial_chamber_y",
    "initial_chamber_radius",
    "size_init_chamber",
    "tunnel_angle",
]


_WORKER_RECOGNIZER: CommonFeatureRecognizer | None = None


class CommonFeatureRecognizer:
    def __init__(
        self,
        image_dir: str,
        center_prior_jsonl: str | None = None,
    ) -> None:
        from stage3.get_angles_and_slope import get_angles_and_slope
        from stage3.recognize import recognize_feature
        from stage3.utils import get_circle_points

        self.image_dir = image_dir
        self.center_prior_jsonl = center_prior_jsonl
        self.get_angles_and_slope = get_angles_and_slope
        self.recognize_feature = recognize_feature
        self.get_circle_points = get_circle_points

    def recognize_features(self, image_info: dict[str, Any]) -> dict[str, Any]:
        image_name = str(image_info["image"])
        img_path = os.path.join(self.image_dir, image_name)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        if img.ndim != 3 or img.shape[2] < 4:
            raise ValueError(f"Image must have an alpha channel for contour detection: {img_path}")

        h, w = img.shape[:2]
        processed_info = dict(image_info)
        processed_info["img_height"] = h
        processed_info["img_width"] = w

        contour = self._get_largest_contour(img)
        x_min = min(point[0][0] for point in contour)
        y_min = min(point[0][1] for point in contour)
        x_max = max(point[0][0] for point in contour)
        y_max = max(point[0][1] for point in contour)

        processed_info["length"] = (x_max - x_min) * image_info["pixel_mm"]
        processed_info["width"] = (y_max - y_min) * image_info["pixel_mm"]
        if processed_info["width"] == 0:
            raise ValueError(f"Detected zero fossil width: {image_name}")
        processed_info["ratio"] = processed_info["length"] / processed_info["width"]
        processed_info["size"] = self._classify_size(processed_info["length"])

        angles_and_scores = self.get_angles_and_slope(img_path)
        left_angle, right_angle, upper_angle, lower_angle = angles_and_scores[:4]
        slope_scores = [float(score) for score in angles_and_scores[4:]]

        equator_angle = (upper_angle + lower_angle) / 2
        poles_angle = (left_angle + right_angle) / 2
        slopes_score = float(np.sum(slope_scores))
        slopes = self._classify_slopes(slopes_score)

        processed_info.update(
            {
                "left_angle_deg": float(np.degrees(left_angle)),
                "right_angle_deg": float(np.degrees(right_angle)),
                "upper_angle_deg": float(np.degrees(upper_angle)),
                "lower_angle_deg": float(np.degrees(lower_angle)),
                "equator_angle_deg": float(np.degrees(equator_angle)),
                "equator": self._classify_equator(equator_angle),
                "poles_angle_deg": float(np.degrees(poles_angle)),
                "poles": self._classify_poles(poles_angle),
                "slopes_score_lu": slope_scores[0],
                "slopes_score_ru": slope_scores[1],
                "slopes_score_ld": slope_scores[2],
                "slopes_score_rd": slope_scores[3],
                "slopes_score": slopes_score,
                "slopes": slopes,
                "shape": self._classify_shape(processed_info["ratio"], slopes),
            }
        )

        volutions_dict, thickness_dict, initial_chamber, tunnel_angles = self.recognize_feature(
            img_path, center_prior_jsonl=self.center_prior_jsonl
        )
        num_volutions, volutions_dict = self._normalize_volutions(volutions_dict)
        processed_info["num_volutions"] = num_volutions
        processed_info["volution_heights"] = self._calculate_volution_heights(
            volutions_dict, initial_chamber
        )

        avg_thickness, thickness_per_vol = self._calculate_thickness(thickness_dict, h)
        processed_info["avg_thickness"] = avg_thickness
        processed_info["thickness_per_vol"] = thickness_per_vol

        if initial_chamber is not None:
            processed_info["initial_chamber_x"] = initial_chamber[0]
            processed_info["initial_chamber_y"] = initial_chamber[1]
            processed_info["initial_chamber_radius"] = initial_chamber[2]
            processed_info["size_init_chamber"] = initial_chamber[2] * image_info["pixel_mm"] * 1000

        processed_tunnel_angles = self.post_process_tunnel_angles(
            dict(tunnel_angles), int(num_volutions)
        )
        if processed_tunnel_angles:
            processed_info["tunnel_angle"] = int(
                sum(processed_tunnel_angles.values()) / len(processed_tunnel_angles)
            )

        return processed_info

    @staticmethod
    def _get_largest_contour(img: np.ndarray) -> np.ndarray:
        img_contour = img[:, :, 3]
        contours, _ = cv2.findContours(img_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            raise ValueError("No contour found in image alpha channel")
        return max(contours, key=cv2.contourArea)

    @staticmethod
    def _classify_size(length: float) -> str:
        if length < 1:
            return "minute"
        if length < 3:
            return "small"
        if length < 6:
            return "medium"
        if length < 10:
            return "large"
        if length < 20:
            return "mega"
        return "gigantic"

    @staticmethod
    def _classify_equator(equator_angle: float) -> str:
        if equator_angle < 2.95:
            return "convex"
        if equator_angle < 3.15:
            return "straight"
        return "concave"

    @staticmethod
    def _classify_poles(poles_angle: float) -> str:
        if poles_angle < 2.35:
            return "pointed"
        return "blunted"

    @staticmethod
    def _classify_slopes(slopes_score: float) -> str:
        if slopes_score < -2.3:
            return "convex"
        if slopes_score < -0.8:
            return "straight"
        return "concave"

    @staticmethod
    def _classify_shape(ratio: float, slopes: str) -> str:
        shape_type = "ellipsoidal" if slopes == "convex" else "fusiform"
        ellipsoidal_classes = {
            "lentoid": [0, 0.9],
            "prolate spherical": [0.9, 0.98],
            "spherical": [0.98, 1.05],
            "sub-spherical": [1.05, 1.11],
            "ellipsoidal": [1.11, 3],
            "elongate ellipsoidal": [3, 6],
            "cylindrical": [6, float("inf")],
        }
        fusiform_classes = {
            "lentoid": [0, 0.75],
            "rhombus": [0.75, 1.3],
            "inflated fusiform": [1.3, 1.9],
            "fusiform": [1.9, 3.5],
            "elongate fusiform": [3.5, float("inf")],
        }
        ratio2shape = ellipsoidal_classes if shape_type == "ellipsoidal" else fusiform_classes
        for shape, ratio_range in ratio2shape.items():
            if ratio_range[0] < ratio <= ratio_range[1]:
                return shape
        return ""

    @staticmethod
    def _normalize_volutions(
        volutions_dict: dict[int, list[tuple[int, int]]]
    ) -> tuple[float, dict[int, list[tuple[int, int]]]]:
        num_positive_keys = len([k for k in volutions_dict if k > 0])
        num_negative_keys = len([k for k in volutions_dict if k < 0])
        larger_key = max(num_positive_keys, num_negative_keys)
        smaller_key = min(num_positive_keys, num_negative_keys)

        if larger_key > smaller_key + 1:
            if num_positive_keys > num_negative_keys:
                volutions_dict = {k: v for k, v in volutions_dict.items() if k > 0}
            else:
                volutions_dict = {k: v for k, v in volutions_dict.items() if k < 0}
            return float(larger_key), volutions_dict

        return (num_positive_keys + num_negative_keys) / 2, volutions_dict

    def _calculate_volution_heights(
        self,
        volutions_dict: dict[int, list[tuple[int, int]]],
        initial_chamber: tuple[float, float, float] | None,
    ) -> dict[int, float]:
        volution_heights: dict[int, float] = {}
        for idx, points in volutions_dict.items():
            if idx > 0 and idx - 1 in volutions_dict:
                next_points = volutions_dict[idx - 1]
            elif idx < 0 and idx + 1 in volutions_dict:
                next_points = volutions_dict[idx + 1]
            elif idx == 1 and initial_chamber is not None:
                next_points = self.get_circle_points(
                    center=initial_chamber[:2],
                    radius=int(initial_chamber[2]) // 2,
                    angle_range=[225, 315],
                )
            elif idx == -1 and initial_chamber is not None:
                next_points = self.get_circle_points(
                    center=initial_chamber[:2],
                    radius=int(initial_chamber[2]) // 2,
                    angle_range=[45, 135],
                )
            else:
                continue

            y_mean = np.mean([point[1] for point in points])
            next_y_mean = np.mean([point[1] for point in next_points])
            height = abs(y_mean - next_y_mean)
            abs_idx = abs(idx)
            if abs_idx not in volution_heights:
                volution_heights[abs_idx] = height
            else:
                volution_heights[abs_idx] = (volution_heights[abs_idx] + height) / 2

        return dict(sorted(volution_heights.items(), key=lambda item: item[0]))

    @staticmethod
    def _calculate_thickness(
        thickness_dict: dict[int, float], img_height: int
    ) -> tuple[float, dict[int, float]]:
        if not thickness_dict:
            return float("nan"), {}

        avg_thickness = float(np.mean([thickness for thickness in thickness_dict.values()])) * img_height
        thickness_per_vol: dict[int, float] = {}
        for idx, thickness in thickness_dict.items():
            abs_idx = abs(idx)
            thickness_px = thickness * img_height
            if abs_idx not in thickness_per_vol:
                thickness_per_vol[abs_idx] = thickness_px
            else:
                thickness_per_vol[abs_idx] = (thickness_per_vol[abs_idx] + thickness_px) / 2

        return avg_thickness, dict(sorted(thickness_per_vol.items(), key=lambda item: item[0]))

    @staticmethod
    def post_process_tunnel_angles(
        tunnel_angles: dict[int, int],
        num_volutions: int,
        low_thres: int = 15,
        high_thres: int = 55,
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
        if not tunnel_angles:
            return tunnel_angles

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


def init_worker(image_dir: str, center_prior_jsonl: str | None) -> None:
    global _WORKER_RECOGNIZER

    _WORKER_RECOGNIZER = CommonFeatureRecognizer(
        image_dir=image_dir, center_prior_jsonl=center_prior_jsonl
    )


def recognize_one_for_worker(payload: tuple[int, dict[str, Any]]) -> tuple[int, dict[str, Any]]:
    index, image_info = payload
    if _WORKER_RECOGNIZER is None:
        raise RuntimeError("Worker recognizer has not been initialized")

    processed_info = _WORKER_RECOGNIZER.recognize_features(image_info)
    return index, build_csv_row(processed_info)


def build_csv_row(processed_info: dict[str, Any]) -> dict[str, Any]:
    pixel_mm = processed_info["pixel_mm"]
    heights_micron = [
        int(height * pixel_mm * 1000) for height in processed_info["volution_heights"].values()
    ]
    thickness_micron = [
        int(thickness * pixel_mm * 1000) for thickness in processed_info["thickness_per_vol"].values()
    ]

    row: dict[str, Any] = {field: "" for field in CSV_FIELDS}
    row.update(
        {
            "image": processed_info["image"],
            "length": processed_info["length"],
            "width": processed_info["width"],
            "ratio": processed_info["ratio"],
            "size": processed_info["size"],
            "shape": processed_info["shape"],
            "left_angle_deg": processed_info["left_angle_deg"],
            "right_angle_deg": processed_info["right_angle_deg"],
            "upper_angle_deg": processed_info["upper_angle_deg"],
            "lower_angle_deg": processed_info["lower_angle_deg"],
            "equator_angle_deg": processed_info["equator_angle_deg"],
            "equator": processed_info["equator"],
            "poles_angle_deg": processed_info["poles_angle_deg"],
            "poles": processed_info["poles"],
            "slopes_score_lu": processed_info["slopes_score_lu"],
            "slopes_score_ru": processed_info["slopes_score_ru"],
            "slopes_score_ld": processed_info["slopes_score_ld"],
            "slopes_score_rd": processed_info["slopes_score_rd"],
            "slopes_score": processed_info["slopes_score"],
            "slopes": processed_info["slopes"],
            "num_volutions": processed_info["num_volutions"],
            "heights_micron": heights_micron,
            "avg_thickness": processed_info["avg_thickness"],
            "thickness_micron": thickness_micron,
            "tunnel_angle": processed_info.get("tunnel_angle", ""),
        }
    )

    for field in (
        "initial_chamber_x",
        "initial_chamber_y",
        "initial_chamber_radius",
        "size_init_chamber",
    ):
        row[field] = processed_info.get(field, "")

    return row


def extract_valid_images(
    data_dict: dict[str, Any],
    available_images: set[str],
    include_non_axial: bool = False,
) -> list[dict[str, Any]]:
    images: list[dict[str, Any]] = []
    for info in data_dict.values():
        for image_dict in info["images"]:
            section_type = image_dict.get("section_type", "")
            pixel_mm = image_dict.get("pixel/mm", 0)
            image = image_dict.get("image", "")
            if not include_non_axial and "axial" not in section_type.lower():
                continue
            if pixel_mm <= 0 or image not in available_images:
                continue
            images.append({"image": image, "pixel_mm": pixel_mm, "desc": info.get("desc", "")})

    return images


def resolve_image_dir(image_info_json: str, image_dir: str | None) -> str:
    if image_dir is not None:
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
        return image_dir

    data_root = str(os.path.dirname(image_info_json))
    candidates = ["whole_images", "filtered_images", "images"]
    for candidate in candidates:
        candidate_path = os.path.join(data_root, candidate)
        if os.path.isdir(candidate_path):
            return candidate_path

    candidate_text = ", ".join(os.path.join(data_root, candidate) for candidate in candidates)
    raise FileNotFoundError(
        "Cannot find an image directory. Pass --image-dir explicitly, or create one of: "
        f"{candidate_text}"
    )


def generate_common_feature_csv(
    image_info_json: str,
    image_dir: str,
    output_path: str,
    start_pos: int = 0,
    end_pos: int | None = None,
    include_non_axial: bool = False,
    fail_fast: bool = False,
    failures_path: str | None = None,
    center_prior_jsonl: str | None = None,
    num_workers: int = 1,
) -> None:
    from common.args import logger

    with open(image_info_json, "r", encoding="utf-8") as f:
        data_dict = json.load(f)

    available_images = set(os.listdir(image_dir))
    images = extract_valid_images(data_dict, available_images, include_non_axial=include_non_axial)
    selected_images = images[start_pos:end_pos]
    if not selected_images:
        raise ValueError("No images selected for feature recognition")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    num_workers = max(1, num_workers)
    rows: list[dict[str, Any] | None] = [None] * len(selected_images)
    failures: list[dict[str, str]] = []
    if num_workers == 1:
        recognizer = CommonFeatureRecognizer(image_dir=image_dir, center_prior_jsonl=center_prior_jsonl)
        for index, image_info in tqdm(
            enumerate(selected_images), total=len(selected_images), desc="Recognizing features"
        ):
            try:
                processed_info = recognizer.recognize_features(image_info)
                rows[index] = build_csv_row(processed_info)
            except Exception as exc:
                if fail_fast:
                    raise
                logger.exception(f"Failed to recognize features for {image_info['image']}: {exc}")
                failures.append({"image": image_info["image"], "error": repr(exc)})
    else:
        logger.info(f"Recognizing features with {num_workers} worker processes")
        payloads = list(enumerate(selected_images))
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=init_worker,
            initargs=(image_dir, center_prior_jsonl),
        ) as executor:
            future_to_image = {
                executor.submit(recognize_one_for_worker, payload): payload[1] for payload in payloads
            }
            for future in tqdm(
                as_completed(future_to_image),
                total=len(future_to_image),
                desc="Recognizing features",
            ):
                image_info = future_to_image[future]
                try:
                    index, row = future.result()
                    rows[index] = row
                except Exception as exc:
                    if fail_fast:
                        for pending in future_to_image:
                            pending.cancel()
                        raise
                    logger.exception(f"Failed to recognize features for {image_info['image']}: {exc}")
                    failures.append({"image": image_info["image"], "error": repr(exc)})

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            if row is not None:
                writer.writerow(row)

    if failures and failures_path:
        failures_dir = os.path.dirname(failures_path)
        if failures_dir:
            os.makedirs(failures_dir, exist_ok=True)
        with open(failures_path, "w", encoding="utf-8") as f:
            for failure in failures:
                f.write(json.dumps(failure, ensure_ascii=False) + "\n")

    logger.info(
        f"Saved {len(selected_images) - len(failures)} rows to {output_path}; "
        f"{len(failures)} failures"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recognize visual-tool features for dataset/common and export the same CSV schema "
            "as recognized_features.csv."
        )
    )
    parser.add_argument(
        "--image-info-json",
        type=str,
        default=None,
        help="Input metadata JSON. Defaults to <fossil_data_path>/filtered_data.json.",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="dataset/common/image0423",
        help=(
            "Directory containing PNG images. If omitted, the script searches whole_images, "
            "filtered_images, then images beside --image-info-json."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output CSV path. Defaults to <metadata_dir>/recognized_features.csv.",
    )
    parser.add_argument(
        "--start-pos",
        "--start_pos",
        "-s",
        dest="start_pos",
        type=int,
        default=0,
        help="Start index in valid image list.",
    )
    parser.add_argument(
        "--end-pos",
        "--end_pos",
        "-e",
        dest="end_pos",
        type=int,
        default=None,
        help="End index in valid image list.",
    )
    parser.add_argument(
        "--include-non-axial",
        action="store_true",
        help="Process non-axial images too. By default only axial images are used.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately when one image fails instead of logging and continuing.",
    )
    parser.add_argument(
        "--failures-path",
        type=str,
        default=None,
        help="Optional JSONL path for failed image names and errors.",
    )
    parser.add_argument(
        "--center-prior-jsonl",
        type=str,
        default=None,
        help="Optional center prior JSONL passed to stage3.recognize.recognize_feature.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker processes for feature recognition. Use 1 for serial execution.",
    )

    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return args


def main() -> None:
    args = parse_args()

    from common.args import feat_recog_args

    fossil_data_path = str(feat_recog_args.fossil_data_path)
    image_info_json = (
        str(args.image_info_json)
        if args.image_info_json is not None
        else os.path.join(fossil_data_path, "filtered_data.json")
    )
    image_dir_arg = str(args.image_dir) if args.image_dir is not None else None
    image_dir = resolve_image_dir(image_info_json, image_dir_arg)
    output_path = str(args.output_path) if args.output_path is not None else os.path.join(
        os.path.dirname(image_info_json), "recognized_features.csv"
    )
    failures_path = str(args.failures_path) if args.failures_path is not None else None
    if failures_path is None:
        failures_path = f"{os.path.splitext(output_path)[0]}_failures.jsonl"
    center_prior_jsonl = (
        str(args.center_prior_jsonl) if args.center_prior_jsonl is not None else None
    )

    generate_common_feature_csv(
        image_info_json=image_info_json,
        image_dir=image_dir,
        output_path=output_path,
        start_pos=int(args.start_pos),
        end_pos=int(args.end_pos) if args.end_pos is not None else None,
        include_non_axial=bool(args.include_non_axial),
        fail_fast=bool(args.fail_fast),
        failures_path=failures_path,
        center_prior_jsonl=center_prior_jsonl,
        num_workers=int(args.num_workers),
    )


if __name__ == "__main__":
    main()
