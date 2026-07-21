import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common.args import feat_recog_args


class ProloculusDetector:
    def __init__(self, block_num: int = 3, center_prior_jsonl: str | None = None):
        self.block_num = block_num
        self.center_prior_dict = self._load_center_priors(center_prior_jsonl)

    def _load_center_priors(self, center_prior_jsonl: str | None) -> dict[str, tuple[int, int]]:
        if center_prior_jsonl is None:
            return {}

        center_prior_dict = {}
        with open(center_prior_jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parsed = json.loads(line)
                for image_name, center in parsed.items():
                    if isinstance(center, dict) and "x" in center and "y" in center:
                        center_prior_dict[str(image_name)] = (int(center["x"]), int(center["y"]))

        return center_prior_dict

    @staticmethod
    def get_score_kernel(window_size: int) -> np.ndarray:
        """Build the same circular reward kernel as the scalar sliding-window implementation."""
        coordinates = np.arange(window_size, dtype=np.float64)
        rows, columns = np.meshgrid(coordinates, coordinates, indexing="ij")
        center = (window_size - 1) / 2
        distance = np.hypot(rows - center, columns - center) / window_size
        outer_radius = (window_size // 2) / window_size
        inner_radius = (window_size // 2) * feat_recog_args.inner_radius_ratio / window_size

        positive_mask = distance <= inner_radius
        negative_mask = (distance > inner_radius) & (distance <= outer_radius)
        positive_weights = np.zeros_like(distance)
        negative_weights = np.zeros_like(distance)
        positive_weights[positive_mask] = 0.5 * (1 + np.exp(-10 * distance[positive_mask]))
        negative_weights[negative_mask] = -0.5 * np.exp(-10 * (distance[negative_mask] - inner_radius))

        total_positive_weight = float(np.sum(positive_weights))
        total_negative_weight = float(abs(np.sum(negative_weights)))
        with np.errstate(divide="ignore", invalid="ignore"):
            return positive_weights / total_positive_weight + negative_weights / (3 * total_negative_weight)

    def find_center(self, window_size: int, threshold: float = 0.25, max_candidates: int | None = None):
        """Find proloculus candidates using vectorized, valid-window correlation."""
        img_array = self.img_center_block.astype(np.float64) / 255
        valid_height = img_array.shape[0] - window_size + 1
        valid_width = img_array.shape[1] - window_size + 1
        if valid_height <= 0 or valid_width <= 0:
            return []

        kernel = self.get_score_kernel(window_size)
        anchor = window_size // 2
        response = cv2.filter2D(
            img_array, cv2.CV_64F, kernel, anchor=(anchor, anchor), borderType=cv2.BORDER_CONSTANT
        )
        score_map = response[anchor : anchor + valid_height, anchor : anchor + valid_width]
        row_indices, column_indices = np.nonzero(score_map > threshold)
        if len(row_indices) == 0:
            return []

        scores = score_map[row_indices, column_indices]
        center_x = column_indices + anchor
        center_y = row_indices + anchor
        abs_center_x = center_x + self.block_origin_x
        abs_center_y = center_y + self.block_origin_y
        ref_x = (self.width / 2 + self.prior_center[0]) / 2
        ref_y = (self.height / 2 + self.prior_center[1]) / 2
        distances = np.hypot(abs_center_x - ref_x, abs_center_y - ref_y) / self.block_width
        rankings = scores - distances

        if max_candidates == 1:
            order = np.array([int(np.argmax(rankings))])
        elif max_candidates is not None and len(rankings) > max_candidates:
            order = np.argpartition(-rankings, max_candidates - 1)[:max_candidates]
            order = order[np.argsort(-rankings[order], kind="stable")]
        else:
            order = np.argsort(-rankings, kind="stable")

        return [
            (int(center_x[index]), int(center_y[index]), float(scores[index]), float(distances[index]))
            for index in order
        ]

    def detect_initial_chamber(
        self,
        image_path_to_detect: str,
        threshold: float = 0.25,
        visualize_result: bool = False,
        rough_center: tuple[int, int] | None = None,
    ):
        self.img = cv2.imread(image_path_to_detect)
        self.width, self.height = self.img.shape[1], self.img.shape[0]
        # Convert to grayscale and extract center block
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        image_name = os.path.basename(image_path_to_detect)
        if rough_center is None and image_name in self.center_prior_dict:
            rough_center = self.center_prior_dict[image_name]
        self.prior_center = rough_center if rough_center is not None else (self.width // 2, self.height // 2)
        self.img_center_block = self.get_search_block(rough_center)

        # Calculate score with different window size
        points_with_max_score = []
        block_short_edge = min(self.img_center_block.shape[:2])
        min_size = max(3, int(block_short_edge * 0.08))
        max_size = block_short_edge
        for size in range(min_size, max_size + 1, 2):
            candidate_centers = self.find_center(
                window_size=size, threshold=threshold, max_candidates=3 if visualize_result else 1
            )
            if len(candidate_centers) > 0:
                points_with_max_score.append(
                    {
                        "size": size,
                        "points": [candidate[:2] for candidate in candidate_centers[:]],
                        "score": [candidate[2] for candidate in candidate_centers[:]],
                        "distance": [candidate[3] for candidate in candidate_centers[:]],
                    }
                )

        sizes = [
            point_with_max_score["size"] * feat_recog_args.inner_radius_ratio
            for point_with_max_score in points_with_max_score
        ]
        scores = [
            point_with_max_score["score"][0] - point_with_max_score["distance"][0]
            for point_with_max_score in points_with_max_score
        ]
        if not scores:
            return None

        if visualize_result:
            visualize(points_with_max_score, image_path_to_detect)

        # Find the point with the highest score
        max_score_index = scores.index(max(scores))
        max_score_point = points_with_max_score[max_score_index]["points"][0]
        diameter = sizes[max_score_index]

        x = max_score_point[0] + self.block_origin_x
        y = max_score_point[1] + self.block_origin_y
        return [x, y, diameter]

    def get_search_block(self, rough_center: tuple[int, int] | None = None):
        self.block_height = max(1, self.height // self.block_num)
        self.block_width = max(1, self.width // self.block_num)

        if rough_center is None:
            center_x, center_y = self.width // 2, self.height // 2
        else:
            center_x = int(np.clip(rough_center[0], 0, self.width - 1))
            center_y = int(np.clip(rough_center[1], 0, self.height - 1))

        max_x_start = max(self.width - self.block_width, 0)
        max_y_start = max(self.height - self.block_height, 0)
        x_start = int(np.clip(center_x - self.block_width // 2, 0, max_x_start))
        y_start = int(np.clip(center_y - self.block_height // 2, 0, max_y_start))
        x_end = x_start + self.block_width
        y_end = y_start + self.block_height

        self.block_origin_x = x_start
        self.block_origin_y = y_start

        center_block = self.img[y_start:y_end, x_start:x_end]
        return center_block


def visualize(points_with_max_score, image_path):
    img_name = image_path.split("/")[-1]
    block_image_path = f"dataset/common/center_block/{img_name}"
    # visualize points with max score with size as x-axis and score as y-axis
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sizes = [
        point_with_max_score["size"] * feat_recog_args.inner_radius_ratio
        for point_with_max_score in points_with_max_score
    ]
    scores = [
        point_with_max_score["score"][0] - point_with_max_score["distance"][0]
        for point_with_max_score in points_with_max_score
    ]
    # Find the point with the highest score
    max_score_index = scores.index(max(scores))
    max_score_point = points_with_max_score[max_score_index]["points"][0]
    max_score_size = sizes[max_score_index]

    plt.scatter(sizes, scores)
    # Add labels to the scatter plot showing the coordinates of each point
    for i, point_with_max_score in enumerate(points_with_max_score):
        point = point_with_max_score["points"][0]
        plt.annotate(
            f"({point[0]}, {point[1]})",
            (sizes[i], scores[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )
    # Add labels to the axes
    plt.xlabel("Index (corresponds to size)")
    plt.ylabel("Score")
    plt.title("Points with Maximum Score")

    # visualize candidate centers
    plt.subplot(1, 2, 2)
    img = cv2.imread(block_image_path)
    for points_dict in points_with_max_score:
        for point in points_dict["points"][:3]:
            cv2.circle(img, (point[0], point[1]), 1, (0, 0, 255), -1)
    cv2.circle(img, (max_score_point[0], max_score_point[1]), 1, (0, 255, 0), -1)
    cv2.circle(img, (max_score_point[0], max_score_point[1]), int(max_score_size / 2), (0, 255, 0), 1)
    plt.imshow(img)

    plt.savefig(f"visualize_tools/testset_min16/{img_name}")
    plt.close()
