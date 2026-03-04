import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common.args import feat_recog_args
from stage3.utils import circle_weight_array


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

    def find_center(self, window_size: int, threshold: float = 0.25):
        # Find center of proloculus using sliding window
        img_array = self.img_center_block
        windows = np.lib.stride_tricks.sliding_window_view(img_array, (window_size, window_size))
        windows = windows.reshape(-1, window_size, window_size)

        pos_weight_array, neg_weight_array, total_positive_weight, total_negative_weight = (
            circle_weight_array(window_size)
        )
        candidate_centers = []
        for i, window in enumerate(windows):
            # Calculate the score of the window
            pos_score = np.sum(window / 255 * pos_weight_array) / total_positive_weight
            neg_score = np.sum(window / 255 * neg_weight_array) / (3 * total_negative_weight)
            score = pos_score + neg_score

            if score > threshold:
                # Calculate the row and column indices from the flattened index
                row_idx = i // (img_array.shape[1] - window_size + 1)
                col_idx = i % (img_array.shape[1] - window_size + 1)

                # Calculate the center coordinates of the window
                center_y = row_idx + window_size // 2
                center_x = col_idx + window_size // 2

                # Calculate the distance to the center of the image
                distance = np.sqrt(
                    (center_x - self.block_width / 2) ** 2 + (center_y - self.block_height / 2) ** 2
                )

                # Add the center coordinates to candidate_centers
                candidate_centers.append((center_x, center_y, score, distance / self.block_width))

        candidate_centers = sorted(candidate_centers, key=lambda x: x[2] - x[3], reverse=True)

        return candidate_centers

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
        self.img_center_block = self.get_search_block(rough_center)

        # Calculate score with different window size
        points_with_max_score = []
        min_size, max_size = 16, 100
        for size in range(min_size, min(self.block_height, max_size) + 1, 2):
            candidate_centers = self.find_center(window_size=size, threshold=threshold)
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
        self.block_height = self.height // self.block_num
        self.block_width = self.width // self.block_num

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
