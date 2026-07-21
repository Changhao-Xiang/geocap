from typing import Optional

import cv2
import numpy as np

from stage3.utils import bresenham, fit_line, resize_img, split_into_segments


class VolutionCounter:
    def __init__(
        self,
        feat_recog_args,
        width_ratio: float = 0.3,
        adsorption_thres: float = 0.8,
        volution_thres: float = 0.85,
        step: int = 3,
        num_segments: int = 50,
        filter_max_y_ratio: float = 0.01,
        max_adsorption_time: int = 7,
        use_initial_chamber: bool = True,
        use_profile_fallback: bool = True,
        profile_prominence: float = 0.02,
        use_shape_adaptive_params: bool = True,
        low_shell_ratio_threshold: float = 1.0,
    ):
        self.feat_recog_args = feat_recog_args
        self.width_ratio = width_ratio
        self.adsorption_thres = adsorption_thres
        self.volution_thres = volution_thres
        if hasattr(feat_recog_args, "volution_thres"):
            self.volution_thres = feat_recog_args.volution_thres
        self.step = step
        self.num_segments = num_segments
        self.filter_max_y_ratio = filter_max_y_ratio
        self.max_adsorption_time = max_adsorption_time
        self.use_initial_chamber = use_initial_chamber
        self.use_profile_fallback = getattr(feat_recog_args, "use_profile_fallback", use_profile_fallback)
        self.profile_prominence = getattr(feat_recog_args, "profile_prominence", profile_prominence)
        self.use_shape_adaptive_params = getattr(
            feat_recog_args, "use_shape_adaptive_volution_params", use_shape_adaptive_params
        )
        self.low_shell_ratio_threshold = getattr(
            feat_recog_args, "low_shell_ratio_threshold", low_shell_ratio_threshold
        )
        self.configure_shape_parameters(self.low_shell_ratio_threshold)

        self.finish = False  # tag for scanning process
        self.detection_mode = "adsorption"

    def get_shell_geometry(self, img_path: str) -> tuple[float, tuple[int, int, int, int]]:
        """Reuse the project's alpha-contour detector to estimate shell proportions."""
        orig_h, orig_w = self.original_shape
        fallback = (orig_w / max(orig_h, 1), (0, 0, orig_w, orig_h))
        try:
            from stage3.get_angles_and_slope import find_contour

            contour = find_contour(img_path)
            if isinstance(contour, str) or len(contour) < 3:
                return fallback
            points = contour.reshape(-1, 2)
            x, y = np.min(points, axis=0).astype(int)
            max_x, max_y = np.max(points, axis=0).astype(int)
            width, height = max_x - x, max_y - y
            if width <= 0 or height <= 0:
                return fallback
            return width / height, (int(x), int(y), int(width), int(height))
        except (IndexError, TypeError, ValueError, cv2.error):
            return fallback

    def configure_shape_parameters(self, shell_ratio: float) -> None:
        """Condition scan and profile parameters without changing the counting axis."""
        severity = 0.0
        if self.use_shape_adaptive_params and shell_ratio < self.low_shell_ratio_threshold:
            transition_width = max(0.25, self.low_shell_ratio_threshold * 0.5)
            severity = float(np.clip((self.low_shell_ratio_threshold - shell_ratio) / transition_width, 0, 1))
        self.shape_adaptation_severity = severity

        # Low-ratio shells curve more strongly inside the same horizontal span.
        # Use a narrow band to find peak anchors, then a wider and less rigid
        # trace to recover the visible wall curvature.
        self.profile_band_width_ratio = self.width_ratio * (1 - 0.5 * severity)
        self.profile_trace_width_ratio = self.width_ratio * (1 + 0.5 * severity)
        if self.use_shape_adaptive_params:
            self.profile_min_distance_ratio = 0.025 + 0.025 * severity
        else:
            self.profile_min_distance_ratio = 0.05 if shell_ratio < 0.9 else 0.025
        self.trace_search_radius = 2 + int(round(2 * severity))
        self.trace_deviation_penalty = 0.08 - 0.05 * severity
        self.trace_smoothing_sigma = 1.2 + 0.8 * severity

        # Fixed 50-way splitting produces two- or three-pixel segments on a
        # narrow shell. Fewer segments and a larger curvature tolerance make
        # adsorption scores comparable to those of the usual elongated shells.
        self.active_width_ratio = self.width_ratio * (1 + severity / 3)
        self.active_num_segments = max(20, int(round(self.num_segments * (1 - 0.4 * severity))))
        self.active_filter_max_y_ratio = self.filter_max_y_ratio * (1 + 3 * severity)
        self.active_adsorption_thres = self.adsorption_thres - 0.08 * severity
        self.active_volution_thres = self.volution_thres - 0.07 * severity

    def process_img(self, img_path: str):
        img_rgb = cv2.imread(img_path)
        orig_h, orig_w = img_rgb.shape[:2]
        self.original_shape = (orig_h, orig_w)
        self.shell_ratio, shell_bbox = self.get_shell_geometry(img_path)
        self.configure_shape_parameters(self.shell_ratio)

        # Keep a normalized grayscale image for the profile fallback. Unlike the
        # adsorption image, it deliberately avoids morphology so fine and sparse
        # spirothecae remain visible.
        profile_img = resize_img(img_rgb.copy())
        self.profile_gray = cv2.cvtColor(profile_img, cv2.COLOR_BGR2GRAY)

        # Opening preprocess and resize
        kernel = np.ones((3, 3), np.int8)
        img_rgb = cv2.morphologyEx(img_rgb, cv2.MORPH_OPEN, kernel)

        img_rgb = resize_img(img_rgb)
        h, w = img_rgb.shape[:2]
        box_x, box_y, box_width, box_height = shell_bbox
        self.shell_bbox = (
            box_x * w / orig_w,
            box_y * h / orig_h,
            box_width * w / orig_w,
            box_height * h / orig_h,
        )

        resized_center = (self.center[0] * w // orig_w, self.center[1] * h // orig_h)
        self.center = resized_center

        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        assert img_gray.ndim == 2, "grayscale image required."

        # Binarization
        img_gray = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 2)

        # Morphological opening to remove noise
        kernel = np.ones((5, 5), np.int8)
        self.img_gray = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, kernel)

        self.get_outer_volution()

    @staticmethod
    def smooth_profile(values: np.ndarray, sigma: float = 2.0) -> np.ndarray:
        """Smooth a one-dimensional profile without adding a scipy dependency."""
        radius = max(1, int(round(3 * sigma)))
        positions = np.arange(-radius, radius + 1, dtype=float)
        kernel = np.exp(-(positions**2) / (2 * sigma**2))
        kernel /= np.sum(kernel)
        padded = np.pad(values.astype(float), radius, mode="reflect")
        return np.convolve(padded, kernel, mode="valid")

    @staticmethod
    def find_profile_peaks(
        profile: np.ndarray, start: int, end: int, min_distance: int, prominence: float
    ) -> list[int]:
        """Find separated local maxima with a local-prominence constraint."""
        if end - start < 3:
            return []

        segment = profile[start:end]
        prominence_window = max(12, int(0.08 * len(segment)))
        candidates = []
        for idx in range(1, len(segment) - 1):
            value = segment[idx]
            if value < segment[idx - 1] or value <= segment[idx + 1]:
                continue

            left = segment[max(0, idx - prominence_window) : idx + 1]
            right = segment[idx : min(len(segment), idx + prominence_window + 1)]
            local_prominence = value - max(float(np.min(left)), float(np.min(right)))
            if local_prominence >= prominence:
                candidates.append((local_prominence, value, idx + start))

        # Non-maximum suppression keeps the more prominent response when two
        # candidates represent the same thick spirotheca.
        selected = []
        for _, _, peak in sorted(candidates, reverse=True):
            if all(abs(peak - saved_peak) >= min_distance for saved_peak in selected):
                selected.append(peak)
        return sorted(selected)

    def balance_profile_peaks(
        self,
        profile: np.ndarray,
        center_y: int,
        min_distance: int,
        upper_peaks: list[int],
        lower_peaks: list[int],
    ) -> tuple[list[int], list[int]]:
        """Recover weak peaks only on the sparser side of the proloculus."""
        counts = [len(upper_peaks), len(lower_peaks)]
        if counts[0] == counts[1]:
            return upper_peaks, lower_peaks

        sparse_side = int(counts[1] < counts[0])
        bounds = [(0, center_y), (center_y, len(profile))]
        start, end = bounds[sparse_side]
        relaxed_peaks = self.find_profile_peaks(
            profile, start, end, min_distance, max(0.005, self.profile_prominence * 0.5)
        )
        old_gap = abs(counts[0] - counts[1])
        relaxed_counts = counts.copy()
        relaxed_counts[sparse_side] = len(relaxed_peaks)
        if len(relaxed_peaks) > counts[sparse_side] and abs(relaxed_counts[0] - relaxed_counts[1]) < old_gap:
            if sparse_side == 0:
                upper_peaks = relaxed_peaks
            else:
                lower_peaks = relaxed_peaks
        return upper_peaks, lower_peaks

    def trace_profile_curve(
        self, response: np.ndarray, anchor_y: int, x_start: int, x_end: int
    ) -> list[tuple[int, int]]:
        """Trace a smooth dark ridge left and right from a profile peak."""
        h, w = response.shape
        center_x = int(np.clip(self.center[0], x_start, x_end - 1))
        refine_radius = 3
        center_slice = response[
            max(0, anchor_y - refine_radius) : min(h, anchor_y + refine_radius + 1),
            max(x_start, center_x - 2) : min(x_end, center_x + 3),
        ]
        if center_slice.size:
            row_scores = np.mean(center_slice, axis=1)
            anchor_y = max(0, anchor_y - refine_radius) + int(np.argmax(row_scores))

        def trace(x_values: list[int]) -> list[tuple[int, int]]:
            points = []
            cur_y = anchor_y
            for x in x_values:
                candidates = range(
                    max(0, cur_y - self.trace_search_radius), min(h, cur_y + self.trace_search_radius + 1)
                )
                best_y = max(
                    candidates,
                    key=lambda y: float(response[y, x]) - self.trace_deviation_penalty * abs(y - cur_y),
                )
                cur_y = best_y
                points.append((x, cur_y))
            return points

        left = trace(list(range(center_x, x_start - 1, -1)))
        right = trace(list(range(center_x + 1, x_end)))
        points = list(reversed(left)) + right
        if len(points) < 5:
            return points

        # A short moving average suppresses pixel-scale zig-zags while retaining
        # the shallow curvature expected in the central band.
        y_values = np.array([point[1] for point in points], dtype=float)
        smoothed_y = self.smooth_profile(y_values, sigma=self.trace_smoothing_sigma)
        return [(point[0], int(round(y))) for point, y in zip(points, smoothed_y)]

    def get_profile_volutions(self) -> tuple[list[list[list[tuple[int, int]]]], list[list[float]]]:
        """Detect all central spirotheca candidates from a radial darkness profile."""
        gray = cv2.GaussianBlur(self.profile_gray, (5, 5), 0)
        h, w = gray.shape
        center_x, center_y = self.center
        shell_width = self.shell_bbox[2] if self.shape_adaptation_severity > 0 else w
        band_half_width = max(6, int(self.profile_band_width_ratio * 0.5 * shell_width))
        band_x_start = max(0, center_x - band_half_width)
        band_x_end = min(w, center_x + band_half_width + 1)
        trace_half_width = max(12, int(self.profile_trace_width_ratio * 0.5 * shell_width))
        trace_x_start = max(0, center_x - trace_half_width)
        trace_x_end = min(w, center_x + trace_half_width + 1)
        if band_x_end - band_x_start < 3 or trace_x_end - trace_x_start < 3:
            return [[], []], [[], []]

        band = gray[:, band_x_start:band_x_end].astype(float) / 255
        profile = self.smooth_profile(1 - np.mean(band, axis=1), sigma=2.0)
        profile_range = float(np.ptp(profile))
        if profile_range <= 1e-6:
            return [[], []], [[], []]
        profile = (profile - np.min(profile)) / profile_range

        min_distance = max(4, int(self.profile_min_distance_ratio * h))
        upper_peaks = self.find_profile_peaks(profile, 0, center_y, min_distance, self.profile_prominence)
        lower_peaks = self.find_profile_peaks(profile, center_y, h, min_distance, self.profile_prominence)
        upper_peaks, lower_peaks = self.balance_profile_peaks(
            profile, center_y, min_distance, upper_peaks, lower_peaks
        )

        darkness = 1 - gray.astype(float) / 255
        vertical_gradient = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
        gradient_scale = float(np.percentile(vertical_gradient, 95))
        if gradient_scale > 1e-6:
            vertical_gradient = np.clip(vertical_gradient / gradient_scale, 0, 1)
        response = cv2.GaussianBlur(0.6 * darkness + 0.4 * vertical_gradient, (3, 3), 0)
        profile_volutions = [[], []]
        profile_thickness = [[], []]
        for side, peaks in enumerate([upper_peaks, list(reversed(lower_peaks))]):
            for peak in peaks:
                curve = self.trace_profile_curve(response, peak, trace_x_start, trace_x_end)
                if len(curve) < 5:
                    continue
                profile_volutions[side].append(curve)

                half_height = profile[peak] - max(self.profile_prominence, 0.02) / 2
                top = peak
                while top > 0 and profile[top] > half_height:
                    top -= 1
                bottom = peak
                while bottom < h - 1 and profile[bottom] > half_height:
                    bottom += 1
                profile_thickness[side].append(float(max(1, bottom - top)))

        return profile_volutions, profile_thickness

    def should_use_profile_fallback(self, profile_volutions: list[list[list[tuple[int, int]]]]) -> bool:
        """Use the fallback only for out-of-distribution image regimes."""
        if not self.use_profile_fallback:
            return False

        orig_h, orig_w = self.original_shape
        shell_ratio = getattr(self, "shell_ratio", orig_w / max(orig_h, 1))
        black_ratio = float(np.mean(self.img_gray == 0))
        edge_ratio = float(np.mean(cv2.Canny(self.profile_gray, 50, 150) > 0))
        laplacian_variance = float(cv2.Laplacian(self.profile_gray, cv2.CV_64F).var())
        out_of_distribution = (
            shell_ratio < 1.2
            or min(orig_h, orig_w) >= 400
            or black_ratio > 0.45
            or edge_ratio > 0.18
            or laplacian_variance > 1000
        )
        if not out_of_distribution:
            return False

        counts = [len(side) for side in profile_volutions]
        return min(counts) >= 2 and max(counts) <= 14 and abs(counts[0] - counts[1]) <= 4

    def set_scan_direction(self, line: list[tuple[int, int]]):
        self.direction = np.sign(self.center[1] - line[0][1])

    def get_outer_volution(self):
        img_gray = self.img_gray
        h, w = img_gray.shape

        x_mid = self.center[0]
        y_top = int(np.min(np.where(img_gray[:, x_mid] == 0)[0]))
        y_bottom = int(np.max(np.where(img_gray[:, x_mid] == 0)[0]))

        shell_width = self.shell_bbox[2] if self.shape_adaptation_severity > 0 else w
        mid_img_width = int(self.active_width_ratio * 0.5 * shell_width)
        mid_img_width = min(mid_img_width, x_mid, w - x_mid - 1)

        line_upper = [(x_mid, y_top)]
        line_lower = [(x_mid, y_bottom)]
        for i in range(1, mid_img_width):
            x1 = x_mid + i
            x2 = x_mid - i
            for x in [x1, x2]:
                # add top point to line_upper
                for y in range(h):
                    if img_gray[y, x] == 0:
                        line_upper.append((x, y))
                        break

                # add bottom point to line_lower
                for y in range(h - 1, -1, -1):
                    if img_gray[y, x] == 0:
                        line_lower.append((x, y))
                        break

        self.line_upper = sorted(line_upper, key=lambda point: point[0])
        self.line_lower = sorted(line_lower, key=lambda point: point[0])

    def is_adsorption(self, points: list[tuple[int, int]]) -> bool:
        indensity = 0
        for x, y in points:
            if self.img_gray[y, x] == 0:
                indensity += 1
        return indensity / len(points) > self.active_adsorption_thres

    def is_volution(self, check_adsorption_mask: list[bool]) -> bool:
        if check_adsorption_mask:
            adsorption_rate = sum(check_adsorption_mask) / len(check_adsorption_mask)
            return adsorption_rate > self.active_volution_thres
        else:
            self.finish = True
            return False

    def distance_between_volutions(
        self, detected_volution: list[tuple[int, int]], cur_volutions: list[list[tuple[int, int]]]
    ) -> int:
        if not cur_volutions:
            return 10000

        last_volution = cur_volutions[-1]
        min_distance = 10000

        # Create a dictionary of x -> y for the last volution for faster lookup
        last_volution_dict = {point[0]: point[1] for point in last_volution}

        # For each point in the detected volution, find the corresponding point in the last volution
        for x, y in detected_volution:
            if x in last_volution_dict:
                # Calculate the absolute y-distance between points with the same x-coordinate
                distance = abs(y - last_volution_dict[x])
                min_distance = min(min_distance, distance)

        return min_distance

    def move(self, points: list[tuple[int, int]]) -> bool:
        """Move the points towards the center of the volution."""
        for i in range(len(points)):
            x, y = points[i]
            target_x, target_y = self.center
            # Calculate the direction vector
            direction_x = target_x - x
            direction_y = target_y - y

            if direction_y * self.direction <= 0:
                self.finish = True
                break

            # Normalize the direction vector
            magnitude = np.sqrt(direction_x**2 + direction_y**2)
            direction_x /= magnitude
            direction_y /= magnitude

            # Move a small distance towards the target point
            step_size = self.step * magnitude / abs(target_y - y)  # self.step / cos(\theta)
            points[i] = (int(x + direction_x * step_size), int(y + direction_y * step_size))

        adsorption_mask = self.is_adsorption(points)
        return adsorption_mask

    def catch_frontier(
        self, line_segments: list[list[tuple[int, int]]], step_forward: list[int], i: int, mask: bool
    ):
        max_step = max(step_forward)
        num_step = max_step - step_forward[i] - 1
        for _ in range(num_step):
            mask = self.move(line_segments[i])
            step_forward[i] += 1
        return mask

    def filter_segments(
        self, line_segments: list[list[tuple[int, int]]], step_forward: Optional[list] = None
    ):
        y_means = np.array([np.mean([point[1] for point in segment]) for segment in line_segments])
        ref_y = np.median(y_means)
        filter_max_y = self.active_filter_max_y_ratio * self.img_gray.shape[1]

        filtered_segments = []
        filtered_step_forward = []
        for i, segment in enumerate(line_segments):
            save = True
            # Filter out segment that are far away from ref_y
            for point in segment:
                if abs(point[1] - ref_y) > filter_max_y:
                    save = False
                    break

            # Filter out discontinuous segment
            for s in range(len(segment) - 1):
                point = segment[s]
                next_point = segment[s + 1]
                if abs(point[1] - next_point[1]) > 3:
                    save = False
                    break

            if save:
                filtered_segments.append(segment)
                if step_forward:
                    filtered_step_forward.append(step_forward[i])

        line_segments = filtered_segments
        if step_forward is not None:
            step_forward = filtered_step_forward

        return line_segments, step_forward

    def get_continuous_black_line(self, vertex: tuple[int, int], theta: float, max_expand_len: int):
        points = []
        for r in range(max_expand_len):
            x = int(vertex[0] + r * np.cos(theta))
            y = int(vertex[1] + r * np.sin(theta))
            if x < 0 or x >= self.img_gray.shape[1] or y < 0 or y >= self.img_gray.shape[0]:
                break
            if self.img_gray[y, x] == 255:
                break
            points.append((x, y))

        return points, len(points)

    def get_almost_black_line(self, vertex: tuple[int, int], theta: float, max_expand_len: int):
        points = []
        for r in range(max_expand_len):
            x = int(vertex[0] + r * np.cos(theta))
            y = int(vertex[1] + r * np.sin(theta))
            if x < 0 or x >= self.img_gray.shape[1] or y < 0 or y >= self.img_gray.shape[0]:
                break
            # if self.img_gray[y, x] == 0:
            points.append((x, y))

        return points, len([p for p in points if self.img_gray[p[1], p[0]] == 0])

    def get_expand_points(
        self, vertex: tuple[int, int], theta_ref: float, theta_margin: float, max_expand_len: int
    ):
        expand_points = []
        max_score = -1
        theta_range = np.arange(theta_ref - theta_margin, theta_ref + theta_margin, 0.01)
        for theta in theta_range:
            # points, num_points = self.get_continuous_black_line(vertex, theta, max_expand_len)
            points, num_points = self.get_almost_black_line(vertex, theta, max_expand_len)
            score = (num_points / max_expand_len) - 1.0 * (abs(theta - theta_ref) / theta_margin)
            if score > max_score:
                expand_points = points
                max_score = score
                new_theta_ref = theta

        return expand_points, new_theta_ref

    def expand_line_segments(
        self,
        line_segments: list[list[tuple[int, int]]],
        num_volutions: int,
        theta_margin: float = 0.5 * np.pi,
        max_expand_times: int = 6,
        max_expand_len: int = 10,
        base_expand_ratio: float = 0.8,
    ):
        num_original_points = sum([len(segment) for segment in line_segments])
        num_segs = len(line_segments)
        max_expand_ratio = base_expand_ratio**num_volutions

        # Calculate the slope of the left part
        left_points = []
        for segment in line_segments[: num_segs // 3]:
            left_points.extend(segment)
        vertex = line_segments[0][0]
        slope, intercept = fit_line(left_points)
        if np.isnan(slope):
            return line_segments
        theta_ref = np.arctan(slope) + np.pi

        # Expand left black pixel
        num_expand_points = 0
        for _ in range(max_expand_times):
            expand_points, _ = self.get_expand_points(vertex, theta_ref, theta_margin, max_expand_len)

            n = len(line_segments[0])
            if expand_points:
                vertex = expand_points[-1]
                for i in range(0, len(expand_points), n):
                    new_segments = expand_points[i + n : i : -1]  # reverse order
                    if len(new_segments) >= n:
                        line_segments.insert(0, new_segments)
                        num_expand_points += len(new_segments)

            if num_expand_points >= max_expand_ratio * num_original_points:
                break

        # Calculate the slope of the right part
        right_points = []
        for segment in line_segments[2 * num_segs // 3 :]:
            right_points.extend(segment)
        vertex = line_segments[-1][-1]
        slope, intercept = fit_line(right_points)
        if np.isnan(slope):
            return line_segments
        theta_ref = np.arctan(slope)

        # Expand right black pixel
        num_expand_points = 0
        for _ in range(max_expand_times):
            expand_points, _ = self.get_expand_points(vertex, theta_ref, theta_margin, max_expand_len)

            n = len(line_segments[-1])
            if expand_points:
                vertex = expand_points[-1]
                for i in range(0, len(expand_points), n):
                    new_segments = expand_points[i : i + n + 1]
                    if len(new_segments) >= n:
                        line_segments.append(new_segments)
                        num_expand_points += len(new_segments)

            if num_expand_points >= max_expand_ratio * num_original_points:
                break

        return line_segments

    def update_line_segments(self, line_segments: list[list[tuple[int, int]]], num_split: int = 9):
        if not line_segments:
            return line_segments
        new_line = []
        while num_split > len(line_segments):
            num_split = num_split // 2
        split_len = len(line_segments) // num_split

        if len(line_segments) - split_len > 0:
            for idx in range(0, len(line_segments) - split_len, split_len):
                p1 = line_segments[idx][0]
                p2 = line_segments[idx + split_len][0]
                new_line.extend(bresenham(p1, p2))

            new_line.extend(bresenham(p2, line_segments[-1][-1]))
            line_segments = split_into_segments(new_line, self.active_num_segments)
        return line_segments

    def reach_step_limit(self, step_forward: list[int]) -> bool:
        max_step = max(step_forward)
        limit = 0.1 * self.img_gray.shape[0] / self.step
        return max_step >= limit

    def scan_in_volution(self, line_segments: list[list[tuple[int, int]]]):
        check_adsorption_mask = [self.is_adsorption(segment) for segment in line_segments]

        step_forward = [0 for _ in range(len(line_segments))]

        # Move line_segments until all segments leave current volution
        while any(check_adsorption_mask) and not self.finish:
            for i, adsorption in enumerate(check_adsorption_mask):
                if adsorption:
                    check_adsorption_mask[i] = self.move(line_segments[i])
                    step_forward[i] += 1
            if self.reach_step_limit(step_forward):
                break

        # Post process: filter out segments that moved too far
        line_segments, step_forward = self.filter_segments(line_segments, step_forward)

        assert step_forward is not None
        if len(step_forward) > 0:
            thickness = np.mean(step_forward) * self.step
        else:
            thickness = 0

        for i in range(len(line_segments)):
            _ = self.catch_frontier(line_segments, step_forward, i, check_adsorption_mask[i])

        return line_segments, thickness

    def scan_between_volutions(self, line_segments: list[list[tuple[int, int]]]):
        check_adsorption_mask = [self.is_adsorption(segment) for segment in line_segments]
        step_forward = [0 for _ in range(len(line_segments))]
        cur_adsorption_time = [0 for _ in range(len(line_segments))]

        while not self.is_volution(check_adsorption_mask) and not self.finish:
            for i, adsorption in enumerate(check_adsorption_mask):
                if adsorption:
                    cur_adsorption_time[i] += 1
                    if cur_adsorption_time[i] == self.max_adsorption_time:
                        # end adsorption and catch frontier segment
                        check_adsorption_mask[i] = self.catch_frontier(
                            line_segments, step_forward, i, check_adsorption_mask[i]
                        )
                        cur_adsorption_time[i] = 0
                else:
                    check_adsorption_mask[i] = self.move(line_segments[i])
                    step_forward[i] += 1

        if not self.finish:
            for i in range(len(line_segments)):
                check_adsorption_mask[i] = self.catch_frontier(
                    line_segments, step_forward, i, check_adsorption_mask[i]
                )

        line_segments, _ = self.filter_segments(line_segments, step_forward)

        return line_segments

    def count_volutions(
        self, img_path: str, center: tuple[int, int]
    ) -> tuple[dict[int, list], dict[int, float]]:
        """
        Detect the volutions and measure the thickness of each volution in the image.
        Try to detect the initial chamber with a high confidence level.

        Parameters:
        img_path (string): The path of input image.
        center (tuple): Center of the initial chamber.

        Returns:
        volutions_dict (dict): Dictionary of detected volutions where:
            * Key (int): Volution index number, 1 represents the innermost volution:
                - Positive numbers represent upper volutions (e.g., 1, 2, 3...)
                - Negative numbers represent lower volutions (e.g., -1, -2, -3...)
            * Value (list): List of (x, y) tuples representing points along the volution curve,
                where x and y are normalized coordinates (0.0 to 1.0)

        thickness_dict (dict): Dictionary of volution thickness measurements where:
            * Key (int): Volution index number (matches keys in volutions_dict)
            * Value (float): Normalized thickness of the volution (0.0 to 1.0)
        """
        self.center = center
        self.process_img(img_path)

        volutions = [[], []]  # upper and lower
        thickness_per_vol = [[], []]

        for i, line in enumerate([self.line_upper, self.line_lower]):
            self.set_scan_direction(line)
            self.finish = False
            line_segments = split_into_segments(line, self.active_num_segments)
            line_segments, _ = self.filter_segments(line_segments)
            line_segments = self.update_line_segments(line_segments)

            while not self.finish:
                detected_volution = [segment[0] for segment in line_segments]
                if (
                    len(detected_volution) > 3
                    and self.distance_between_volutions(detected_volution, volutions[i]) > 2
                ):
                    volutions[i].append(detected_volution)
                else:
                    break

                line_segments, thickness = self.scan_in_volution(line_segments)

                thickness_per_vol[i].append(thickness)
                if self.finish:
                    break

                line_segments = self.scan_between_volutions(line_segments)
                if self.finish:
                    break

                line_segments = [segment for segment in line_segments if self.is_adsorption(segment)]

                if not line_segments:
                    break

                line_segments = self.update_line_segments(line_segments)
                line_segments = self.expand_line_segments(line_segments, len(volutions[i]))
                line_segments = self.update_line_segments(line_segments)

        profile_volutions, profile_thickness = self.get_profile_volutions()
        if self.should_use_profile_fallback(profile_volutions):
            volutions = profile_volutions
            thickness_per_vol = profile_thickness
            self.detection_mode = "profile"
        else:
            self.detection_mode = "adsorption"

        # Return relative values
        h, w = self.img_gray.shape
        for volution in volutions:
            for line_segments in volution:
                for i, point in enumerate(line_segments):
                    line_segments[i] = (point[0] / w, point[1] / h)

        for thickness_list in thickness_per_vol:
            for i in range(len(thickness_list)):
                thickness_list[i] = thickness_list[i] / h

        # Reformat into dict
        volutions_dict = {}
        thickness_dict = {}
        for i in range(2):
            vols = volutions[i]
            thicks = thickness_per_vol[i]
            for j in range(len(vols)):
                vol_idx = (len(vols) - j) * (
                    -1
                ) ** i  # positive value for upper voluitons, negative for lower
                volutions_dict[vol_idx] = vols[j]
                thickness_dict[vol_idx] = thicks[j]

        return volutions_dict, thickness_dict
