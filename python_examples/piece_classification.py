from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from contour_to_signal import (
    enclosing_circle_approx,
    find_contour_chain_approx_simple,
    load_grayscale_image,
    radial_signal,
    trace_contour,
)

IMAGE_DIR = Path(__file__).parent.parent.parent / "data"


@dataclass
class ClassifiedPiece:
    label: int
    edges: list[str]
    edge_deviations: list[float]
    corner_ranks: list[int]
    valid: bool


@dataclass
class PieceAnalysis:
    label: int
    boundary_points: list[tuple[int, int]]
    smooth_points: list[tuple[int, int]]
    center: tuple[float, float]
    radius: float
    signal: list[float]
    corner_ranks: list[int]
    classified: ClassifiedPiece


def edge_length_between(rank_a: int, rank_b: int, piece_length: int) -> int:
    return (rank_b + piece_length - rank_a) % piece_length


def cross2d(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * by - ay * bx


def classify_single_edge(
    points: list[tuple[int, int]],
    piece_start: int,
    piece_length: int,
    corner_a_idx: int,
    corner_b_idx: int,
    center_x: float,
    center_y: float,
    edge_threshold: float,
) -> tuple[str, float]:
    a_x, a_y = points[corner_a_idx]
    b_x, b_y = points[corner_b_idx]

    rank_a = corner_a_idx
    rank_b = corner_b_idx
    edge_len = edge_length_between(rank_a, rank_b, piece_length)

    abx = float(b_x - a_x)
    aby = float(b_y - a_y)
    ab_len = float(np.hypot(abx, aby))

    if ab_len == 0.0:
        return "L", 0.0

    acx = center_x - float(a_x)
    acy = center_y - float(a_y)
    center_side = cross2d(abx, aby, acx, acy)
    center_sign = 1.0 if center_side > 0.0 else -1.0

    from_rank = edge_len // 4
    to_rank = (3 * edge_len) // 4

    deviation_sum = 0.0
    count = 0

    for d in range(from_rank, to_rank + 1):
        rank = (rank_a + d) % piece_length
        idx = piece_start + rank

        if idx >= len(points):
            continue

        p_x, p_y = points[idx]
        apx = float(p_x - a_x)
        apy = float(p_y - a_y)

        point_side = cross2d(abx, aby, apx, apy)
        signed_outward_distance = -(point_side / ab_len) * center_sign

        deviation_sum += signed_outward_distance
        count += 1

    if count == 0:
        return "L", 0.0

    mean_deviation = deviation_sum / float(count)

    if mean_deviation > edge_threshold:
        return "V", mean_deviation

    if mean_deviation < -edge_threshold:
        return "C", mean_deviation

    return "L", mean_deviation


def classify_edges(
    points: list[tuple[int, int]],
    corner_ranks: list[int],
    center_x: float,
    center_y: float,
    edge_threshold: float,
) -> ClassifiedPiece:
    classified = ClassifiedPiece(
        label=0,
        edges=["L", "L", "L", "L"],
        edge_deviations=[0.0, 0.0, 0.0, 0.0],
        corner_ranks=corner_ranks,
        valid=False,
    )

    if len(corner_ranks) != 4:
        return classified

    piece_start = 0
    piece_length = len(points)

    if piece_length == 0:
        return classified

    ok = True

    for edge in range(4):
        corner_a_idx = corner_ranks[edge]
        corner_b_idx = corner_ranks[(edge + 1) % 4]

        if corner_a_idx >= len(points) or corner_b_idx >= len(points):
            ok = False
            continue

        edge_label, mean_deviation = classify_single_edge(
            points,
            piece_start,
            piece_length,
            corner_a_idx,
            corner_b_idx,
            center_x,
            center_y,
            edge_threshold,
        )

        classified.edges[edge] = edge_label
        classified.edge_deviations[edge] = mean_deviation

    classified.valid = ok
    return classified


def detect_corners_from_signal(points: list[tuple[int, int]]) -> tuple[
    tuple[float, float],
    float,
    list[float],
    list[int],
]:
    center, radius = enclosing_circle_approx(points)
    signal = radial_signal(points, center)
    peaks_idx = detect_corners_like_cuda(
        signal,
        peak_detection_radius=70,
        prominence_threshold=8.0,
        distance_threshold=0,
    )
    return center, radius, signal, peaks_idx


def cyclic_distance(a: int, b: int, length: int) -> int:
    diff = abs(a - b)
    return min(diff, length - diff)


def detect_corners_like_cuda(
    signal: list[float],
    peak_detection_radius: int,
    prominence_threshold: float,
    distance_threshold: int,
) -> list[int]:
    num_points = len(signal)
    if num_points == 0:
        return []

    candidates: list[tuple[int, float]] = []

    for idx, self_distance in enumerate(signal):
        is_peak = True
        neighbor_sum = 0.0
        neighbor_count = 0

        radius_max = min(peak_detection_radius, num_points)
        for radius in range(1, radius_max + 1):
            right_rank = (idx + radius) % num_points
            left_rank = (idx + num_points - radius) % num_points

            right_distance = signal[right_rank]
            left_distance = signal[left_rank]

            if self_distance <= right_distance or self_distance <= left_distance:
                is_peak = False
                break

            neighbor_sum += right_distance + left_distance
            neighbor_count += 2

        if not is_peak or neighbor_count == 0:
            continue

        neighbor_avg = neighbor_sum / float(neighbor_count)
        prominence = self_distance - neighbor_avg

        if prominence >= prominence_threshold:
            candidates.append((idx, prominence))

    min_corner_distance = distance_threshold
    if min_corner_distance == 0:
        min_corner_distance = max(1, num_points // 8)

    selected: list[int] = []

    for idx, _score in sorted(candidates, key=lambda item: item[1], reverse=True):
        if all(
            cyclic_distance(idx, chosen, num_points) >= min_corner_distance
            for chosen in selected
        ):
            selected.append(idx)

        if len(selected) == 4:
            break

    selected.sort()
    return selected


def extract_connected_contours(boundary: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = boundary > 0
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    neighbors = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]

    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue

            queue: deque[tuple[int, int]] = deque([(y, x)])
            visited[y, x] = True
            component_pixels: list[tuple[int, int]] = []

            while queue:
                cy, cx = queue.popleft()
                component_pixels.append((cy, cx))

                for dy, dx in neighbors:
                    ny = cy + dy
                    nx = cx + dx
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and mask[ny, nx]
                        and not visited[ny, nx]
                    ):
                        visited[ny, nx] = True
                        queue.append((ny, nx))

            if len(component_pixels) < 10:
                continue

            component_mask = np.zeros_like(boundary, dtype=np.uint8)
            for cy, cx in component_pixels:
                component_mask[cy, cx] = 255

            contour = trace_contour(component_mask)
            if len(contour) > 0:
                components.append([(int(x), int(y)) for y, x in contour])

    return components


def load_smooth_points_from_boundary() -> list[tuple[int, int]]:
    boundary = load_grayscale_image(IMAGE_DIR / "processed" / "boundary.png")
    contour = find_contour_chain_approx_simple(boundary)
    raw_points = [(int(x), int(y)) for x, y in contour]
    return smooth_contour(raw_points, window=5)


def load_piece_analyses_from_boundary() -> list[PieceAnalysis]:
    boundary = load_grayscale_image(IMAGE_DIR / "processed" / "boundary.png")
    contours = extract_connected_contours(boundary)
    analyses: list[PieceAnalysis] = []

    for label, contour in enumerate(contours):
        smooth_points = smooth_contour(contour, window=5)
        center, radius, signal, peaks_idx = detect_corners_from_signal(smooth_points)
        classified = classify_edges(
            smooth_points,
            peaks_idx,
            center[0],
            center[1],
            edge_threshold=15.0,
        )
        classified.label = label
        analyses.append(
            PieceAnalysis(
                label=label,
                boundary_points=contour,
                smooth_points=smooth_points,
                center=center,
                radius=radius,
                signal=signal,
                corner_ranks=peaks_idx,
                classified=classified,
            )
        )

    return analyses


def save_classification_csv(piece: ClassifiedPiece, filename: Path) -> None:
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "label",
                "edge",
                "corner_rank_a",
                "corner_rank_b",
                "class",
                "mean_deviation",
                "valid",
            ]
        )
        for edge_index, edge_label in enumerate(piece.edges):
            corner_a = (
                piece.corner_ranks[edge_index]
                if edge_index < len(piece.corner_ranks)
                else ""
            )
            corner_b_index = (
                (edge_index + 1) % len(piece.corner_ranks) if piece.corner_ranks else 0
            )
            corner_b = piece.corner_ranks[corner_b_index] if piece.corner_ranks else ""
            writer.writerow(
                [
                    piece.label,
                    edge_index,
                    corner_a,
                    corner_b,
                    edge_label,
                    piece.edge_deviations[edge_index],
                    piece.valid,
                ]
            )


def save_all_classifications_csv(
    pieces: list[PieceAnalysis],
    filename: Path,
) -> None:
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "label",
                "edge",
                "corner_rank_a",
                "corner_rank_b",
                "class",
                "mean_deviation",
                "valid",
            ]
        )

        for piece in pieces:
            if len(piece.classified.edges) != 4:
                continue

            for edge_index, edge_label in enumerate(piece.classified.edges):
                corner_ranks = piece.classified.corner_ranks
                corner_a = (
                    corner_ranks[edge_index] if edge_index < len(corner_ranks) else ""
                )
                corner_b_index = (
                    (edge_index + 1) % len(corner_ranks) if corner_ranks else 0
                )
                corner_b = corner_ranks[corner_b_index] if corner_ranks else ""
                writer.writerow(
                    [
                        piece.label,
                        edge_index,
                        corner_a,
                        corner_b,
                        edge_label,
                        piece.classified.edge_deviations[edge_index],
                        piece.classified.valid,
                    ]
                )


def main(
    smooth_points: list[tuple[int, int]] | None = None,
    show_images: bool = False,
    save_images: bool = False,
    save_dir: Path | None = None,
) -> ClassifiedPiece:
    if smooth_points is None:
        smooth_points = load_smooth_points_from_boundary()

    center, radius, signal, peaks_idx = detect_corners_from_signal(smooth_points)
    classified = classify_edges(
        smooth_points,
        peaks_idx,
        center[0],
        center[1],
        edge_threshold=15.0,
    )

    # print("center:", center)
    # print("radius:", radius)
    # print("corner ranks:", classified.corner_ranks)
    print("edges:", "".join(classified.edges))
    # print("edge deviations:", classified.edge_deviations)
    # print("valid:", classified.valid)

    if save_images:
        if save_dir is None:
            output_csv = IMAGE_DIR / "processed" / "piece_classification.csv"
        else:
            output_csv = save_dir / "piece_classification.csv"
        save_classification_csv(classified, output_csv)

    if save_images or show_images:
        fig, ax = plt.subplots(figsize=(10, 10 / 1000 * 727))
        ax.axis("off")

        xs = [p[0] for p in smooth_points]
        ys = [p[1] for p in smooth_points]
        ax.plot(xs, ys, linewidth=2, color="orange")

        circle = Circle(center, radius, fill=False, linewidth=2, color="green")
        ax.add_patch(circle)
        ax.scatter(center[0], center[1], s=30, color="blue")

        if classified.valid:
            edge_colors = {"L": "white", "V": "red", "C": "cyan"}
            for edge_index, edge_label in enumerate(classified.edges):
                start_idx = classified.corner_ranks[edge_index]
                end_idx = classified.corner_ranks[(edge_index + 1) % 4]

                if start_idx <= end_idx:
                    segment = smooth_points[start_idx : end_idx + 1]
                else:
                    segment = smooth_points[start_idx:] + smooth_points[: end_idx + 1]

                seg_x = [p[0] for p in segment]
                seg_y = [p[1] for p in segment]
                ax.plot(
                    seg_x,
                    seg_y,
                    linewidth=3,
                    color=edge_colors.get(edge_label, "yellow"),
                )

                mid_point = segment[len(segment) // 2]
                ax.text(
                    mid_point[0],
                    mid_point[1],
                    edge_label,
                    color="yellow",
                    fontsize=14,
                    weight="bold",
                )

            corner_points = [smooth_points[idx] for idx in classified.corner_ranks]
            ax.scatter(
                [p[0] for p in corner_points],
                [p[1] for p in corner_points],
                s=80,
                color="red",
            )

        if save_images:
            plt.savefig(
                IMAGE_DIR / "processed" / "piece_classification.png",
                bbox_inches="tight",
                pad_inches=0,
                dpi=300,
            )
        if show_images:
            plt.show()
        else:
            plt.close(fig)

    return classified


def classify_all_pieces(
    boundary_image: np.ndarray | None = None,
    save_images: bool = False,
    save_dir: Path | None = None,
) -> list[PieceAnalysis]:
    if boundary_image is None:
        boundary_image = load_grayscale_image(IMAGE_DIR / "processed" / "boundary.png")

    contours = extract_connected_contours(boundary_image)
    pieces: list[PieceAnalysis] = []

    for label, contour in enumerate(contours):
        smooth_points = smooth_contour(contour, window=5)
        center, radius, signal, peaks_idx = detect_corners_from_signal(smooth_points)
        classified = classify_edges(
            smooth_points,
            peaks_idx,
            center[0],
            center[1],
            edge_threshold=15.0,
        )
        classified.label = label

        piece = PieceAnalysis(
            label=label,
            boundary_points=contour,
            smooth_points=smooth_points,
            center=center,
            radius=radius,
            signal=signal,
            corner_ranks=peaks_idx,
            classified=classified,
        )
        pieces.append(piece)

    if save_images:
        if save_dir is None:
            output_csv = IMAGE_DIR / "processed" / "piece_classification.csv"
        else:
            output_csv = save_dir / "piece_classification.csv"
        save_all_classifications_csv(pieces, output_csv)

    return pieces


if __name__ == "__main__":
    main()
