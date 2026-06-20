from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from PIL import Image

from contour_to_signal import load_grayscale_image
from piece_classification import ClassifiedPiece, load_smooth_points_from_boundary

IMAGE_DIR = Path(__file__).parent.parent.parent / "data"


def load_rgb_image(image_path: Path | str) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    return np.array(image)


def color_for_edge_class(edge_class: str) -> tuple[int, int, int]:
    if edge_class == "L":
        return 255, 255, 100
    if edge_class == "V":
        return 255, 0, 0
    if edge_class == "C":
        return 0, 120, 255
    return 255, 0, 255


def draw_rect_outline(ax, min_x: int, min_y: int, max_x: int, max_y: int) -> None:
    ax.plot(
        [min_x, max_x, max_x, min_x, min_x],
        [min_y, min_y, max_y, max_y, min_y],
        color="green",
        linewidth=2,
    )


def get_piece_bounds(
    points: list[tuple[int, int]], corner_ranks: list[int]
) -> tuple[int, int, int, int]:
    xs = [points[idx][0] for idx in corner_ranks]
    ys = [points[idx][1] for idx in corner_ranks]
    return min(xs), min(ys), max(xs), max(ys)


def visualize_piece(
    base_image_path: Path,
    points: list[tuple[int, int]],
    classified: ClassifiedPiece,
    output_path: Path | None = None,
    show_image: bool = True,
) -> Path | None:
    rgb_image = load_rgb_image(base_image_path)
    grayscale = rgb_image.mean(axis=2).astype(np.uint8)
    display_image = np.repeat((grayscale // 3)[:, :, None], 3, axis=2)

    fig, ax = plt.subplots(figsize=(10, 10 / 1000 * 727))
    ax.imshow(display_image)
    ax.axis("off")

    if classified.valid and len(classified.corner_ranks) == 4:
        min_x, min_y, max_x, max_y = get_piece_bounds(points, classified.corner_ranks)
        draw_rect_outline(ax, min_x - 2, min_y - 2, max_x + 2, max_y + 2)

        piece_length = len(points)
        top_edge_index = 3
        first_corner_idx = classified.corner_ranks[0]
        last_corner_idx = classified.corner_ranks[3]

        if first_corner_idx < piece_length and last_corner_idx < piece_length:
            if points[first_corner_idx][0] < points[last_corner_idx][0]:
                top_edge_index = 0

        label_x = min_x
        label_y = min_y - 20
        cursor_x = label_x
        edge_order = [classified.edges[(top_edge_index + i) % 4] for i in range(4)]

        for edge_label in edge_order:
            ax.text(
                cursor_x,
                label_y,
                edge_label,
                color="white",
                fontsize=14,
                weight="bold",
            )
            cursor_x += 28

        for edge_index in range(4):
            a_idx = classified.corner_ranks[edge_index]
            b_idx = classified.corner_ranks[(edge_index + 1) % 4]

            if a_idx >= piece_length or b_idx >= piece_length:
                continue

            if a_idx <= b_idx:
                segment = points[a_idx : b_idx + 1]
            else:
                segment = points[a_idx:] + points[: b_idx + 1]

            edge_r, edge_g, edge_b = color_for_edge_class(classified.edges[edge_index])
            seg_x = [p[0] for p in segment]
            seg_y = [p[1] for p in segment]
            ax.plot(
                seg_x,
                seg_y,
                color=np.array([edge_r, edge_g, edge_b]) / 255.0,
                linewidth=3,
            )

            for corner_idx in (a_idx, b_idx):
                corner_x, corner_y = points[corner_idx]
                ax.scatter([corner_x], [corner_y], s=80, color="green")

        circle_center = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
        circle_radius = max(
            np.hypot(
                points[idx][0] - circle_center[0], points[idx][1] - circle_center[1]
            )
            for idx in classified.corner_ranks
        )
        ax.add_patch(
            Circle(circle_center, circle_radius, fill=False, linewidth=2, color="green")
        )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=300)

    if show_image:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def main(
    base_image_path: Path | None = None,
    show_image: bool = True,
    save_image: bool = True,
) -> None:
    if base_image_path is None:
        base_image_path = IMAGE_DIR / "input-example-tuwel.jpg"

    boundary = load_grayscale_image(IMAGE_DIR / "processed" / "boundary.png")
    if boundary.ndim == 3:
        boundary = boundary[:, :, 0]

    points = load_smooth_points_from_boundary()

    from piece_classification import detect_corners_from_signal, classify_edges

    center, _, _, peaks_idx = detect_corners_from_signal(points)
    classified = classify_edges(
        points, peaks_idx, center[0], center[1], edge_threshold=15.0
    )

    output_path = (
        IMAGE_DIR / "processed" / "python_visualized_classified.png"
        if save_image
        else None
    )
    visualize_piece(
        base_image_path,
        points,
        classified,
        output_path=output_path,
        show_image=show_image,
    )


if __name__ == "__main__":
    main()
