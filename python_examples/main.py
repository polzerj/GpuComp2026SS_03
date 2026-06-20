import os
import time
from pathlib import Path

from preprocessing import main as preprocessing_main
from boundary_detection import main as boundary_detection_main
from contour_to_signal import main as contour_to_signal_main
from piece_classification import main as piece_classification_main
from visualizer import visualize_piece


def process(image_path: Path) -> None:
    benchmark_file = "benchmark.txt"

    start = time.time()
    binary_image = preprocessing_main(image_path)
    time_taken = time.time() - start
    with open(benchmark_file, "a") as f:
        f.write(f"{image_path.name} - Preprocessing: {time_taken:.2f} seconds\n")
    print(f"Time taken for preprocessing: {time_taken:.2f} seconds")

    start = time.time()
    boundary_image = boundary_detection_main(
        binary_image, show_images=False, save_images=False
    )
    time_taken = time.time() - start
    with open(benchmark_file, "a") as f:
        f.write(f"{image_path.name} - Boundary Detection: {time_taken:.2f} seconds\n")
    print(f"Time taken for boundary detection: {time_taken:.2f} seconds")

    start = time.time()
    smooth_points, px, py = contour_to_signal_main(
        boundary_image, show_images=False, save_images=False
    )
    time_taken = time.time() - start
    with open(benchmark_file, "a") as f:
        f.write(f"{image_path.name} - Contour to Signal: {time_taken:.2f} seconds\n")
    print(f"Time taken for contour to signal: {time_taken:.2f} seconds")

    start = time.time()
    classified_piece = piece_classification_main(
        smooth_points, show_images=False, save_images=False
    )
    time_taken = time.time() - start
    with open(benchmark_file, "a") as f:
        f.write(f"{image_path.name} - Piece Classification: {time_taken:.2f} seconds\n")
    print(f"Time taken for piece classification: {time_taken:.2f} seconds")

    start = time.time()
    visualize_piece(
        image_path,
        smooth_points,
        classified_piece,
        output_path=Path("../data/processed") / f"{image_path.stem}_visualized.png",
        show_image=False,
    )
    time_taken = time.time() - start
    with open(benchmark_file, "a") as f:
        f.write(f"{image_path.name} - Visualization: {time_taken:.2f} seconds\n")
    print(f"Time taken for visualization: {time_taken:.2f} seconds")


def main() -> None:
    benchmark_file = "benchmark.txt"

    process(Path("../data") / "single.JPG")
    return

    complete_start = time.time()
    images = os.listdir("../data")
    for image in images:
        if image.endswith(".jpg"):
            print(f"Processing {image}...")
            individual_start = time.time()
            process(Path("../data") / image)

            time_taken = time.time() - individual_start
            with open(benchmark_file, "a") as f:
                f.write(f"{image}: {time_taken:.2f} seconds\n")
            print(f"Time taken for {image}: {time_taken:.2f} seconds")

    print()
    print(f"Total time taken: {time.time() - complete_start:.2f} seconds")


if __name__ == "__main__":
    main()
