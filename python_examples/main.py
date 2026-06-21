import os
import time
from pathlib import Path

from preprocessing import main as preprocessing_main
from boundary_detection import main as boundary_detection_main
from piece_classification import classify_all_pieces
from visualizer import visualize_pieces


def process(image_path: Path) -> None:
    print(f"Processing {image_path.name}...")
    individual_start = time.time()

    benchmark_file = "benchmark.txt"

    start = time.time()
    binary_image = preprocessing_main(image_path)
    time_taken = time.time() - start
    with open(benchmark_file, "a") as f:
        f.write(f"{image_path.name} - Preprocessing: {time_taken:.6f} seconds\n")
    print(f"Time taken for preprocessing: {time_taken:.6f} seconds")

    start = time.time()
    boundary_image = boundary_detection_main(
        binary_image, show_images=False, save_images=False
    )
    time_taken = time.time() - start
    with open(benchmark_file, "a") as f:
        f.write(f"{image_path.name} - Boundary Detection: {time_taken:.6f} seconds\n")
    print(f"Time taken for boundary detection: {time_taken:.6f} seconds")

    start = time.time()
    pieces = classify_all_pieces(
        boundary_image,
        save_images=True,
        save_dir=Path("../data/processed") / image_path.stem,
    )
    time_taken = time.time() - start
    with open(benchmark_file, "a") as f:
        f.write(f"{image_path.name} - Piece Classification: {time_taken:.6f} seconds\n")
    print(f"Time taken for piece classification: {time_taken:.6f} seconds")
    print(f"Detected {len(pieces)} pieces")

    start = time.time()
    visualize_pieces(
        image_path,
        pieces,
        output_path=Path("../data/processed") / f"{image_path.stem}_visualized.png",
        show_image=False,
    )
    time_taken = time.time() - start
    with open(benchmark_file, "a") as f:
        f.write(f"{image_path.name} - Visualization: {time_taken:.6f} seconds\n")
    print(f"Time taken for visualization: {time_taken:.6f} seconds")

    time_taken = time.time() - individual_start
    with open(benchmark_file, "a") as f:
        f.write(f"{image_path.name}: {time_taken:.6f} seconds\n")
    print(f"Time taken for {image_path.name}: {time_taken:.6f} seconds")


def main() -> None:
    benchmark_file = "benchmark.txt"
    with open(benchmark_file, "w") as f:
        f.write("Benchmark Results:\n")
        f.write("==================\n")

    process(Path("../data") / "single.JPG")
    process(Path("../data") / "3_p2 (Copy).jpg")
    return

    complete_start = time.time()
    images = os.listdir("../data")
    for image in images:
        if image.endswith(".jpg"):
            process(Path("../data") / image)

    print()
    print(f"Total time taken: {time.time() - complete_start:.6f} seconds")


if __name__ == "__main__":
    main()
