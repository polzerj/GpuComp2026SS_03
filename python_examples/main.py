import os
import time

from preprocessing import main as preprocessing_main
from boundary_detection import main as boundary_detection_main
from contour_to_signal import main as contour_to_signal_main
from plot_signal import main as plot_signal_main


def process(image_path: str) -> None:
    benchmark_file = "benchmark.txt"

    start = time.time()
    binary_image = preprocessing_main(image_path)
    time_taken = time.time() - start
    with open(benchmark_file, "a") as f:
        f.write(
            f"{os.path.basename(image_path)} - Preprocessing: {time_taken:.2f} seconds\n"
        )
    print(f"Time taken for preprocessing: {time_taken:.2f} seconds")

    start = time.time()
    boundary_image = boundary_detection_main(
        binary_image, show_images=False, save_images=False
    )
    time_taken = time.time() - start
    with open(benchmark_file, "a") as f:
        f.write(
            f"{os.path.basename(image_path)} - Boundary Detection: {time_taken:.2f} seconds\n"
        )
    print(f"Time taken for boundary detection: {time_taken:.2f} seconds")

    start = time.time()
    smooth_points, px, py = contour_to_signal_main(
        boundary_image, show_images=False, save_images=False
    )
    time_taken = time.time() - start
    with open(benchmark_file, "a") as f:
        f.write(
            f"{os.path.basename(image_path)} - Contour to Signal: {time_taken:.2f} seconds\n"
        )
    print(f"Time taken for contour to signal: {time_taken:.2f} seconds")


def main() -> None:
    benchmark_file = "benchmark.txt"

    complete_start = time.time()
    images = os.listdir("../data")
    for image in images:
        if image.endswith(".jpg"):
            print(f"Processing {image}...")
            individual_start = time.time()
            process(os.path.join("../data", image))

            time_taken = time.time() - individual_start
            with open(benchmark_file, "a") as f:
                f.write(f"{image}: {time_taken:.2f} seconds\n")
            print(f"Time taken for {image}: {time_taken:.2f} seconds")

    print()
    print(f"Total time taken: {time.time() - complete_start:.2f} seconds")


if __name__ == "__main__":
    main()
