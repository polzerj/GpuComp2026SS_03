import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import filters
import numpy as np
from PIL import Image
from pathlib import Path

IMAGE_DIR = Path(__file__).parent.parent.parent / "data"


def load_grayscale_image(image_path):
    image = Image.open(image_path).convert("L")
    return np.array(image)


def show_img(img, width=10):
    plt.figure(figsize=(width, width / 1000 * 727))
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.show()


def erode(image, kernel, iterations=1):
    result = image.copy()
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    for _ in range(iterations):
        padded = np.pad(result, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")
        new_image = np.zeros_like(result)
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                region = padded[i : i + kh, j : j + kw]

                # Erosion: all kernel positions must match foreground
                if np.all(region[kernel == 1] == 255):
                    new_image[i, j] = 255
        result = new_image
    return result


def dilate(image, kernel, iterations=1):
    result = image.copy()
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    for _ in range(iterations):
        padded = np.pad(result, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")
        new_image = np.zeros_like(result)
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                region = padded[i : i + kh, j : j + kw]

                # Dilation: any kernel position matches foreground
                if np.any(region[kernel == 1] == 255):
                    new_image[i, j] = 255
        result = new_image

    return result


def morphological_open(image, kernel, iterations=1):
    eroded = erode(image, kernel, iterations)
    opened = dilate(eroded, kernel, iterations)
    return opened


def get_external_boundary_mask(binary) -> np.ndarray:
    binary = (binary > 0).astype(np.uint8)
    h, w = binary.shape

    # Find external black background
    external_bg = np.zeros((h, w), dtype=bool)

    stack = []

    for x in range(w):
        if binary[0, x] == 0:
            stack.append((0, x))

        if binary[h - 1, x] == 0:
            stack.append((h - 1, x))

    for y in range(h):

        if binary[y, 0] == 0:
            stack.append((y, 0))

        if binary[y, w - 1] == 0:
            stack.append((y, w - 1))

    neighbors4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while stack:
        y, x = stack.pop()
        if y < 0 or y >= h or x < 0 or x >= w:
            continue

        if external_bg[y, x]:
            continue

        if binary[y, x] != 0:
            continue

        external_bg[y, x] = True
        for dy, dx in neighbors4:
            stack.append((y + dy, x + dx))

    # Keep only white pixels touching external background

    boundary = np.zeros((h, w), dtype=np.uint8)

    neighbors8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for y in range(h):
        for x in range(w):
            if binary[y, x] == 0:
                continue

            for dy, dx in neighbors8:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if external_bg[ny, nx]:
                        boundary[y, x] = 255
                        break
    return boundary


def main(
    binary_image: np.ndarray | None = None,
    show_images: bool = False,
    save_images: bool = False,
) -> np.ndarray:
    if binary_image is None:
        binary = load_grayscale_image(IMAGE_DIR / "processed" / "binary.png")
    else:
        binary = binary_image

    if show_images:
        show_img(binary)

    kernel = np.ones((4, 4), dtype=np.uint8)
    # binary should contain values 0 and 255
    cleaned = morphological_open(binary, kernel, iterations=2)
    if show_images:
        show_img(cleaned)

    if save_images:
        plt.imsave(IMAGE_DIR / "processed" / "cleaned.png", cleaned, cmap="gray")

    boundary = get_external_boundary_mask(cleaned)
    if show_images:
        show_img(boundary)

    if save_images:
        plt.imsave(IMAGE_DIR / "processed" / "boundary.png", boundary, cmap="gray")

    return boundary


if __name__ == "__main__":
    main(IMAGE_DIR / "processed" / "binary.png", show_images=True, save_images=True)
