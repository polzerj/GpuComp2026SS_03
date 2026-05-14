import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import filters
import numpy as np
from PIL import Image
from pathlib import Path
IMAGE_DIR = Path(__file__).parent.parent.parent/"data"


def load_grayscale_image(image_path):
    image = Image.open(image_path).convert("L")
    return np.array(image)


def show_img(img, width=10):
    plt.figure(figsize=(width, width / 1000 * 727))
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.show()


def create_gaussian_kernel(size=8, sigma=1.0):
    """
    Create a Gaussian kernel matrix.
    """
    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)
    return kernel


def gaussian_filter(image, kernel_size=5, sigma=1.0):
    """
    Apply Gaussian Blur manually using convolution.
    """
    # Create Gaussian kernel
    kernel = create_gaussian_kernel(kernel_size, sigma)

    # Image dimensions
    height, width = image.shape

    # Padding size
    pad = kernel_size // 2

    # Pad image borders
    padded = np.pad(image, pad_width=pad, mode="constant")

    # Output image
    blurred = np.zeros_like(image, dtype=np.float32)

    # Convolution
    for y in range(height):
        for x in range(width):
            # Extract local region
            region = padded[y : y + kernel_size, x : x + kernel_size]
            # Apply kernel
            blurred[y, x] = np.sum(region * kernel)
    return blurred.astype(np.uint8)


def otsu_threshold(image):
    hist, _ = np.histogram(image.ravel(), bins=256, range=(0, 256))
    total_pixels = image.size
    sum_total = np.dot(np.arange(256), hist)

    sum_background = 0
    weight_background = 0
    max_variance = 0
    threshold = 0

    for t in range(256):
        weight_background += hist[t]
        if weight_background == 0:
            continue

        weight_foreground = total_pixels - weight_background
        if weight_foreground == 0:
            break

        sum_background += t * hist[t]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        variance_between = (
            weight_background
            * weight_foreground
            * (mean_background - mean_foreground) ** 2
        )

        if variance_between > max_variance:
            max_variance = variance_between
            threshold = t
    binary = image > threshold
    return binary.astype(np.uint8) * 255, threshold


image = load_grayscale_image(IMAGE_DIR + "/single.JPG")

show_img(image)

blurred = gaussian_filter(image, sigma=1)
plt.imsave(IMAGE_DIR + "/processed/blurred.png", blurred, cmap="gray")
binary, threshold_value = otsu_threshold(blurred)

show_img(binary)
plt.imsave(IMAGE_DIR + "/processed/binary.png", binary, cmap="gray")
