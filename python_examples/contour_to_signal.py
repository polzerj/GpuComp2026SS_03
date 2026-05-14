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


# Clockwise neighbors
DIRECTIONS = [
    (0, 1),  # right
    (1, 1),  # down-right
    (1, 0),  # down
    (1, -1),  # down-left
    (0, -1),  # left
    (-1, -1),  # up-left
    (-1, 0),  # up
    (-1, 1),  # up-right
]


def trace_contour(boundary):
    h, w = boundary.shape
    ys, xs = np.where(boundary > 0)

    if len(xs) == 0:
        return []

    # Start point
    start = (ys[0], xs[0])
    contour = [start]
    current = start
    prev_dir = 0
    visited = set()
    visited.add(start)

    while True:
        found = False

        # Moore neighborhood tracing
        for i in range(8):
            dir_idx = (prev_dir + i) % 8
            dy, dx = DIRECTIONS[dir_idx]
            ny = current[0] + dy
            nx = current[1] + dx

            if 0 <= ny < h and 0 <= nx < w and boundary[ny, nx]:

                next_point = (ny, nx)
                if next_point == start and len(contour) > 10:
                    return contour

                if next_point not in visited:
                    contour.append(next_point)
                    visited.add(next_point)
                    current = next_point
                    prev_dir = (dir_idx + 5) % 8
                    found = True
                    break

        if not found:
            break

    return contour


def simplify_chain_approx(contour):

    if len(contour) <= 2:
        return contour

    simplified = [contour[0]]

    for i in range(1, len(contour) - 1):
        y0, x0 = contour[i - 1]
        y1, x1 = contour[i]
        y2, x2 = contour[i + 1]
        dx1 = np.sign(x1 - x0)
        dy1 = np.sign(y1 - y0)
        dx2 = np.sign(x2 - x1)
        dy2 = np.sign(y2 - y1)

        # Keep only corners/direction changes
        if (dx1, dy1) != (dx2, dy2):
            simplified.append(contour[i])

    simplified.append(contour[-1])
    return simplified


def find_contour_chain_approx_simple(boundary):

    # boundary = get_external_boundary(binary)
    contour = trace_contour(boundary)
    contour = simplify_chain_approx(contour)

    # Convert to OpenCV-style (x, y)
    contour = np.array([(x, y) for y, x in contour])
    return contour


def smooth_contour(points, window=15):
    pts = np.array(points, dtype=float)
    xs = pts[:, 0]
    ys = pts[:, 1]

    # Circular padding (important for closed contours)
    pad = window // 2
    xs_pad = np.pad(xs, (pad, pad), mode="wrap")
    ys_pad = np.pad(ys, (pad, pad), mode="wrap")
    kernel = np.ones(window) / window
    xs_smooth = np.convolve(xs_pad, kernel, mode="valid")
    ys_smooth = np.convolve(ys_pad, kernel, mode="valid")
    smooth_points = [
        (int(round(x)), int(round(y))) for x, y in zip(xs_smooth, ys_smooth)
    ]

    return smooth_points


def enclosing_circle_approx(points):

    pts = np.array(points, dtype=float)
    min_x = np.min(pts[:, 0])
    max_x = np.max(pts[:, 0])
    min_y = np.min(pts[:, 1])
    max_y = np.max(pts[:, 1])

    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2

    distances = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
    radius = np.max(distances)
    return (cx, cy), radius


def radial_signal(points, center):
    cx, cy = center
    signal = []
    for x, y in points:
        d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        signal.append(d)
    return signal


def find_triangular_peaks(
    signal, min_distance=10, min_prominence=10, smooth_window=21, min_sharpness=20
):
    signal = np.asarray(signal, dtype=float)

    # Smooth
    # kernel = np.ones(smooth_window) / smooth_window
    # s = np.convolve(signal, kernel, mode="same")

    kernel = np.ones(smooth_window) / smooth_window
    pad = smooth_window // 2
    signal_pad = np.pad(signal, (pad, pad), mode="wrap")
    s = np.convolve(signal_pad, kernel, mode="valid")

    # Detect ONLY local maxima
    peaks_idx, props = find_peaks(s, distance=min_distance, prominence=min_prominence)
    selected = []

    for i in peaks_idx:
        left = max(0, i - 20)
        right = min(len(s), i + 20)
        sharpness = s[i] - np.mean([s[left], s[right - 1]])

        if sharpness >= min_sharpness:
            selected.append(i)

    return list(selected), [signal[i] for i in selected], s


boundary = load_grayscale_image(IMAGE_DIR / "processed" / "boundary.png")
show_img(boundary)

contour = find_contour_chain_approx_simple(boundary)
points = [(int(x), int(y)) for x, y in contour]

smooth_points = smooth_contour(points, window=5)
center, radius = enclosing_circle_approx(smooth_points)
print("center:", center)
print("radius", radius)

fig, ax = plt.subplots(figsize=(10, 10 / 1000 * 727))

display_boundary = boundary.copy()

if display_boundary.max() <= 1:

    display_boundary = display_boundary * 255

ax.imshow(display_boundary, cmap="gray", vmin=0, vmax=255)

ax.axis("off")

# Draw contour
xs = [p[0] for p in points]
ys = [p[1] for p in points]

ax.plot(xs, ys, linewidth=1)

# Draw smoothed contour
xs_s = [p[0] for p in smooth_points]
ys_s = [p[1] for p in smooth_points]
ax.plot(xs_s, ys_s, linewidth=2)

# Draw enclosing circle
circle = plt.Circle(center, radius, fill=False, linewidth=2, color="green")
ax.add_patch(circle)

# Draw center
ax.scatter(center[0], center[1], s=30)
plt.savefig(
    IMAGE_DIR / "processed" / "contour_circle.png",
    bbox_inches="tight",
    pad_inches=0,
    dpi=300,
)
plt.show()

signal = radial_signal(smooth_points, center)

plt.figure(figsize=(12, 4))
plt.plot(signal)
plt.title("Segnale radiale del contorno")
plt.xlabel("Indice punto contorno")
plt.ylabel("Distanza dal centro")
plt.grid(True)
plt.show()

peaks_idx, peaks_val, smooth = find_triangular_peaks(
    signal, min_distance=5, min_prominence=5, smooth_window=2, min_sharpness=20
)

peaks_idx = [x for x in peaks_idx if x != 1]
peaks_val = [signal[i] for i in peaks_idx]

# Remove first peak
# peaks_idx = peaks_idx[1:]
# Recompute values
# peaks_val = [signal[i] for i in peaks_idx]

plt.figure(figsize=(14, 5))
plt.plot(signal)

plt.scatter(peaks_idx, peaks_val, marker="o")

plt.grid(True)
plt.show()


fig, ax = plt.subplots(figsize=(10, 10 / 1000 * 727))
ax.imshow(display_boundary, cmap="gray", vmin=0, vmax=255)
ax.axis("off")

# Contorno smooth
ax.plot(xs_s, ys_s, linewidth=2, color="orange")

# Cerchio
circle = plt.Circle(center, radius, fill=False, linewidth=2, color="green")
ax.add_patch(circle)

# Centro
ax.scatter(center[0], center[1], s=30, color="blue")

# Picchi sul contorno
peak_points = [smooth_points[i] for i in peaks_idx]
px = [p[0] for p in peak_points]
py = [p[1] for p in peak_points]
ax.scatter(px, py, s=80, color="red")
plt.savefig(
    IMAGE_DIR / "processed" / "contour_circle_peaks.png",
    bbox_inches="tight",
    pad_inches=0,
    dpi=300,
)
plt.show()
