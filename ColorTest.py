
import torch
import torch.nn.functional as F
from torchvision import transforms
from Modellen.DataImport import RealVsAIDataset

import matplotlib.pyplot as plt
import numpy as np


# ---------------- SETTINGS ----------------

CSV_FILE = "data/train_labels.csv"
ROOT_DIR = "data/my_real_vs_ai_dataset/my_real_vs_ai_dataset"

SAMPLES_PER_CLASS = 7000
SEED = 0
TOLERANCE = 0

# Maximum number of unique colours shown in RGB triangle
MAX_TRIANGLE_POINTS = 15000


# ---------------- LOAD DATASET ----------------

dataset = RealVsAIDataset(
    CSV_FILE,
    ROOT_DIR,
    transform=transforms.PILToTensor()
)

labels = torch.tensor(dataset.annotations.iloc[:, 2].values)

print("Label counts in CSV:", {
    0: int((labels == 0).sum()),
    1: int((labels == 1).sum())
})


# ---------------- RANDOM SAMPLE ----------------

gen = torch.Generator().manual_seed(SEED)

chosen = []

for lab in (0, 1):

    idx = (labels == lab).nonzero().flatten()

    if len(idx) < SAMPLES_PER_CLASS:
        raise ValueError(
            f"Not enough images in class {lab}. "
            f"Found {len(idx)}."
        )

    idx = idx[
        torch.randperm(len(idx), generator=gen)[:SAMPLES_PER_CLASS]
    ]

    chosen.append(idx)

order = torch.cat(chosen)


# ---------------- INITIALIZE STATISTICS ----------------

def new_stats():
    return {
        "n": 0,

        # RGB sums
        "sum": torch.zeros(3, dtype=torch.long),

        # Brightness
        "bright_sum": 0.0,

        # Unique colours
        "colours": torch.zeros(256 ** 3, dtype=torch.bool),

        # RGB histograms
        "hist_rgb": torch.zeros((3, 256), dtype=torch.long),

        # Brightness histogram
        "hist_brightness": torch.zeros(256, dtype=torch.long),

        # 2D colour histograms
        "hist_rg": torch.zeros((256, 256), dtype=torch.long),
        "hist_gb": torch.zeros((256, 256), dtype=torch.long),
        "hist_rb": torch.zeros((256, 256), dtype=torch.long),
    }


stats = {
    0: new_stats(),
    1: new_stats()
}

brightness_weights = torch.tensor([0.299, 0.587, 0.114])


# ---------------- PROCESS IMAGES ----------------

for count, i in enumerate(order.tolist(), start=1):

    image, label = dataset[i]

    s = stats[int(label)]

    # Convert image to list of RGB pixels
    pixels = image.permute(1, 2, 0).reshape(-1, 3).long()

    r = pixels[:, 0]
    g = pixels[:, 1]
    b = pixels[:, 2]

    # RGB colour as one number
    code = r * 65536 + g * 256 + b

    # Number of pixels
    s["n"] += len(pixels)

    # RGB sums
    s["sum"] += pixels.sum(dim=0)

    # Brightness
    brightness = pixels.float() @ brightness_weights

    s["bright_sum"] += brightness.sum().item()

    # Unique colours
    s["colours"][code] = True

    # RGB histograms
    for channel in range(3):

        s["hist_rgb"][channel] += torch.bincount(
            pixels[:, channel],
            minlength=256
        )

    # Brightness histogram
    brightness_int = brightness.round().long().clamp(0, 255)

    s["hist_brightness"] += torch.bincount(
        brightness_int,
        minlength=256
    )

    # Red vs Green
    s["hist_rg"] += torch.bincount(
        r * 256 + g,
        minlength=256 ** 2
    ).reshape(256, 256)

    # Green vs Blue
    s["hist_gb"] += torch.bincount(
        g * 256 + b,
        minlength=256 ** 2
    ).reshape(256, 256)

    # Red vs Blue
    s["hist_rb"] += torch.bincount(
        r * 256 + b,
        minlength=256 ** 2
    ).reshape(256, 256)

    if count % 100 == 0:
        print(f"Processed {count} / {len(order)} images")


# ---------------- SUMMARY ----------------

names = {
    0: "AI",
    1: "Real"
}

for label in (0, 1):

    s = stats[label]

    print(f"\n=== {names[label]} ===")

    mean = s["sum"].float() / s["n"]

    print(f"Pixels: {s['n']:,}")

    for c, ch in enumerate("RGB"):
        print(f"{ch}: mean {mean[c]:.2f}")

    print(f"Brightness: {s['bright_sum'] / s['n']:.2f}")

    print(f"Distinct colours: {int(s['colours'].sum()):,}")


# ---------------- UNIQUE COLOURS ----------------

def grow(colours, tol):

    cube = colours.reshape(
        1, 1, 256, 256, 256
    ).float()

    width = 2 * tol + 1

    for dim in range(3):

        kernel = [1, 1, 1]
        pad = [0, 0, 0]

        kernel[dim] = width
        pad[dim] = tol

        cube = F.max_pool3d(
            cube,
            tuple(kernel),
            stride=1,
            padding=tuple(pad)
        )

    return cube.reshape(-1) > 0


only_ai = (
    stats[0]["colours"]
    & ~grow(stats[1]["colours"], TOLERANCE)
)

only_real = (
    stats[1]["colours"]
    & ~grow(stats[0]["colours"], TOLERANCE)
)

print(f"\nUnique colours with tolerance +/-{TOLERANCE}:")
print(f"Colours only in AI: {int(only_ai.sum()):,}")
print(f"Colours only in Real: {int(only_real.sum()):,}")


# ==================================================
#                  FIGURE 1
#                RGB HISTOGRAMS
# ==================================================

def plot_rgb_histograms():

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    colors = ["red", "green", "blue"]

    for i, ax in enumerate(axes):

        for label, name, color in [
            (0, "AI", "black"),
            (1, "Real", colors[i])
        ]:

            hist = stats[label]["hist_rgb"][i].numpy()

            hist = hist / hist.sum()

            ax.plot(
                np.arange(256),
                hist,
                color=color,
                label=name
            )

        ax.set_title(f"{colors[i].upper()} distribution")
        ax.set_xlabel("Pixel intensity (0-255)")
        ax.set_ylabel("Proportion of pixels")

        ax.legend()
        ax.grid(alpha=0.2)

    plt.suptitle("RGB channel distributions")
    plt.tight_layout()
    plt.show()


# ==================================================
#                  FIGURE 2
#               BRIGHTNESS
# ==================================================

def plot_brightness():

    plt.figure(figsize=(10, 5))

    for label, name, color in [
        (0, "AI", "red"),
        (1, "Real", "blue")
    ]:

        hist = stats[label]["hist_brightness"].numpy()

        hist = hist / hist.sum()

        plt.plot(
            np.arange(256),
            hist,
            label=name,
            color=color
        )

    plt.title("Brightness distribution")
    plt.xlabel("Brightness (0-255)")
    plt.ylabel("Proportion of pixels")

    plt.legend()
    plt.grid(alpha=0.2)

    plt.tight_layout()
    plt.show()


# ==================================================
#                  FIGURE 3
#                 RGB TRIANGLE
# ==================================================

def plot_rgb_triangle():

    fig, ax = plt.subplots(figsize=(9, 8))

    rng = np.random.default_rng(SEED)

    for label, name, color in [
        (0, "AI", "red"),
        (1, "Real", "blue")
    ]:

        unique = stats[label]["colours"].numpy()

        codes = np.flatnonzero(unique)

        if len(codes) > MAX_TRIANGLE_POINTS:

            codes = rng.choice(
                codes,
                MAX_TRIANGLE_POINTS,
                replace=False
            )

        r = codes // 65536
        g = (codes // 256) % 256
        b = codes % 256

        total = r + g + b

        valid = total > 0

        r = r[valid] / total[valid]
        g = g[valid] / total[valid]
        b = b[valid] / total[valid]

        # RGB to 2D triangle coordinates
        x = g + b * 0.5
        y = b * np.sqrt(3) / 2

        ax.scatter(
            x,
            y,
            s=3,
            alpha=0.15,
            color=color,
            label=name
        )

    ax.set_title("RGB chromaticity distribution")
    ax.set_xlabel("Chromaticity X")
    ax.set_ylabel("Chromaticity Y")

    ax.legend()
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.show()


# ==================================================
#                 RUN FIGURES
# ==================================================

print("\nGenerating figure 1...", flush=True)
plot_rgb_histograms()

print("\nGenerating figure 2...", flush=True)
plot_brightness()

print("\nGenerating figure 3...", flush=True)
plot_rgb_triangle()

print("\nAll figures generated!", flush=True)
