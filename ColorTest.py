import torch
import torch.nn.functional as F
from torchvision import transforms
from Modellen.DataImport import RealVsAIDataset


CSV_FILE = "data/test_labels.csv"
ROOT_DIR = "data/my_real_vs_ai_dataset/my_real_vs_ai_dataset"
SAMPLES_PER_CLASS = 100
SEED = 0
TOLERANCE = 5
# -------------------

# PILToTensor gives a uint8 tensor of shape (3, height, width), values 0-255
dataset = RealVsAIDataset(CSV_FILE, ROOT_DIR, transform=transforms.PILToTensor()) #Loader test csv
labels = torch.tensor(dataset.annotations.iloc[:, 2].astype(int).values) # Fordi datasættet er en form for liste af tuples bruger vi iloc
print("Label counts in CSV:", {0: int((labels == 0).sum()), 1: int((labels == 1).sum())})

# ---- Pick random images from each class ----
gen = torch.Generator().manual_seed(SEED)
chosen = []
for lab in (0, 1):
    idx = (labels == lab).nonzero().flatten() #Giver en liste / tensor med indekset for alle TRUE values (så én for hver class). Btw den bruger torch.nonzero()
    idx = idx[torch.randperm(len(idx), generator=gen)][:SAMPLES_PER_CLASS] #Finder de random 500 billeder
    chosen.append(idx)
order = torch.cat(chosen) #Laver de 2 tensorer til 1

# ---- One bundle of numbers per class ----
def new_stats():
    return {
        "n": 0,                                    # pixels seen
        "sum": torch.zeros(3, dtype=torch.long),   # running sum of R, G, B  (3 for 3 kanaler)
        "bright_sum": 0.0,
        "colours": torch.zeros(256 ** 3, dtype=torch.bool),  # which colours occurred
        "seen": set(),                             # (position, colour) pairs
        "repeats": 0, # colour already seen at that position
    }

stats = {0: new_stats(), 1: new_stats()}
brightness_weights = torch.tensor([0.299, 0.587, 0.114])
image_shape = None


# ---- Go through every pixel of every chosen image ----
for count, i in enumerate(order.tolist(), start=1): #Enumerate tager bare hvert element i en liste/tensor
    image, label = dataset[i]                      # image: (3, H, W)

    s = stats[label]
    pixels = image.permute(1, 2, 0).reshape(-1, 3).long()   # (num_pixels, 3) (reshape er basically flatten())

    # Colour as one number: R * 65536 + G * 256 + B
    code = pixels[:, 0] * 65536 + pixels[:, 1] * 256 + pixels[:, 2]

    # Basic stats
    s["n"] += len(pixels)
    s["sum"] += pixels.sum(dim=0)

    # Brightness
    b = pixels.float() @ brightness_weights #Ganger 2 tensorer
    s["bright_sum"] += b.sum().item()

    # Remember which colours occurred
    s["colours"][code] = True

    # Has this exact colour been at this exact position before?
    keys = (torch.arange(len(pixels)) * 16777216 + code).tolist()
    s["repeats"] += sum(k in s["seen"] for k in keys)
    s["seen"].update(keys)

    if count % 10 == 0:
        print(f"Processed {count} / {len(order)} images")

# ---- Summary ----
names = {0: "Class 0 (AI)", 1: "Class 1 (Real)"}
for label in (0, 1):
    s = stats[label]
    print(f"\n=== {names[label]} ===")
    mean = s["sum"] / s["n"] #<-- mean = tensor([mean_R, mean_G, mean_B])
    print(f"Pixels: {s['n']:,}")
    for c, ch in enumerate("RGB"): #<-- enumerate("RGB") = (0, "R"), (1, "G"), (2, "B") hvilket passer til index i mean[]
        print(f"{ch}: mean {mean[c]:.2f}")
    print(f"Brightness: mean {s['bright_sum'] / s['n']:.2f}")
    print(f"Distinct colours: {int(s['colours'].sum()):,}")
    print(f"Pixels with a colour already seen at that position: "
          f"{s['repeats']:,} ({100 * s['repeats'] / s['n']:.2f}%)")

# ---- Colours unique to one class (with tolerance) ----
def grow(colours, tol):
    """Mark every colour within +/- tol of a colour that occurred."""
    cube = colours.reshape(1, 1, 256, 256, 256).float()   # one axis each for R, G, B
    width = 2 * tol + 1
    for dim in range(3):                                   # grow along R, then G, then B
        kernel = [1, 1, 1]
        pad = [0, 0, 0]
        kernel[dim] = width
        pad[dim] = tol
        cube = F.max_pool3d(cube, tuple(kernel), stride=1, padding=tuple(pad))
    return cube.reshape(-1) > 0

only_ai = stats[0]["colours"] & ~grow(stats[1]["colours"], TOLERANCE)
only_real = stats[1]["colours"] & ~grow(stats[0]["colours"], TOLERANCE)

print(f"\nUnique colours with tolerance +/-{TOLERANCE}:")
print(f"Colours only in class 0 (AI): {int(only_ai.sum()):,}")
print(f"Colours only in class 1 (Real): {int(only_real.sum()):,}")

