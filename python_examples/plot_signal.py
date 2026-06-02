import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

csv_path = Path("../data/processed/contour_signal.csv")
df = pd.read_csv(csv_path)
labels = df["label"].unique()
print("labels:", df["label"].unique())
print("num labels:", df["label"].nunique())

# Pick the first piece
label = labels[0]
piece = df[df["label"] == label]#.sort_values("rank")

plt.figure(figsize=(12, 4))
plt.plot(piece["rank"], piece["distance"])
plt.title(f"Contour distance signal for label {label}")
plt.xlabel("Contour rank")
plt.ylabel("Distance to centroid")
plt.grid(True)
plt.tight_layout()
plt.savefig('signals.png', dpi=300, bbox_inches='tight')
plt.show()
