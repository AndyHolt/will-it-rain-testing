import matplotlib.pyplot as plt

results = [
    ("Persistence", 0.631, "baseline"),
    ("Forecast: precipitation @ T >= 0.1mm", 0.693, "baseline"),
    ("Forecast: precipitation_probability @ T >= 50%", 0.700, "baseline"),
    ("LightGBM, F1-max threshold", 0.588, "model"),
    ("LightGBM, rank-based threshold", 0.672, "model"),
    ("LightGBM + isotonic calibration", 0.746, "model"),
]

labels = [r[0] for r in results]
scores = [r[1] for r in results]
colors = ["#888888" if r[2] == "baseline" else "#1f77b4" for r in results]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(labels, scores, color=colors)
ax.set_xlabel("Test F1 (rain class)")
ax.set_title("Rain prediction: baseline and model F1 progression")
ax.set_xlim(0, 1.0)
ax.invert_yaxis()
ax.axvline(max(scores), color="#aaaaaa", linestyle=":", linewidth=1)

for bar, score in zip(bars, scores):
    ax.text(score + 0.01, bar.get_y() + bar.get_height() / 2, f"{score:.3f}", va="center")

fig.tight_layout()
fig.savefig("results.png", dpi=150)
print("Wrote results.png")
