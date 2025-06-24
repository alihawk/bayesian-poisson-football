import pandas as pd
import matplotlib.pyplot as plt

# Sample data
data = {
    "zone_id": [101, 102, 103],
    "time_bin": [8, 17, 23],
    "speed_mean": [1.2, 5.8, 0.5],
    "speed_median": [1.1, 6.0, 0.4],
    "speed_min": [0.8, 4.5, 0.2],
    "speed_max": [1.6, 7.2, 0.9],
    "speed_var": [0.05, 0.30, 0.02],
    "speed_q25": [1.0, 5.0, 0.3],
    "speed_q75": [1.3, 6.5, 0.7],
    "ping_count": [250, 180, 75],
    "unique_devs": [50, 30, 10],
    "pings_per_dev": [250/50, 180/30, 75/10],
    "dev_entropy": [4.2, 3.5, 1.8],
    "dwell_mean": [300, 120, 15],
    "dwell_median": [280, 100, 0],
    "dwell_min": [60, 30, 0],
    "dwell_max": [600, 240, 30],
    "entries_count": [40, 25, 5],
    "exits_count": [35, 20, 4],
    "entries_mean_speed": [1.3, 5.5, 0.6],
    "exits_mean_speed": [1.2, 6.0, 0.5],
    "trans_entropy": [2.1, 1.7, 0.5],
    "is_morning_commute": [1, 0, 0],
    "is_evening_commute": [0, 1, 0],
    "is_late_night": [0, 0, 1],
    "prior_walk": [0.5, 0.5, 0.5],
    "prior_car": [0.5, 0.5, 0.5]
}

# Create DataFrame and transpose
df = pd.DataFrame(data).set_index("zone_id").T

# Plot as an SVG table
fig, ax = plt.subplots(figsize=(8, 10))
ax.axis('off')
table = ax.table(
    cellText=df.values,
    rowLabels=df.index,
    colLabels=df.columns,
    cellLoc='center',
    rowLoc='center',
    colLoc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.2)

# Save to SVG file
fig.savefig("feature_matrix.svg", format="svg", bbox_inches="tight")
plt.close(fig)
