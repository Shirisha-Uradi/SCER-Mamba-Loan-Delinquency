import pandas as pd
import matplotlib.pyplot as plt

# Load final results
df = pd.read_csv(
    "output/final_results/final_sequential_model_comparison.csv"
)

# Short names for graph
df["Model"] = [
    "LSTM",
    "SCER Pre-Mamba",
    "SCER-Mamba V2",
    "Final SCER-Mamba"
]

# Focus on minority-class metrics
metrics = [
    "Precision",
    "Recall",
    "F1",
    "PR_AUC"
]

plot_df = df.set_index("Model")[metrics]

ax = plot_df.plot(
    kind="bar",
    figsize=(11, 6)
)

plt.title("Sequential Model Performance Comparison")
plt.ylabel("Score")
plt.xlabel("Model")
plt.ylim(0, 0.45)

plt.xticks(
    rotation=15,
    ha="right"
)

plt.legend(
    title="Metric"
)

plt.tight_layout()

plt.savefig(
    "output/final_results/final_model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "Saved: output/final_results/final_model_comparison.png"
)