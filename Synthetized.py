import pandas as pd
from sdv.tabular import GaussianCopula

# Load file
df = pd.read_csv(list(uploaded.keys())[0])
print("Loaded dataset:", df.shape)

model = GaussianCopula()


print("Training GaussianCopula...")
model.fit(df)
print("Training completed!")

synthetic_df = model.sample(num_rows=len(df))
print("Synthetic dataset created:", synthetic_df.shape)

synthetic_df.to_csv("synthetic_dataset.csv", index=False)
files.download("synthetic_dataset.csv")

import seaborn as sns
import matplotlib.pyplot as plt

example_col = df.columns[0]

plt.figure(figsize=(12,4))
sns.histplot(df[example_col], color="crimson", alpha=0.5, stat="density", label="Real")
sns.histplot(synthetic_df[example_col], color="black", alpha=0.5, stat="density", label="Synthetic")
plt.title(f"Real vs Synthetic Distribution — {example_col}")
plt.legend()
plt.show()
