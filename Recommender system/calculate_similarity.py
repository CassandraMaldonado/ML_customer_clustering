import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("shopping_behavior_updated.csv")

# Ensure required columns exist
required_columns = {"Customer ID", "Item Purchased", "Color", "Season"}
missing_cols = required_columns - set(df.columns)
if missing_cols:
    raise ValueError(f"Missing columns in dataset: {missing_cols}")

# Create a unique feature for recommendation
df["Item Key"] = df["Item Purchased"] + " - " + df["Color"] + " - " + df["Season"]

# Create user-item interaction matrix
user_item = df.groupby(['Customer ID', 'Item Key']).size().unstack(fill_value=0)
user_item_binary = (user_item > 0).astype(int)

# Compute item similarity matrix
X = user_item_binary.values
cooccurrence = X.T.dot(X)  

item_counts = np.diag(cooccurrence).astype(float)
norm = np.outer(np.sqrt(item_counts), np.sqrt(item_counts))

similarity_matrix = np.divide(cooccurrence, norm, out=np.zeros_like(cooccurrence, dtype=float), where=norm != 0)
np.fill_diagonal(similarity_matrix, 0)

# Create index mappings
items = list(user_item_binary.columns)
item_to_index = {item: idx for idx, item in enumerate(items)}
index_to_item = {idx: item for item, idx in item_to_index.items()}

# Save similarity matrix to CSV
similarity_df = pd.DataFrame(similarity_matrix, index=items, columns=items)
similarity_df.to_csv("precomputed_similarity.csv")

# Save user-item matrix for reference
user_item_binary.to_csv("user_item_matrix.csv")

print("✅ Precomputed similarity saved as 'precomputed_similarity.csv'")
print("✅ User-item matrix saved as 'user_item_matrix.csv'")
