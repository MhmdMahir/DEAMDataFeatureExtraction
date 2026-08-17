import numpy as np
import json
from sklearn.decomposition import PCA

NPY_FILE = "features/music2vec_embeddings.npy"
JSON_FILE = "features/music2vec_embeddings_13.json"

# Load 768D Music2Vec embeddings
data = np.load(
    NPY_FILE,
    allow_pickle=True
).item()

ids = data["ids"]
embeddings = data["embeddings"]

print("Original:", embeddings.shape)

# 768 → 13
pca = PCA(n_components=13)

embeddings_13 = pca.fit_transform(
    embeddings
)

print("Reduced:", embeddings_13.shape)

# [id, [13 values]]
dataset = [
    [
        int(song_id),
        embedding.tolist()
    ]
    for song_id, embedding
    in zip(ids, embeddings_13)
]

# Save JSON
with open(JSON_FILE, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2)

print("Saved:", JSON_FILE)

print("Number of songs:", len(dataset))
print("Dimensions:", len(dataset[0][1]))

print(
    "Explained variance:",
    pca.explained_variance_ratio_.sum()
)
