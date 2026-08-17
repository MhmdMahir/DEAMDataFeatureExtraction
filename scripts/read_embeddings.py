import numpy as np

FILE = "features/music2vec_embeddings.npy"

data = np.load(
    FILE,
    allow_pickle=True
).item()

ids = data["ids"]
embeddings = data["embeddings"]

print("============================================")
print("Music2Vec Embeddings")
print("============================================")

print("Number of IDs:", len(ids))
print("Embedding shape:", embeddings.shape)

print("\nFirst  10 ID:")
print(ids[0 : 10])

print("\nFirst 10 embedding:")
print(embeddings[0 : 10])

print("\nFirst embedding dimensions:")
print(len(embeddings[0]))

print("\nFirst 10 values:")
print(embeddings[0][:10])
