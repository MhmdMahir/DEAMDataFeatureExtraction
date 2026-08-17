import os
import json
import numpy as np
import torch
import librosa

from tqdm import tqdm
from transformers import (
    Wav2Vec2Processor,
    Data2VecAudioModel,
    Data2VecAudioConfig
)


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "m-a-p/music2vec-v1"
PROCESSOR_NAME = "facebook/data2vec-audio-base-960h"

INPUT_DIR = "data/deam/mp3"

OUTPUT_DIR = "features/music2vec"

JSON_OUTPUT = "features/music2vec_embeddings.json"
NPY_OUTPUT = "features/music2vec_embeddings.npy"


# ============================================================
# Create output directory
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("============================================")
print("DEAM Music2Vec Feature Extraction")
print("============================================")
print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# Load processor
# ============================================================

print("\nLoading processor...")

processor = Wav2Vec2Processor.from_pretrained(
    PROCESSOR_NAME
)


# ============================================================
# Load configuration
# ============================================================

print("Loading Data2Vec configuration...")

config = Data2VecAudioConfig.from_pretrained(
    PROCESSOR_NAME
)

# Required because the Music2Vec repository has an invalid
# vocab_size value in its config.
config.vocab_size = 32


# ============================================================
# Load Music2Vec
# ============================================================

print("Loading Music2Vec model...")

model = Data2VecAudioModel.from_pretrained(
    MODEL_NAME,
    config=config
)

model = model.to(device)
model.eval()

print("Music2Vec model loaded.")


# ============================================================
# Find audio files
# ============================================================

audio_files = sorted(
    [
        f
        for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(
            (".mp3", ".wav", ".flac")
        )
    ],
    key=lambda x: int(os.path.splitext(x)[0])
    if os.path.splitext(x)[0].isdigit()
    else x
)

print("\nAudio files found:", len(audio_files))


# ============================================================
# Storage
#
# Format:
#
# [
#     [id, [768 embedding values]],
#     [id, [768 embedding values]],
#     ...
# ]
# ============================================================

all_embeddings = []


# ============================================================
# Process audio
# ============================================================

for index, filename in enumerate(
    tqdm(audio_files, desc="Extracting Music2Vec")
):

    # --------------------------------------------------------
    # Extract ID from filename
    # --------------------------------------------------------

    file_id = os.path.splitext(filename)[0]

    try:
        file_id = int(file_id)
    except ValueError:
        pass


    # --------------------------------------------------------
    # Input path
    # --------------------------------------------------------

    input_path = os.path.join(
        INPUT_DIR,
        filename
    )


    try:

        # ----------------------------------------------------
        # Load audio
        # ----------------------------------------------------

        audio, sr = librosa.load(
            input_path,
            sr=16000,
            mono=True
        )


        # ----------------------------------------------------
        # Prepare audio
        # ----------------------------------------------------

        inputs = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }


        # ----------------------------------------------------
        # Music2Vec inference
        # ----------------------------------------------------

        with torch.no_grad():

            outputs = model(
                **inputs,
                output_hidden_states=True
            )


        # ----------------------------------------------------
        # Shape:
        #
        # 13 × time × 768
        # ----------------------------------------------------

        hidden_states = torch.stack(
            outputs.hidden_states
        ).squeeze(1)


        # ----------------------------------------------------
        # Temporal pooling:
        #
        # 13 × time × 768
        #             ↓
        # 13 × 768
        # ----------------------------------------------------

        features_13x768 = hidden_states.mean(
            dim=1
        )


        # ----------------------------------------------------
        # Layer pooling:
        #
        # 13 × 768
        #       ↓
        # 768
        # ----------------------------------------------------

        embedding_768 = features_13x768.mean(
            dim=0
        )


        # ----------------------------------------------------
        # Convert to Python list
        # ----------------------------------------------------

        embedding = (
            embedding_768
            .cpu()
            .numpy()
            .astype(np.float32)
            .tolist()
        )


        # ----------------------------------------------------
        # Store:
        #
        # [id, [768 values]]
        # ----------------------------------------------------

        all_embeddings.append(
            [
                file_id,
                embedding
            ]
        )


    except Exception as e:

        print(
            f"\nERROR processing {filename}:"
        )

        print(e)


# ============================================================
# Save JSON
# ============================================================

print("\nSaving JSON...")

with open(
    JSON_OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        all_embeddings,
        f
    )


# ============================================================
# Save NumPy
# ============================================================

print("Saving NumPy file...")

ids = np.array(
    [
        item[0]
        for item in all_embeddings
    ]
)

embeddings = np.array(
    [
        item[1]
        for item in all_embeddings
    ],
    dtype=np.float32
)

np.save(
    NPY_OUTPUT,
    {
        "ids": ids,
        "embeddings": embeddings
    },
    allow_pickle=True
)


# ============================================================
# Final information
# ============================================================

print("\n============================================")
print("EXTRACTION COMPLETE")
print("============================================")

print(
    "Number of embeddings:",
    len(all_embeddings)
)

print(
    "Embedding matrix shape:",
    embeddings.shape
)

print(
    "JSON:",
    JSON_OUTPUT
)

print(
    "NumPy:",
    NPY_OUTPUT
)

if len(all_embeddings) > 0:

    print("\nFirst entry:")

    print(
        "ID:",
        all_embeddings[0][0]
    )

    print(
        "Embedding dimensions:",
        len(all_embeddings[0][1])
    )
