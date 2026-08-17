import os
import numpy as np
import torch
import librosa

from tqdm import tqdm
from transformers import (
    Wav2Vec2Processor,
    Data2VecAudioModel
)


# ==================================================
# Configuration
# ==================================================

MODEL_NAME = "m-a-p/music2vec-v1"

INPUT_DIR = "data/deam/mp3"
OUTPUT_DIR = "features/music2vec"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==================================================
# Device
# ==================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ==================================================
# Load Music2Vec
# ==================================================

print("Loading Music2Vec...")

processor = Wav2Vec2Processor.from_pretrained(
    MODEL_NAME
)

model = Data2VecAudioModel.from_pretrained(
    MODEL_NAME
)

model = model.to(device)
model.eval()

print("Model loaded.")


# ==================================================
# Find audio files
# ==================================================

audio_files = [
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith((".mp3", ".wav"))
]

audio_files.sort()

print("Number of files:", len(audio_files))


# ==================================================
# Process files
# ==================================================

for filename in tqdm(audio_files):

    input_path = os.path.join(
        INPUT_DIR,
        filename
    )

    output_name = os.path.splitext(filename)[0] + ".npy"

    output_path = os.path.join(
        OUTPUT_DIR,
        output_name
    )

    # Skip already processed files
    if os.path.exists(output_path):
        continue

    try:

        # ------------------------------------------
        # Load audio
        # ------------------------------------------

        audio, sr = librosa.load(
            input_path,
            sr=16000,
            mono=True
        )

        # ------------------------------------------
        # Processor
        # ------------------------------------------

        inputs = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        # ------------------------------------------
        # Music2Vec
        # ------------------------------------------

        with torch.no_grad():

            outputs = model(
                **inputs,
                output_hidden_states=True
            )

        # ------------------------------------------
        # 13 × time × 768
        # ------------------------------------------

        hidden_states = torch.stack(
            outputs.hidden_states
        ).squeeze(1)

        # ------------------------------------------
        # Average over time
        #
        # 13 × time × 768
        #          ↓
        # 13 × 768
        # ------------------------------------------

        features = hidden_states.mean(dim=1)

        # ------------------------------------------
        # Average layers
        #
        # 13 × 768
        #          ↓
        # 768
        # ------------------------------------------

        embedding_768 = features.mean(dim=0)

        # ------------------------------------------
        # Save
        # ------------------------------------------

        embedding_768 = (
            embedding_768
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        np.save(
            output_path,
            embedding_768
        )

    except Exception as e:

        print(
            f"\nERROR processing {filename}: {e}"
        )


print("\nFinished.")
