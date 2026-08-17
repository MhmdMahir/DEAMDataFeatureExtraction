import torch
import librosa

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

AUDIO_PATH = "../data/deam/mp3/2.mp3"


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("============================================")
print("DEAM Music2Vec Test")
print("============================================")
print("Device:", device)


# ============================================================
# Load processor
# ============================================================

print("\nLoading processor...")

processor = Wav2Vec2Processor.from_pretrained(
    PROCESSOR_NAME
)

print("Processor loaded.")


# ============================================================
# Load a VALID Data2Vec configuration
#
# The Music2Vec repository currently contains a malformed
# vocab_size value in config.json.
#
# We use the original Data2Vec Audio configuration instead.
# ============================================================

print("\nLoading Data2Vec configuration...")

config = Data2VecAudioConfig.from_pretrained(
    PROCESSOR_NAME
)

# Make sure vocab_size is an integer
config.vocab_size = 32

print("Configuration loaded.")
print("vocab_size:", config.vocab_size)
print("hidden_size:", config.hidden_size)
print("num_hidden_layers:", config.num_hidden_layers)


# ============================================================
# Load Music2Vec weights
# ============================================================

print("\nLoading Music2Vec model...")

model = Data2VecAudioModel.from_pretrained(
    MODEL_NAME,
    config=config
)

model = model.to(device)
model.eval()

print("Music2Vec model loaded.")


# ============================================================
# Load audio
# ============================================================

print("\nLoading audio:")
print(AUDIO_PATH)

audio, sr = librosa.load(
    AUDIO_PATH,
    sr=16000,
    mono=True
)

print("Sample rate:", sr)
print("Audio samples:", len(audio))
print("Audio duration:", len(audio) / sr, "seconds")


# ============================================================
# Prepare audio
# ============================================================

print("\nPreparing audio...")

inputs = processor(
    audio,
    sampling_rate=16000,
    return_tensors="pt"
)

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}


# ============================================================
# Run Music2Vec
# ============================================================

print("\nRunning Music2Vec...")

with torch.no_grad():

    outputs = model(
        **inputs,
        output_hidden_states=True
    )


# ============================================================
# Extract representations
# ============================================================

hidden_states = torch.stack(
    outputs.hidden_states
).squeeze(1)


print("\n============================================")
print("Music2Vec Results")
print("============================================")

print(
    "All layers shape:",
    hidden_states.shape
)


# ============================================================
# Temporal mean pooling
#
# [13, time, 768]
#        ↓
# [13, 768]
# ============================================================

features_13x768 = hidden_states.mean(dim=1)

print(
    "After temporal pooling:",
    features_13x768.shape
)


# ============================================================
# Average the 13 layers
#
# [13, 768]
#      ↓
# [768]
# ============================================================

embedding_768 = features_13x768.mean(dim=0)

print(
    "Final 768-dimensional embedding:",
    embedding_768.shape
)


# ============================================================
# Display values
# ============================================================

print("\nFirst 10 embedding values:")

print(
    embedding_768[:10].cpu().numpy()
)


print("\n============================================")
print("SUCCESS")
print("============================================")
