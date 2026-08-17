import torch
import librosa

from transformers import Wav2Vec2Processor, Data2VecAudioModel


MODEL_NAME = "m-a-p/music2vec-v1"

audio_path = "data/deam/mp3/2.mp3"


# --------------------------------------------------
# Load model
# --------------------------------------------------

processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Data2VecAudioModel.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()


# --------------------------------------------------
# Load audio
# --------------------------------------------------

audio, sr = librosa.load(
    audio_path,
    sr=16000,
    mono=True
)


# --------------------------------------------------
# Prepare input
# --------------------------------------------------

inputs = processor(
    audio,
    sampling_rate=16000,
    return_tensors="pt"
)

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}


# --------------------------------------------------
# Extract Music2Vec features
# --------------------------------------------------

with torch.no_grad():

    outputs = model(
        **inputs,
        output_hidden_states=True
    )


# --------------------------------------------------
# Get all 13 layers
# --------------------------------------------------

hidden_states = torch.stack(
    outputs.hidden_states
)

print("Raw shape:")
print(hidden_states.shape)
