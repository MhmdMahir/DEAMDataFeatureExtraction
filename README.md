# DEAM Music2Vec Feature Extraction

This project extracts audio embeddings from the DEAM (Database for Emotional Analysis of Music) dataset using Music2Vec and reduces the resulting high-dimensional embeddings to exactly 13 dimensions using Principal Component Analysis (PCA).

The extracted features are intended for use in an emotion-aware music recommendation system, where audio representations can later be combined with DEAM's Valence (V) and Arousal (A) annotations.

--- 

## Overview
```
The overall pipeline is:

DEAM Audio
    ↓
Audio Preprocessing
    ↓
Music2Vec
    ↓
768-dimensional Audio Embedding
    ↓
PCA
    ↓
13-dimensional Audio Embedding
    ↓
Match with DEAM Valence / Arousal
    ↓
Emotion-aware Recommendation
    ↓
PPO / DQN / A2C / EMOGRP
```

--- 

### 1. Dataset: DEAM

The project uses the DEAM (Database for Emotional Analysis of Music) dataset.

DEAM contains music audio together with emotional annotations. The two primary emotional dimensions are:

Valence (V) — represents how positive or negative the emotion is.
Arousal (A) — represents the intensity or activation level of the emotion.

Conceptually, each song contains:

- Song ID
- Audio
- Valence
- Arousal


### 2. Audio Processing

The DEAM audio files are loaded and prepared for Music2Vec.

The audio is resampled to:
16,000 Hz

> The processed waveform is then passed to the Music2Vec model.

### 3. Music2Vec Feature Extraction

Music2Vec is used as the audio representation model.

Instead of manually designing traditional audio features such as:

- MFCC
- Spectral Centroid
- Spectral Rolloff
- Zero Crossing Rate


### 4. Temporal Pooling

Music2Vec produces representations across multiple time frames. However, the recommendation system requires one fixed-size representation for each song.

Therefore, temporal mean pooling is applied.

> This produces one 768-dimensional representation for each Music2Vec layer.

### 5. Final Music2Vec Embedding

The 13 layer representations are then averaged across layers:

13 × 768
    ↓
Mean across layers
    ↓
768


Therefore, every song receives one:

768-dimensional Music2Vec embedding

> The embeddings are stored in:
> features/music2vec_embeddings.npy


### 6. What is PCA?

Principal Component Analysis (PCA) is a dimensionality-reduction technique.

PCA transforms a dataset with many features into a smaller number of new features called principal components.

In this project:

Original feature space
```

768 dimensions
     │
     │ PCA
     ▼
13 principal components

```

### 7. Why Use PCA?

The project already has the Music2Vec embeddings:

1795 × 768


The goal is to obtain:

1795 × 13


without rerunning Music2Vec.

PCA provides a straightforward way to compress the existing 768-dimensional embeddings into exactly 13 dimensions.

The transformation is:

768D Music2Vec
       ↓
      PCA
       ↓
13D Audio Representation



### 8. Summary

This project converts raw DEAM music audio into compact, machine-learning-ready audio representations.

The complete process is:
 ```
DEAM Audio
    ↓
Audio Preprocessing
    ↓
Music2Vec
    ↓
768D Audio Embedding
    ↓
PCA
    ↓
13D Audio Embedding
    ↓
Match with DEAM V/A
    ↓
Emotion-aware Recommendation
    ↓
PPO / DQN / A2C / EMOGRP
```

In short:

Music2Vec learns a rich representation of the audio.
Temporal pooling converts the time-dependent representation into fixed-size layer representations.
Layer pooling produces one 768-dimensional embedding per song.
PCA reduces the 768-dimensional embedding to exactly 13 dimensions.
DEAM Valence/Arousal provides the emotional annotations.
The 13-dimensional audio representation and emotional annotations can be combined for downstream emotion-aware recommendation.
The original 768-dimensional embeddings are retained for future experiments.

The final 13-dimensional representation is therefore a compressed audio embedding, not a direct replacement for the DEAM Valence and Arousal labels.

```
Feature Extraction at a Glance
                    ┌──────────────────┐
                    │   DEAM Dataset   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  Audio @ 16 kHz  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │    Music2Vec     │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ 13 × Time × 768  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Temporal Pooling │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │    13 × 768      │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   Layer Pooling  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │      768D        │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │       PCA        │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │       13D        │
                    └────────┬─────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
      Audio Representation           DEAM V / A Labels
              │                             │
              └──────────────┬──────────────┘
                             ▼
                  Emotion-aware Dataset
                             │
                             ▼
                 Recommendation Models
                             │
                  ┌──────────┼──────────┐
                  ▼          ▼          ▼
                 PPO        DQN        A2C
                             │
                             ▼
                           EMOGRP
```

```
DeamFeatureExtraction/
│
├── data/
│   └── deam/
│       └── mp3/
│           ├── 2.mp3
│           ├── 3.mp3
│           └── ...
│
├── features/
│   ├── music2vec_embeddings.npy
│   └── music2vec_embeddings_13.json
│
├── scripts/
│   ├── test_music2vec.py
│   ├── extract_music2vec.py
│   └── npy_to_json.py
│
├── .gitignore
├── requirements.txt
└── README.md

```
---
---
# Data Attribution
This project uses data from the DEAM (MediaEval Database for Emotional Analysis of Music) dataset, developed by Anna Aljanaki, Yi-Hsuan Yang, and Mohammad Soleymani. <br>
[DEAM Dataset - University of Geneva](https://cvml.unige.ch/databases/DEAM/)

If you use this repository or its processed data, please also cite:

> Aljanaki, A., Yang, Y.-H., & Soleymani, M. (2017). Developing a benchmark for emotional analysis of music. PLOS ONE, 12(3), e0173392. https://doi.org/10.1371/journal.pone.0173392

## License and Usage

The original DEAM dataset is distributed under a Non-Commercial Creative Commons (BY-NC) license. Please refer to the original dataset's terms of use before redistributing or using the data, particularly for commercial purposes.<br>
This repository contains processed/derived data generated from the original DEAM dataset. It is not the original DEAM dataset.<br>
For the original dataset, annotations, audio, features, and complete terms of use, please refer to the official DEAM website and manual. <br>


## Author
**Mohamed Mahir**
> Project: Song Recommendation System Based on User Emotion
