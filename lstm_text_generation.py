"""
=============================================================
  LSTM Text Generation - Generative AI Engineer Task
  Dataset: Shakespeare's Works (Project Gutenberg)
=============================================================

HOW TO RUN:
  Step 1: pip install tensorflow numpy requests
  Step 2: python lstm_text_generation.py

Dataset Link:
  https://www.gutenberg.org/files/100/100-0.txt
  (Shakespeare's Complete Works - Public Domain)
=============================================================
"""

import os
import re
import sys
import time
import random
import pickle
import requests
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

# ─────────────────────────────────────────────
#  CONFIGURATION  (tumi hya values change karu shakta)
# ─────────────────────────────────────────────
DATASET_URL    = "https://www.gutenberg.org/files/100/100-0.txt"
DATASET_FILE   = "shakespeare.txt"
SEQ_LENGTH     = 40       # Input sequence madhye kiti characters
STEP           = 3        # Sequence extraction step
EMBEDDING_DIM  = 64       # Embedding layer size
LSTM_UNITS     = 128      # LSTM neurons
DROPOUT_RATE   = 0.2      # Overfitting rokanyasathi
BATCH_SIZE     = 256
EPOCHS         = 30       # Max training epochs (early stopping ahe)
VALIDATION_SPLIT = 0.1
MODEL_FILE     = "lstm_model.keras"
VOCAB_FILE     = "vocab.pkl"

# ─────────────────────────────────────────────
#  STEP 1: DATASET DOWNLOAD
# ─────────────────────────────────────────────
def download_dataset():
    """
    Shakespeare dataset download karto Project Gutenberg varun.
    Jara file aadheech asel tar re-download karnar nahi.
    """
    if os.path.exists(DATASET_FILE):
        print(f"[INFO] Dataset already exists: {DATASET_FILE}")
        return

    print(f"[INFO] Downloading dataset from:\n       {DATASET_URL}")
    try:
        response = requests.get(DATASET_URL, timeout=30)
        response.raise_for_status()
        with open(DATASET_FILE, "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"[INFO] Dataset saved as '{DATASET_FILE}'")
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        print("       Manually download from the URL and save as 'shakespeare.txt'")
        sys.exit(1)


# ─────────────────────────────────────────────
#  STEP 2: PREPROCESSING
# ─────────────────────────────────────────────
def load_and_preprocess(max_chars=200_000):
    """
    Text load karto, lowercase karto, punctuation kadto,
    aani character-level sequences tayyar karto.

    max_chars: Jasta text asel tar training slow hoil,
               mhanun first 200k characters vapru.
    """
    print("\n[STEP 2] Preprocessing ...")

    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Gutenberg header/footer kadto (Project Gutenberg boilerplate)
    # Text usually "THE SONNETS" pasun suru hoto
    start_marker = "THE SONNETS"
    end_marker   = "End of the Project Gutenberg"
    start_idx = raw_text.find(start_marker)
    end_idx   = raw_text.find(end_marker)
    if start_idx != -1:
        raw_text = raw_text[start_idx:end_idx if end_idx != -1 else None]

    # Lowercase + punctuation remove
    text = raw_text.lower()
    text = re.sub(r"[^a-z\s]", "", text)   # Only a-z aani spaces thevto
    text = re.sub(r"\s+", " ", text).strip()

    # Limit characters (speed sathi)
    text = text[:max_chars]
    print(f"       Total characters used : {len(text):,}")

    # Unique characters = vocabulary
    vocab = sorted(set(text))
    print(f"       Vocabulary size        : {len(vocab)} characters")

    # Character <-> Index mappings
    char2idx = {c: i for i, c in enumerate(vocab)}
    idx2char = {i: c for i, c in enumerate(vocab)}

    return text, vocab, char2idx, idx2char


def create_sequences(text, char2idx):
    """
    Input-Output pairs tayar karto character level var.
    Input  : SEQ_LENGTH characters cha sequence
    Output : Tyanantar cha next character
    """
    print("\n[STEP 3] Creating sequences ...")

    sequences = []
    next_chars = []

    for i in range(0, len(text) - SEQ_LENGTH, STEP):
        sequences.append(text[i: i + SEQ_LENGTH])
        next_chars.append(text[i + SEQ_LENGTH])

    print(f"       Total sequences created: {len(sequences):,}")

    # Vectorize: One-hot nahi, integer indices vaparto (Embedding layer ahe)
    X = np.array([[char2idx[c] for c in seq] for seq in sequences], dtype=np.int32)
    y = np.array([char2idx[c] for c in next_chars], dtype=np.int32)
    y_cat = to_categorical(y, num_classes=len(char2idx))

    print(f"       X shape: {X.shape}")
    print(f"       y shape: {y_cat.shape}")

    return X, y_cat


# ─────────────────────────────────────────────
#  STEP 3: MODEL DESIGN
# ─────────────────────────────────────────────
def build_model(vocab_size):
    """
    LSTM model build karto:
      - Embedding Layer   : Integer indices -> dense vectors
      - LSTM Layer 1      : Sequence patterns shikto
      - Dropout           : Overfitting rokto
      - LSTM Layer 2      : Deeper pattern learning (Bonus architecture)
      - Dropout
      - Dense (Softmax)   : Next character probability
    """
    print("\n[STEP 4] Building model ...")

    model = Sequential([
        # Embedding layer: vocab_size -> EMBEDDING_DIM vector
        Embedding(input_dim=vocab_size,
                  output_dim=EMBEDDING_DIM,
                  input_length=SEQ_LENGTH),

        # LSTM Layer 1 (return_sequences=True because next LSTM ahe)
        LSTM(LSTM_UNITS, return_sequences=True),
        Dropout(DROPOUT_RATE),

        # LSTM Layer 2 (Bonus: deeper architecture)
        LSTM(LSTM_UNITS),
        Dropout(DROPOUT_RATE),

        # Output Layer
        Dense(vocab_size, activation="softmax")
    ])

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    model.summary()
    return model


# ─────────────────────────────────────────────
#  STEP 4: TRAINING
# ─────────────────────────────────────────────
def train_model(model, X, y):
    """
    Model train karto with:
      - EarlyStopping  : val_loss improve nahi zala tar stop
      - ModelCheckpoint: Best model save karto
    """
    print("\n[STEP 5] Training ...")

    callbacks = [
        EarlyStopping(monitor="val_loss",
                      patience=3,
                      restore_best_weights=True,
                      verbose=1),
        ModelCheckpoint(MODEL_FILE,
                        monitor="val_loss",
                        save_best_only=True,
                        verbose=1)
    ]

    history = model.fit(
        X, y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=1
    )

    print("\n[INFO] Training complete! Best model saved as:", MODEL_FILE)
    return history


# ─────────────────────────────────────────────
#  STEP 5: TEXT GENERATION
# ─────────────────────────────────────────────
def sample(predictions, temperature=1.0):
    """
    Temperature sampling:
      - temperature < 1.0 : Conservative (repetitive but safe)
      - temperature = 1.0 : Balanced
      - temperature > 1.0 : Creative (random pan chalti)
    """
    preds = np.asarray(predictions).astype("float64")
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_text(model, seed_text, char2idx, idx2char,
                  num_chars=300, temperature=0.8):
    """
    Seed text pasun new text generate karto.

    Params:
      seed_text   : Starting input (minimum SEQ_LENGTH chars)
      num_chars   : Kiti characters generate karayche
      temperature : Creativity level
    """
    vocab_size = len(char2idx)

    # Seed prepare: SEQ_LENGTH cha window
    seed_text = seed_text.lower()
    seed_text = re.sub(r"[^a-z\s]", "", seed_text)
    seed_text = re.sub(r"\s+", " ", seed_text).strip()

    if len(seed_text) < SEQ_LENGTH:
        # Seed chhoti asel tar pad karto
        seed_text = seed_text.rjust(SEQ_LENGTH)

    current_seq = seed_text[-SEQ_LENGTH:]
    generated = current_seq

    for _ in range(num_chars):
        # Current sequence -> integer indices
        x_pred = np.array([[char2idx.get(c, 0) for c in current_seq]])

        # Prediction
        preds = model.predict(x_pred, verbose=0)[0]

        # Sample next character
        next_idx  = sample(preds, temperature)
        next_char = idx2char[next_idx]

        generated   += next_char
        current_seq  = current_seq[1:] + next_char  # Sliding window

    return generated


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("   LSTM Text Generation - Shakespeare Dataset")
    print("=" * 60)

    # Step 1: Download
    download_dataset()

    # Step 2: Preprocess
    text, vocab, char2idx, idx2char = load_and_preprocess()
    vocab_size = len(vocab)

    # Vocab save karto (generation sathi nantar vapru)
    with open(VOCAB_FILE, "wb") as f:
        pickle.dump((vocab, char2idx, idx2char), f)
    print(f"[INFO] Vocabulary saved to '{VOCAB_FILE}'")

    # Step 3: Sequences
    X, y = create_sequences(text, char2idx)

    # Step 4: Model
    if os.path.exists(MODEL_FILE):
        print(f"\n[INFO] Saved model found: '{MODEL_FILE}'. Loading...")
        model = load_model(MODEL_FILE)
    else:
        model = build_model(vocab_size)
        # Step 5: Train
        train_model(model, X, y)

    # ─────────────────────────────────────────────
    #  STEP 6: Generate Text with Different Seeds
    # ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("   TEXT GENERATION - Sample Outputs")
    print("=" * 60)

    seed_inputs = [
        # Seed 1: Shakespeare pasun ghetle
        "shall i compare thee to a summers day",
        # Seed 2: Common English phrase
        "to be or not to be that is the question",
        # Seed 3: Love theme
        "love is a smoke made with the fume of sighs",
    ]

    temperatures = [0.5, 0.8, 1.2]  # Bonus: Different temperatures

    for temp in temperatures:
        print(f"\n{'─'*60}")
        print(f"  Temperature: {temp}")
        print(f"{'─'*60}")
        for seed in seed_inputs:
            print(f"\n  [Seed] → \"{seed[:50]}...\"")
            generated = generate_text(
                model, seed, char2idx, idx2char,
                num_chars=200, temperature=temp
            )
            print(f"  [Output]\n  {generated}\n")

    print("\n[DONE] Text generation complete!")
    print(f"       Model saved at : {MODEL_FILE}")
    print(f"       Vocab saved at : {VOCAB_FILE}")


# ─────────────────────────────────────────────
#  BONUS: Experiment Function
#  (Different architectures compare karto)
# ─────────────────────────────────────────────
def experiment_architectures(X, y, vocab_size):
    """
    BONUS: 3 different architectures train karto ani compare karto.
    """
    architectures = {
        "Shallow_1xLSTM": [
            Embedding(vocab_size, 64, input_length=SEQ_LENGTH),
            LSTM(128),
            Dense(vocab_size, activation="softmax")
        ],
        "Deep_2xLSTM": [
            Embedding(vocab_size, 64, input_length=SEQ_LENGTH),
            LSTM(128, return_sequences=True),
            Dropout(0.2),
            LSTM(128),
            Dense(vocab_size, activation="softmax")
        ],
        "Wide_2xLSTM_256": [
            Embedding(vocab_size, 128, input_length=SEQ_LENGTH),
            LSTM(256, return_sequences=True),
            Dropout(0.3),
            LSTM(256),
            Dropout(0.3),
            Dense(vocab_size, activation="softmax")
        ],
    }

    results = {}
    for name, layers in architectures.items():
        print(f"\n[EXPERIMENT] Architecture: {name}")
        m = Sequential(layers)
        m.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        cb = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
        h  = m.fit(X, y, batch_size=256, epochs=10,
                   validation_split=0.1, callbacks=[cb], verbose=0)

        best_val_loss = min(h.history["val_loss"])
        best_val_acc  = max(h.history["val_accuracy"])
        results[name] = {"val_loss": best_val_loss, "val_accuracy": best_val_acc}
        print(f"       Val Loss: {best_val_loss:.4f} | Val Accuracy: {best_val_acc:.4f}")

    print("\n[EXPERIMENT RESULTS]")
    print(f"{'Architecture':<25} {'Val Loss':>10} {'Val Accuracy':>14}")
    print("-" * 52)
    for name, r in results.items():
        print(f"{name:<25} {r['val_loss']:>10.4f} {r['val_accuracy']:>14.4f}")

    return results


if __name__ == "__main__":
    main()

    # BONUS experiment run karyasathi khali line uncomment kara:
    # text, vocab, char2idx, idx2char = load_and_preprocess()
    # X, y = create_sequences(text, char2idx)
    # experiment_architectures(X, y, len(vocab))
