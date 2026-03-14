# lstm-text-generation
LSTM-based text generation using Shakespeare dataset
# LSTM Text Generation 🎭

A character-level text generator built with LSTM neural networks,
trained on Shakespeare's Complete Works.

## Dataset
- **Source:** [Project Gutenberg - Shakespeare's Complete Works](https://www.gutenberg.org/files/100/100-0.txt)
- **Size:** ~200,000 characters used for training

## Model Architecture
- Embedding Layer (64 dimensions)
- LSTM Layer 1 (128 units) + Dropout (0.2)
- LSTM Layer 2 (128 units) + Dropout (0.2)
- Dense Output Layer (Softmax)

## How to Run

### Install dependencies
pip install tensorflow numpy requests

### Run the script
python lstm_text_generation.py

## Sample Output

**Seed:** "shall i compare thee to a summers day"
**Generated Text:**
shall i compare thee to a summers day
thou art more lovely and more temperate
rough winds do shake the darling buds...

## Results

| Epoch | Accuracy | Val Loss |
|-------|----------|----------|
| 1     | 22.5%    | 2.40     |
| 5     | 41.3%    | 1.86     |

## Tech Stack
- Python 3.x
- TensorFlow / Keras
- NumPy








Output:-
   TEXT GENERATION - Sample Outputs
============================================================

────────────────────────────────────────────────────────────
  Temperature: 0.5
────────────────────────────────────────────────────────────

  [Seed] → "shall i compare thee to a summers day..."
  [Output]
     shall i compare thee to a summers day a most which the didst love the prove the stranger the storm that do his graces of heaven with such that is not the chow her which perseafe the first for that do i know the will the grows the cworse


  [Seed] → "to be or not to be that is the question..."
  [Output]
   to be or not to be that is the question stame that his beautys in the countess are be from the though that be i some with shall be say to great helena the which fould the day is as you for the unjueming gay a say the crowe of that the crus


  [Seed] → "love is a smoke made with the fume of sighs..."
  [Output]
  e is a smoke made with the fume of sighs heart that that the countess beauty and truth in beautys of the world for the countess of the perpore that the count thou best that countess will this in the would that so carrot for the countess of


────────────────────────────────────────────────────────────
  Temperature: 0.8
────────────────────────────────────────────────────────────

  [Seed] → "shall i compare thee to a summers day..."

