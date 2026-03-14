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
