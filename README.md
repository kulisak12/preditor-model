# Preditor model

A model used by [Preditor](https://github.com/kulisak12/preditor).

Provides completions while typing, even in the middle of a sentence.
Uses existing large language models without any additional fine-tuning.

## Quick start

```bash
pip install -e .
flask --app preditor.server run -p 3000
```

If your device does not have a GPU, you need to
[install the CPU version of PyTorch](https://pytorch.org/get-started/locally/).

### Configuration

The model can be configured using environment variables.
The recommended way is to use a `.env` file in the root directory of the project.

- `PREDITOR_MODEL_PATH`: Path to the model, either local or on HuggingFace.
- `PREDITOR_MAX_LENGTH`: The number of new tokens to generate.

Other options can be found in the [configuration provider](src/preditor/config.py).

## Requests

The server listens for POST requests with the following JSON body.
Cursor is the position of the cursor in the text.

```json
{
    "text": "The text to complete.",
    "cursor": 21
}
```

The response has the following format.

```json
{
    "prediction": "The predicted text."
}
```
