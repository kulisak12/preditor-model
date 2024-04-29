# Preditor model

A model used by [Preditor](https://github.com/kulisak12/preditor).

Serves as the backend for typing assistance.
Provides completions while typing, even in the middle of a sentence.
For Czech, it allows the user to substitute a word in a sentence
and automatically adjusts the forms of the surrounding words.

Uses an existing large language model without any additional fine-tuning.
A single language model handles all of the tasks mentioned above.

## Quick start

```bash
pip install -e .
# development server
flask --app preditor.server run -p 3000
# production server
gunicorn preditor.server:app -b localhost:3000
```

If your device does not have a GPU, or does not run the latest CUDA, you need to
[install a different version of PyTorch](https://pytorch.org/get-started/locally/).

If the install of `fasttext` fails, use `fasttext-wheel` instead.

### Configuration

The model can be configured using environment variables.
The recommended way is to use a `.env` file in the root directory of the project.

- `PREDITOR_MODEL_PATH`: Path to the model, either local or on HuggingFace.
- `PREDITOR_FASTTEXT_PATH`: Path to the FastText model.
- `PREDITOR_TAGGER_PATH`: Path to the MorphoDiTa tagger.

```bash
PREDITOR_MODEL_PATH=BUT-FIT/CSTinyLlama-1.2B
PREDITOR_FASTTEXT_PATH=/home/user/preditor-model/models/lid.176.ftz
PREDITOR_TAGGER_PATH=/home/user/preditor-model/models/czech-morfflex2.0-pdtc1.0-220710.tagger
```

### Models

Preditor needs a FastText model for language identification
and a MorphoDiTa tagger for morphological analysis.

The FastText model can be downloaded from the
[FastText website](https://fasttext.cc/docs/en/language-identification.html).
The MorphoDiTa tagger can be downloaded from the
[LINDAT repository](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-4794).

We provide a utility script `download-models.sh` that downloads the models
to the `models` directory and configures the environment variables in the `.env` file.

## Requests

The server listens for POST requests.
Both the request and the response are in JSON format.

### Suggestions

Suggestions use a unified format that covers both
prediction (at the end of the text) and infilling (in the middle of the sentence).
The configuration is not required.

```json
{
    "before_cursor": "This is ",
    "after_cursor": "text to complete.",
    "prediction_config": {
        "max_length": 20,
        "confidence": 5.0
    },
    "infilling_config": {
        "max_length": 8,
        "num_variants": 5
    }
}
```

The response has the following format.

```json
{
    "output": "the "
}
```

### Substitution

Substitution uses a different format.
Once again, the configuration is not required.

```json
{
    "before_old": "Mám modré ",
    "old": "kolo",
    "after_old": ", které se mi líbí.",
    "replacement": "barvu",
    "config": {
        "min_variants": 2,
        "relax_count": 8,
        "pool_factor": 5,
        "lp_alpha": 0.5
    }
}
```

The response has the following format.

```json
{
    "output": "Mám modrou barvu, která se mi líbí."
}
```

### Errors

If an error occurs, the response has the following format.
The `details` field may contain additional information about the error,
and its format is not standardized.

```json
{
    "error": "An error occurred.",
    "details": {}
}
```
