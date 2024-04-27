#!/bin/bash
set -ueo pipefail

MODELS_DIR="models"
FASTTEXT="lid.176.ftz"
TAGGER="czech-morfflex2.0-pdtc1.0-220710"

mkdir -p "$MODELS_DIR"
cd "$MODELS_DIR"

# FastText
wget "https://dl.fbaipublicfiles.com/fasttext/supervised-models/$FASTTEXT"

# MorphoDiTa
wget "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-4794/$TAGGER.zip"
unzip "$TAGGER.zip"
mv "$TAGGER/$TAGGER.tagger" .
rm -rf "$TAGGER.zip" "$TAGGER"

cd ..

echo "PREDITOR_FASTTEXT_PATH=$(realpath $MODELS_DIR/$FASTTEXT)" >> .env
echo "PREDITOR_TAGGER_PATH=$(realpath $MODELS_DIR/$TAGGER.tagger)" >> .env
