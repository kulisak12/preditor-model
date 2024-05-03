#!/bin/bash

python eval.py manual.csv simple-00.csv --strategy=simple --lp-alpha=0.0
python eval.py manual.csv simple-05.csv --strategy=simple --lp-alpha=0.5
python eval.py manual.csv simple-10.csv --strategy=simple --lp-alpha=1.0
python eval.py manual.csv cache-00.csv --strategy=cache --lp-alpha=0.0
python eval.py manual.csv cache-05.csv --strategy=cache --lp-alpha=0.5
python eval.py manual.csv cache-10.csv --strategy=cache --lp-alpha=1.0

