#!/bin/bash

# create 1000 examples
# head -n 1342 ~/downloads/news.2007.cs.shuffled.deduped | python gen.py > newscrawl.csv

python eval.py newscrawl.csv blank-match.csv --generate=blank --select=match
python eval.py newscrawl.csv blank-score.csv --generate=blank --select=score
python eval.py newscrawl.csv end-match.csv --generate=end --select=match
python eval.py newscrawl.csv end-score.csv --generate=end --select=score
python eval.py newscrawl.csv predict-match.csv --generate=predict --select=match
python eval.py newscrawl.csv predict-score.csv --generate=predict --select=score

python eval.py manual.csv manual-blank-match.csv --generate=blank --select=match
python eval.py manual.csv manual-blank-score.csv --generate=blank --select=score
python eval.py manual.csv manual-end-match.csv --generate=end --select=match
python eval.py manual.csv manual-end-score.csv --generate=end --select=score
python eval.py manual.csv manual-predict-match.csv --generate=predict --select=match
python eval.py manual.csv manual-predict-score.csv --generate=predict --select=score
