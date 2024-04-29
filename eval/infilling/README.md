# Infilling Evaluation

This directory contains evaluation scripts and datasets for the infilling task.

## Evaluation

To run evaluation on the dataset `manual.csv`
with the generation strategy `blank` and the selection strategy `score`
and save it to `blank-score.csv`, use the following command:

```bash
python eval.py manual.csv blank-score.csv --generate=blank --select=score
```

You may also pass `--max-length` or `--num-variants` to configure the generation.
The switch `--results-only` skips the generation and only evaluates the existing results.

## Datasets

The datasets are in Czech.

### NewsCrawl

We use the [NewsCrawl dataset](https://data.statmt.org/news-crawl/cs/)
to create testing data for the infilling task.

We only keep examples with 8 or more words.
We split each example into words and remove 1 to 3 of them.

You can create the testing data by running the following command.
Note that the number of examples will be lower due to the filtering.

```bash
head -n 1000 news.2007.cs.shuffled.deduped | python gen.py > newscrawl.csv
```

### Manual

We also provide a tiny dataset, `manual.csv`, with manually created examples.
The examples are created such that the infilling depends on the text both before and after the gap
and such that there is a mostly clear answer.

The sentences come from the
[Dataset for semantic text similarity](https://air.kiv.zcu.cz/datasets/sts-ctk).
