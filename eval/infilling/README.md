# Infilling Evaluation

This directory contains evaluation scripts and datasets for the infilling task.
The datasets are in Czech.

## Evaluation

To run evaluation on the dataset `manual.csv`
with the generation strategy `blank` and the selection strategy `score`,
use the following command:

```bash
python eval.py manual.csv --generate=blank --select=score
```

You may also pass `--debug` to print the generated examples,
or `--max-length` and `--num-variants` to configure the generation.

## Datasets

### NewsCrawl

We use the [NewsCrawl dataset](https://data.statmt.org/news-crawl/cs/)
to create testing data for the infilling task.

We split each example into words and remove 1 to 3 of them.
We only keep examples with 8 or words.

You can create the testing data by running the following command.
Note that the number of examples will be lower due to the filtering.

```bash
head -n 1000 news.2007.cs.shuffled.deduped | python gen.py > newscrawl.csv
```

### Manual

We also provide a tiny dataset `manual.csv` with manually created examples.
The examples are created such that the infilling depends on the text both before and after the gap,
and such that there is a mostly clear answer.

The sentences come from the
[Dataset for semantic text similarity](https://air.kiv.zcu.cz/datasets/sts-ctk).
