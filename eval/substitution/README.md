# Substitution Evaluation

This directory contains evaluation scripts and a dataset for the substitution task.

## Evaluation

To run evaluation on the dataset `manual.csv` with the strategy `cache`
and save it to `cache.csv`, use the following command:

```bash
python eval.py manual.csv cache.csv --strategy=cache
```

You may also pass `--min-variants`, `--relax-count`, `--pool-factor`, or `--lp-alpha`
to configure the generation.
The switch `--results-only` skips the generation and only evaluates the existing results.

## Dataset

We provide a dataset `manual.csv` with 100 manually created examples.
We adapt sentences from the
[NewsCrawl dataset](https://data.statmt.org/news-crawl/cs/).

We only used sentences shorter than about 100 characters.
Longer sentences take too long to evaluate and may run out of memory.
The examples are created such that some change always occurs
and such that the answer is mostly clear.

There are 20 examples for each of the following modifications:

- change in number
- change in gender
- change in gender and number
- change in person
- change in person and number
