# <img src="misc/oneruler.png" alt="ONERULER" width="35" height="35"> OneRuler
`Paper`: One ruler to measure them all: Benchmarking multilingual long-context language models 

`Authors`: Yekyung Kim, Jenna Russell, Marzena Karpinska, Mohit Iyyer

ONERULER is a multilingual benchmark designed to evaluate long-context language models across 26 languages. ONERULER adapts the English-only [RULER](https://arxiv.org/pdf/2404.06654) benchmark by including seven synthetic tasks that test both retrieval and aggregation, including new variations of the “needle-in-a-haystack” task that allow for the possibility of a nonexistent needle. We translate English instructions for each task and then collaborating with native speakers to translate them into 25 additional languages and experiment with both 5 open source and 2 closed model.

This code is based on [RULER's Repo](https://github.com/NVIDIA/RULER). 

![Micro-accuracy across context-lengths and languages for all NIAH tasksk](./misc/heatmap.png)

## Run Data Generation


