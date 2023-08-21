# Classifying Multilingual Party Manifestos: Domain transfer across country, time, and genre

## About

This repository contains Jupyter-Notebooks for conducting the analyses the analysis from our paper and for reproducing the results. 

- Authors: M. Aßenmacher, N. Sauter, C. Heumann
- Contact: matthias [at] stat.uni-muenchen.de
- ArXiv: https://arxiv.org/abs/2307.16511

## Data

Since we are not allowed to share the data we extracted from the manifesto project data base, we provide the R-Script that allows recovering it.  
Please 

## Structure

    ├── LICENSE
    ├── README.md          <- This file.
    ├── manifesto.yml      <- The yml-file for creating the environment.
    ├── .gitgnore 
    │
    ├── notebooks          <- Notebooks for reproducing our analyses.
        |
        └── utils          <- Utilities and helper functions.
    │
    ├── results            <- Results of our analyses.
    │
    └── data               <- The R-Script for recovering the data used for the analyses.

## Models

All models we fine-tuned for this research project are available on huggingface: https://huggingface.co/assenmacher

They can either be used (at the example of our `distilbert-base-cased-manifesto-2018`) in the pipeline:

```
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-classification", model="assenmacher/distilbert-base-cased-manifesto-2018")
```

or more flexibly:

```
# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("assenmacher/distilbert-base-cased-manifesto-2018")
model = AutoModelForSequenceClassification.from_pretrained("assenmacher/distilbert-base-cased-manifesto-2018")
```
