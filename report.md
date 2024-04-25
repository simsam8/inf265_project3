---
title: 'Project 3: Sequence models'
author:
- Simon Vedaa
- Sebastian Røkholt
numbersections: true
header-includes: |
    \usepackage{caption}
    \usepackage{subcaption}
---

# Introduction

# Embedding

## Approach and design choices

Tokenization

- Using torchtext basic-english tokenizer
- Exclude tokens of names, digits, and spaces


Dataset

- Context size refers to amount of tokens on each side of target.
    total context becomes 2*context_size
- Unknown tokens and punctuations are excluded from targets.

## Training

-- TODO --
Explain training procedure

## Results 

-- TODO --
Report on test performance, cosine similarity, embedding Visualization, and embedding arithmetic.

![Training and validation loss of selected CBOW model](images/embedding_loss.png)

![Training and validation accuracy of selected CBOW model](images/embedding_accuracy.png)

### Test performance

### Cosine similarity

### Visualization of embedding space

# Conjugation of _have_ and _be_

## Approach and design choices

- Dataset is based on embedding dataset. 
- Only pairs where target is a conjugation of _be_ or _have_ are kept.
- Context size/sequence length refers to the total size of context.

## Training

-- TODO -- 
Explain training, and parameter search

## Results

-- TODO -- 
Report on performance and training/performance time of models

![Training and validation loss of selected model](images/conjugation_loss.png)

![Training and validation accuracy of selected model](images/conjugation_loss.png)

![Average training times for architectures](images/conjugation_training_times.png)

# Text generation

## Approach and design choices

## Training

## Results

### Examples of generated sequences

