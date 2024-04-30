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

![Cosine similarity matrix](images/similarity_matrix.png)

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

Time is taken on the whole training function,
which includes computing loss and accuracy for both training and validation.

![Average training times for architectures](images/conjugation_training_times.png)

# Text generation

## Approach and design choices

## Training

## Results

![Training and validation loss of selected text generation model](images/text_generation_loss.png)

![Training and validation accuracy of selected text generation model](images/text_generation_accuracy.png)

### Examples of generated sequences

