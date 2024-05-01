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

### Tokenization

For tokenization we are using the basic-english tokenizer from `torchtext`.
We exclude tokens which are digits, names, and spaces.
Only words that appear 100 times or more are used in our vocabulary.


### Dataset

The training dataset contains 2,684,706 words, and has 52,105 distinct words.
The defined vocabulary contains 1880 words. 
Many words then becomes unkown words, and could affect performance.

When creating the context, target pairs, we have used a context size of 6.
The context size refers to the number of tokens before and after the target.
The total context then 2*context_size.
Unkown tokens (<unk>), and punctuations are excluded from the targets.

### CBOW Architecture

We have defined two CWOB architectures, `CBOW` and `CBOWDeep`. Both take the vocab size, context size, and 
embedding dimension as parameters. The first layer is a `nn.Embedding` for both.

`CBOW` contains two fully connected layers after the embedding layer, while, `CBOWDeep` contains four.
Relu is used as the activation function, and the last layer is passed through a log_softmax layer.


## Training and Selection

-- TODO --
Explain training procedure

For training we are using `Adam` as the optimizer, and `nn.NLLLoss` as the loss function.
The weights for the vocabulary are passed to the loss function.

The function `src/train` is used for training in all three tasks.

We used a batch size of 64, and trained 15 epochs for every run.

| Learning rate | Embedding dimension |
| -------------- | --------------- |
| 0.001 | 16 |
| 0.001 | 20 |
| 0.008 | 16 |
| 0.008 | 20 |

We have implemented a simple grid search, where;
for each architectures, we train it for every defined hyperparameter combination.

The model with the highest accuracy is chosen.


## Results 

These were the chosen parameters and architecture --INSERT--

-- TODO --
Report on test performance, cosine similarity, embedding Visualization, and embedding arithmetic.

![Training and validation loss of selected CBOW model](images/embedding_loss.png)

![Training and validation accuracy of selected CBOW model](images/embedding_accuracy.png)

### Test performance

The chosen model got a test accuracy of --INSERT--

### Cosine similarity

![Cosine similarity matrix](images/similarity_matrix.png)

### Visualization of embedding space

--INSERT-- plots from tensorflow projector

# Conjugation of _have_ and _be_

## Approach and design choices

### Dataset

The conjugation dataset is based on the generated embedding dataset.
We only include targets that are _be_, _am_, _are_, _is_, _was_, _were_, _been_, _being_, _have_, _has_, _had_, _having_.

The context size/sequence length refers to the total size of context.

### Model architectures

We have defined three architectures for the conjugation task, `SimpleMLP`, `AttentionMLP`, and `ConjugationRNN`.
Each model has an embedding layer as its first layer, which is frozen during initalization.
Additionally each architecture includes a parameter for the max length of the input sequence.

The `SimplMLP` contains three fully connected layers after the embedding layer.
The first linear layer takes an input of size `embedding_dim*max_len`.
The last layer has an output size of 12, corresponding to the number of possible conjugations.
All sizes of the layers in between are adjustable.
Relu is used as activation function, and the output layer is not passed through any function.

The `AttentionMLP` contains a positional encoding layer, a multi-head attention layer,
and a fully connected layer.
The multi-head attention layer is implemented by chaining multiple `SingleHead` layer using 
`nn.ModuleList`, and then concatenating their outputs and passing them through a fully connected layer.
Number of heads and the size of key, query, and value matrices are adjustable.

The `ConjugationRNN` contains a single RNN layer and a fully connected layer. 
The size and number of hidden layers in the RNN are adjustable.

## Training

-- TODO -- 
Explain training, and parameter search

Here we are using `Adam` for the optimizer as well. `nn.CrossEntropyLoss` is used as the 
loss function.

We use the same approach for a simple grid search as in the previous task.
For each model architecture we train with all possible hyper parameter combinations.
Additionally we measure the average training time for each model architecture.

We used a batch size of 64, and trained 30 epochs for every run.

--INSERT-- hyper parameters 

SimpleMLP:

| l1 | l2 |
| -------------- | --------------- |
| 128 | 32 |
| 256 | 64 |
| 256 | 256 |


AttentionMLP:

| n_heads | W size |
| -------------- | --------------- |
| 4 | 8 |
| 8 | 16 |
| 16 | 20 |


ConjugationRNN:

| num_hidden | num_layers | dropout |
| --------------- | --------------- | --------------- |
| 8 | 4 | 0 |
| 16 | 8 | 0.1 |
| 20 | 16 | 0.1|


| Learning rate |
| ------------- |
| 0.008 |
| 0.001 |
| 0.0005 |


The model with the highest accuracy is chosen.


## Results

These were the chosen parameters and architecture --INSERT--

-- TODO -- 
Report on performance and training time of models

![Training and validation loss of selected model](images/conjugation_loss.png)

![Training and validation accuracy of selected model](images/conjugation_loss.png)

Time is taken on the whole training function,
which includes computing loss and accuracy for both training and validation.

![Average training times for architectures](images/conjugation_training_times.png)

# Text generation

## Approach and design choices

### Dataset

Generated from tokenized words.
Context is now before the target.

### Model architectures

-- TODO --
Write about text generation architectures

## Training

-- TODO -- 
Explain training, and parameter search

Here we are using `Adam` for the optimizer as well. `nn.CrossEntropyLoss` is used as the 
loss function.

We use the same approach for a simple grid search as in the previous tasks.
For each model architecture we train with all possible hyper parameter combinations.

We used a batch size of 64, and trained 20 epochs for every run.


| num_hidden | num_layers | dropout |
| --------------- | --------------- | --------------- |
| 8 | 4 | 0 |
| 16 | 8 | 0.1 |
| 20 | 16 | 0.1|

| Learning rate |
| ------------- |
| 0.008 |
| 0.001 |
| 0.01 |
| 0.0005 |

The model with the highest accuracy is chosen.

## Results

These were the chosen parameters and architecture --INSERT--

-- TODO -- 
Report on performance

![Training and validation loss of selected text generation model](images/text_generation_loss.png)

![Training and validation accuracy of selected text generation model](images/text_generation_accuracy.png)

### Examples of generated sequences

-- TODO --
Include some example input and output sequences

