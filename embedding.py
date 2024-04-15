import os
from os.path import isfile
import re
import glob
import torch
from torch.utils.data import TensorDataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


TOKENIZER_EN = get_tokenizer("basic_english")
PATH_GENERATED = "./generated/"
MIN_FREQ = 100


def read_files(datapath="./data/data_train/"):
    files = glob.glob(datapath + "*.txt")

    lines = []
    for f_name in files:
        with open(f_name) as f:
            lines += f.readlines()
    return lines


def tokenize(lines, tokenizer=TOKENIZER_EN):
    list_text = []
    for line in lines:
        list_text += tokenizer(line)
    return list_text


def yield_tokens(lines, tokenizer=TOKENIZER_EN):
    no_digits = "\w*[0-9]+\w*"
    no_names = "\w*[A-Z]+\w*"
    no_spaces = "\s+"

    for line in lines:
        line = re.sub(no_digits, " ", line)
        line = re.sub(no_names, " ", line)
        line = re.sub(no_spaces, " ", line)
        yield tokenizer(line)


def count_freqs(words, vocab):
    freqs = torch.zeros(len(vocab), dtype=torch.int)
    for w in words:
        freqs[vocab[w]] += 1
    return freqs


def create_vocabulary(lines, min_freq=MIN_FREQ):
    vocab = build_vocab_from_iterator(
        yield_tokens(lines), min_freq=min_freq, specials=["<unk>"]
    )
    vocab.append_token("i")
    vocab.set_default_index(vocab["<unk>"])
    return vocab


# Tokenize texts
# Load tokenized texts if they are generated
# else, create it and save it

if os.path.isfile(PATH_GENERATED + "words_train.pt"):
    words_train = torch.load(PATH_GENERATED + "words_train.pt")
    words_val = torch.load(PATH_GENERATED + "words_val.pt")
    words_test = torch.load(PATH_GENERATED + "words_test.pt")
else:
    lines_book_train = read_files("./data/data_train/")
    lines_book_val = read_files("./data/data_val/")
    lines_book_test = read_files("./data/data_test/")

    words_train = tokenize(lines_book_train)
    words_val = tokenize(lines_book_val)
    words_test = tokenize(lines_book_test)

    torch.save(words_train, PATH_GENERATED + "words_train.pt")
    torch.save(words_val, PATH_GENERATED + "words_val.pt")
    torch.save(words_test, PATH_GENERATED + "words_test.pt")


# Create vocabulary

VOCAB_FNAME = "vocabulary.pt"

if os.path.isfile(PATH_GENERATED + VOCAB_FNAME):
    vocab = torch.load(PATH_GENERATED + VOCAB_FNAME)
else:
    vocab = create_vocabulary(lines_book_train, min_freq=MIN_FREQ)
    torch.save(vocab, PATH_GENERATED + VOCAB_FNAME)


# Analysis

VOCAB_SIZE = len(vocab)
print("Total number of words in the training dataset:     ", len(words_train))
print("Total number of words in the validation dataset:   ", len(words_val))
print("Total number of words in the test dataset:         ", len(words_test))
print("Number of distinct words in the training dataset:  ", len(set(words_train)))
print("Number of distinct words kept (vocabulary size):   ", VOCAB_SIZE)

freqs = count_freqs(words_train, vocab)
# print(
#     "occurences:\n",
#     [(f.item(), w) for (f, w) in zip(freqs, vocab.lookup_tokens(range(VOCAB_SIZE)))],
# )

# Define targets


def compute_label(w):
    """
    helper function to defin MAP_TARGET

    - 0 = 'unknown word' i.e. the '<unk>' token
    - 1 = 'punctuation'
    - 2 = 'is an actual word'
    """

    if w in ["<unk>"]:
        return 0
    elif w in [",", ".", "(", ")", "?", "!"]:
        return 1
    else:
        return 2


# true labels for this taks:
MAP_TARGET = {
    vocab[w]: compute_label(w) for w in vocab.lookup_tokens(range(VOCAB_SIZE))
}

CONTEXT_SIZE = 2

# define context / target pairs


def create_dataset(text, vocab, context_size=CONTEXT_SIZE, map_target=MAP_TARGET):
    """
    Create a pytorch dataset of context / target pairs from a text
    """

    n_text = len(text)
    n_vocab = len(vocab)

    if map_target is None:
        map_target = {i: i for i in range(n_vocab)}

    txt = [vocab[w] for w in text]

    contexts = []
    targets = []

    for i in range(context_size, n_text - context_size):

        t = txt[i]
        c = txt[i - context_size : i] + txt[i + 1 : i + context_size]
        targets.append(map_target[t])
        contexts.append(torch.tensor(c))

    # contexts of shape (N_dataset, contexts_size)
    # targets of shape (N_dataset)
    contexts = torch.stack(contexts)
    print(contexts[0])
    targets = torch.tensor(targets)
    print(targets[0])
    return TensorDataset(contexts, targets)


def load_dataset(words, vocab, fname):
    """
    Load dataset if already generated, otherwise, create it and save it.
    """
    if os.path.isfile(PATH_GENERATED + fname):
        dataset = torch.load(PATH_GENERATED + fname)
    else:
        dataset = create_dataset(words, vocab)
        torch.save(dataset, PATH_GENERATED + fname)
    return dataset


data_train = load_dataset(words_train, vocab, "data_train.pt")
data_val = load_dataset(words_val, vocab, "data_val.pt")
data_test = load_dataset(words_test, vocab, "data_test.pt")
