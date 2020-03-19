# coding: utf-8

import re
import mxnet as mx
import gluonnlp as nlp
from gluonnlp.data import TSVDataset, SacreMosesTokenizer
from nltk.corpus import stopwords

tokenizer = SacreMosesTokenizer()
LABELS = ['0', '1']
stopwords =  set(stopwords.words('english'))
word_exp = re.compile(r'[\w]+')


def load_dataset(train_file, val_file, test_file, max_length=20):
    """
    Inputs: training, validation and test files in TSV format
    Outputs: vocabulary (with attached embedding), training, validation and test datasets ready for neural net training
    """
    transformer = BasicTransform(labels=LABELS)
    train_array = TSVDataset(train_file)
    val_array = TSVDataset(val_file)
    test_array = TSVDataset(test_file)
    
    vocabulary  = build_vocabulary(train_array, val_array, test_array)
    train_dataset = preprocess_dataset(train_array, transformer, vocabulary, max_length)
    val_dataset = preprocess_dataset(val_array, transformer, vocabulary, max_length)
    test_dataset = preprocess_dataset(test_array, transformer, vocabulary, max_length)
    return vocabulary, train_dataset, val_dataset, test_dataset


def build_vocabulary(tr_array, val_array, tst_array):
    """
    Inputs: arrays representing the training, validation and test data
    Outputs: vocabulary (Tokenized text as in-place modification of input arrays or returned as new arrays)
    """
    all_tokens = []
    for i, instance in enumerate(tr_array):
        id_num, label, text = instance
        tokens = filter_text(text)
        all_tokens.extend(tokens)
    counter = nlp.data.count_tokens(all_tokens)
    vocab = nlp.Vocab(counter)
    return vocab


def _preprocess(x, vocab, max_len):
    """
    Inputs: data instance x (tokenized), vocabulary, maximum length of input (in tokens)
    Outputs: data mapped to token IDs, with corresponding label
    """
    id_num, label, text_tokens = x
    tokens = filter_text(text_tokens)
    data = vocab[tokens]
    if len(data) > max_len:
        data = data[:max_len]
    return label, data

def filter_text(text_tokens):
    """ """
    basic_tokenize = text_tokens.split(" ")
    text = " ".join([token.lower() for token in basic_tokenize if not "http" in token.lower() and not token.startswith(("#","@"))])
    tokenized = re.findall(word_exp,text)
    return [token for token in tokenized if token not in stopwords]

def preprocess_dataset(dataset, transformer, vocab, max_len):
    preprocessed_dataset = [(transformer(*(_preprocess(x, vocab, max_len)))) for x in dataset if x[1] in LABELS]
    return preprocessed_dataset


class BasicTransform(object):
    """
    This is a callable object used by the transform method for a dataset. It will be
    called during data loading/iteration.  

    Parameters
    ----------
    labels : list string
        List of the valid strings for classification labels
    max_len : int, default 64
        Maximum sequence length - longer seqs will be truncated and shorter ones padded
    
    """
    def __init__(self, labels, max_len=20):
        self._max_seq_length = max_len
        self._label_map = {}
        for (i, label) in enumerate(labels):
            self._label_map[label] = i
    
    def __call__(self, label, data):
        label_id = self._label_map[label]
        padded_data = data + [0] * (self._max_seq_length - len(data))
        return mx.nd.array(padded_data, dtype='int32'), mx.nd.array([label_id], dtype='int32')
        
