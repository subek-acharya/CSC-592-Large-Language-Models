import numpy as np
import torch
from MyNLPDataSet import MyNLPDataSet
from WikitextDataset import WikitextDataset
from torch.utils.data import DataLoader
import gzip

def cycle(loader):
    while True:
        for data in loader:
            yield data

def get_loaders_enwiki8(seq_len, batch_size):
    # ---------prepare enwik8 data-----------
    with gzip.open('./data/enwik8.gz') as file:
        data = np.fromstring(file.read(int(95e6)), dtype = np.uint8)
        data_train, data_val = map(torch.from_numpy, np.split(data, [int(90e6)]))

    train_dataset = MyNLPDataSet(data_train, seq_len)
    val_dataset   = MyNLPDataSet(data_val, seq_len)
    train_loader  = cycle(DataLoader(train_dataset, batch_size = batch_size))
    val_loader    = cycle(DataLoader(val_dataset, batch_size = batch_size))
    return train_loader, val_loader, val_dataset

def get_loaders_wikitext_103(tokenizer, seq_len, batch_size):
    file_path_train = "data/wikitext-103/wiki.train.tokens"
    file_path_valid = "data/wikitext-103/wiki.valid.tokens"
    file_path_test = "data/wikitext-103/wiki.test.tokens"
    train_dataset = WikitextDataset(tokenizer, file_path_train,"TRAIN", seq_len=seq_len)
    val_dataset = WikitextDataset(tokenizer, file_path_valid,"VALID", seq_len=seq_len)
    test_dataset = WikitextDataset(tokenizer, file_path_test,"TEST", seq_len=seq_len)
    train_loader = cycle(DataLoader(train_dataset, batch_size=batch_size, shuffle=True))
    val_loader = cycle(DataLoader(val_dataset, batch_size=batch_size, shuffle=False))
    test_loader = cycle(DataLoader(test_dataset, batch_size=batch_size, shuffle=False))
    return train_loader, val_loader, test_loader, val_dataset
