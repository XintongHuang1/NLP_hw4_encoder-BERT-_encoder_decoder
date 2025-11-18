import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        # TODO
        # Build tokenizer and load examples for the requested split
        self.data_folder = data_folder
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        self.examples = []
        self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        # TODO
        # Load natural language and SQL lines, then tokenize and store as integer id lists.
        nl_path = os.path.join(data_folder, f"{split}.nl")
        sql_path = os.path.join(data_folder, f"{split}.sql")

        nl_lines = []
        sql_lines = []
        if os.path.exists(nl_path):
            with open(nl_path, 'r', encoding='utf-8') as f:
                nl_lines = [l.strip() for l in f.readlines() if l.strip()]

        if os.path.exists(sql_path):
            with open(sql_path, 'r', encoding='utf-8') as f:
                sql_lines = [l.strip() for l in f.readlines() if l.strip()]

        # Load alignment dictionary for entity normalization
        alignment_dict = {}
        alignment_path = os.path.join(data_folder, 'alignment.txt')
        if os.path.exists(alignment_path):
            with open(alignment_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if '\t' in line:
                        parts = line.strip().split('\t')
                        if len(parts) == 2:
                            src, tgt = parts
                            alignment_dict[src.lower()] = tgt

        # If sql_lines is empty (e.g., test split) we still populate examples with encoder-only items
        for i, nl in enumerate(nl_lines):
            # Use standard prompt format: "Translate the question to SQL: {question}"
            prompted_nl = f"Translate the question to SQL: {nl}"
            enc_ids = tokenizer.encode(prompted_nl, add_special_tokens=True, max_length=512, truncation=True)
            if i < len(sql_lines):
                tgt_ids = tokenizer.encode(sql_lines[i], add_special_tokens=True, max_length=512, truncation=True)
                self.examples.append((enc_ids, tgt_ids))
            else:
                # test split: no targets available
                self.examples.append((enc_ids, None))
    
    def __len__(self):
        # TODO
        return len(self.examples)

    def __getitem__(self, idx):
        # TODO
        enc_ids, tgt_ids = self.examples[idx]
        return enc_ids, tgt_ids

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # TODO
    # batch is a list of tuples (enc_ids, tgt_ids)
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    
    enc_seqs = [torch.tensor(x[0], dtype=torch.long) for x in batch]
    tgt_seqs = [torch.tensor(x[1], dtype=torch.long) for x in batch]

    # Pad encoder sequences
    enc_padded = pad_sequence(enc_seqs, batch_first=True, padding_value=PAD_IDX)
    enc_mask = (enc_padded != PAD_IDX).long()

    # Pad target sequences
    tgt_padded = pad_sequence(tgt_seqs, batch_first=True, padding_value=PAD_IDX)

    # Build decoder inputs by shifting targets to the right
    # Use the decoder_start_token_id (0 for T5) as the first token
    decoder_start_token_id = tokenizer.pad_token_id  # T5 uses pad_token_id as decoder start
    decoder_inputs = torch.full_like(tgt_padded, PAD_IDX)
    decoder_inputs[:, 0] = decoder_start_token_id
    if tgt_padded.size(1) > 1:
        decoder_inputs[:, 1:] = tgt_padded[:, :-1]

    # initial decoder input token (first column) returned for evaluation purposes
    initial_decoder_inputs = decoder_inputs[:, 0].unsqueeze(1)

    return enc_padded, enc_mask, decoder_inputs, tgt_padded, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # TODO
    # batch is a list of tuples (enc_ids, None)
    enc_seqs = [torch.tensor(x[0], dtype=torch.long) for x in batch]
    enc_padded = pad_sequence(enc_seqs, batch_first=True, padding_value=PAD_IDX)
    enc_mask = (enc_padded != PAD_IDX).long()

    # initial decoder input token is PAD_IDX for all examples (shape Bx1)
    initial_decoder_inputs = torch.full((enc_padded.size(0), 1), PAD_IDX, dtype=torch.long)
    return enc_padded, enc_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # TODO
    # Load lines for prompting experiments: train/dev/test .nl and .sql files
    train_x = []
    train_y = []
    dev_x = []
    dev_y = []
    test_x = []

    train_nl = os.path.join(data_folder, 'train.nl')
    train_sql = os.path.join(data_folder, 'train.sql')
    dev_nl = os.path.join(data_folder, 'dev.nl')
    dev_sql = os.path.join(data_folder, 'dev.sql')
    test_nl = os.path.join(data_folder, 'test.nl')

    if os.path.exists(train_nl):
        train_x = load_lines(train_nl)
    if os.path.exists(train_sql):
        train_y = load_lines(train_sql)
    if os.path.exists(dev_nl):
        dev_x = load_lines(dev_nl)
    if os.path.exists(dev_sql):
        dev_y = load_lines(dev_sql)
    if os.path.exists(test_nl):
        test_x = load_lines(test_nl)

    return train_x, train_y, dev_x, dev_y, test_x