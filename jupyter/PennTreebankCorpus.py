from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
from collections import Counter
import torch
import os


class Corpus:
    """
    self.load_datasets
    self.preprocess
        tokenize
        build_vocab
        to_ids
        batchify
    """
    def __init__(self, root_dir='data'):
        self.tokenizer = get_tokenizer('basic_english')
        self.root_dir = root_dir

    def load_datasets(self):
        self.train_raw_text = list(self.load_txt('train'))
        self.val_raw_text = list(self.load_txt('valid'))
        self.test_raw_text = list(self.load_txt('test'))

    def load_txt(self, mode):
        file_name = os.path.join(self.root_dir, f'ptb.{mode}.txt')
        yield from self._load_txt(file_name)

    def _load_txt(self, file_name):
        with open(file_name, encoding='utf-8', mode='r') as f:
            for line in f:
                yield line

    def preprocess(self, batch_size):
        print('1. Tokenization')
        print('---------------------')
        print('train: tokenizing ...')
        train_tokenized = self._tokenize(self.train_raw_text)
        print('val: tokenizing ...')
        val_tokenized = self._tokenize(self.val_raw_text)
        print('test: tokenizing ...')
        test_tokenized = self._tokenize(self.test_raw_text)
        print('Done!\n\n')

        print('2. Build Vocab')
        counter = Counter()
        for tokenized in train_tokenized:
            counter.update(tokenized)
        
        self.vocab = Vocab(
            counter, min_freq=1,
            specials=['<unk>', '<sos>', '<eos>'])
        print('Done!\n\n')

        print('3. To ids')
        train_data = torch.cat(list(self._to_ids(train_tokenized)))
        val_data = torch.cat(list(self._to_ids(val_tokenized)))
        test_data = torch.cat(list(self._to_ids(test_tokenized)))
        print('Done!\n\n')

        print('4. batchify')
        train_data = batchify(train_data, batch_size=batch_size)
        val_data = batchify(val_data, batch_size=batch_size)
        test_data = batchify(test_data, batch_size=batch_size)
        print('Done!\n\n')

        return train_data, val_data, test_data

    def _tokenize(self, raw_texts):
        data = [self.tokenizer(raw_text)
                for raw_text in raw_texts]
        return data

    def _to_ids(self, tokenized_texts):
        for tokenized in tokenized_texts:
            if len(tokenized) < 10:
                continue
            ids = torch.tensor(
                [self.vocab['<sos>']] + 
                self.vocab.lookup_indices(tokenized) + 
                [self.vocab['<eos>']]
            )
            yield ids

    def vocab_size(self):
        return len(self.vocab)


def batchify(data, batch_size):
    n_batch = data.size(0) // batch_size
    data = data.narrow(0, 0, n_batch * batch_size)
    data = data.view(batch_size, -1).t().contiguous()
    return data
