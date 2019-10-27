import os
from io import open
import torch

class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.p'))
        self.valid = self.tokenize(os.path.join(path, 'valid.p'))
        self.test = self.tokenize(os.path.join(path, 'test.p'))

