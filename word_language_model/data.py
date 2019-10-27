import os
from io import open
import torch

class Corpus(object):
    def __init__(self, path):
        
        train_graph = '/home/soumyar/dataExpOneTrain.p'
        val_graph='/home/soumyar/dataExpOneVal.p'
        test_graph='/home/soumyar/dataExpOneTest.p'

        f = open(train_graph, 'rb')
        data= load_func(f)
        f.close()

        f=open(val_graph,'rb')
        data_val=load_func(f)
        f.close()

        f=open(test_graph, 'rb')
        data_test=load_func(f)
        f.close()
        
        ##reshape to be 1by 48 and lump them all together!!
        
        self.train = data
        self.valid = data_val
        self.test = data_test

