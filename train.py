import numpy as np
import random
import torch
import matplotlib.pylab as plt
import torchmetrics
import torch.nn as nn
from transformers import BertTokenizer, BertConfig, BertForMaskedLM, BertForNextSentencePrediction,  BertForQuestionAnswering
from transformers import BertModel
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from functools import partial  # partial()函数可以用来固定某些参数值，并返回一个新的callable对象
import pdb
from transformers import BertForTokenClassification