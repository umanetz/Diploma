import os.path
import numpy as np
import pandas as pd

import torch
from torch import nn
from  torch.utils.data import Dataset


class TextDataset(Dataset):

    def __init__(self, data, prob=None):
        self.data = data
        self.prob = None
        if prob is not None:
          self.prob = prob.set_index('id')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data.iloc[idx][np.arange(102)].values # извлекаем индексы слов 
        target = self.data.iloc[idx]['target']
        id_tokens = self.data.iloc[idx]['id']
        sample = {'tokens':torch.tensor(tokens), 'id':  id_tokens, 'target': target}
        
        if self.prob is not None:
            probs = self.prob.loc[id_tokens][np.arange(80)].values # извлекаем соответсвующие логиты
            sample['bert_prob'] = torch.tensor(probs)
        return sample
    
    
def raw_data(data_path, load_probs=False):
    # Загружаем тексты в виде инедексов слов
    train_path = os.path.join(data_path, "tokens/tokens_train.csv")
    valid_path = os.path.join(data_path, "tokens/tokens_dev.csv")
    test_path = os.path.join(data_path, "tokens/tokens_test.csv")

    train_data = pd.read_csv(train_path).iloc[:]
    valid_data = pd.read_csv(valid_path)
    test_data = pd.read_csv(test_path)
    if not load_probs:
        return TextDataset(train_data), TextDataset(valid_data), TextDataset(test_data)
    else:
        # Загружаем логиты, если модель Knowledge distillation
        train_probs = pd.read_csv(os.path.join(data_path,'logits/logits_train.csv'))
        valid_probs = pd.read_csv(os.path.join(data_path,'logits/logits_dev.csv'))
        test_probs  = pd.read_csv(os.path.join(data_path,'logits/logits_test.csv'))
        return TextDataset(train_data, train_probs), TextDataset(valid_data, valid_probs), TextDataset(test_data, test_probs)
    
    
